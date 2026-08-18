# -*- coding: utf-8 -*-
"""构建 P_gold 逐题 Gold evidence 上下文（只含原文证据，不含 gold_answer）。

角色定位
--------
P_gold 是 oracle 路径：把每题 verified evidence 的**原文引用片段**（span.text）
按 answer unit 顺序组装成单行上下文，直接喂给与 P0-P4 完全相同的回答模板。
本脚本是唯一的 P_gold 输入构建入口，**只新增数据产物，不修改 Step 3 /
固定执行器 / 契约 / prompt**，因此现有 system snapshot 保持有效。

事实源与映射
------------
- 事实源是 ``verified_evidence.jsonl``（Step 2.2 验收产出，全部 decision=accept），
  它比题集内嵌的 ``evidence_spans`` 更新更完整（实测 12 题 verified-only 多出
  1 个 span）。
- ``question_id -> verified candidate`` 映射按 ID 后缀规则：
  ``qv2-XXXX -> ec-v2-XXXX``。**禁止按行号对应**；映射失败直接报错。

硬约束（全部 fail-fast，一次全报）
----------------------------------
1. question_id 唯一；
2. 每个 question_id 找到唯一 verified candidate（双向：每个 candidate 也唯一）；
3. 每个 required answer unit 至少 1 个 evidence span；
4. 每个 AU 引用的每个 evidence group 至少 1 个 span（无空 group）；
5. span 的 chunk_id 必须存在于 chunks 文件；
6. ``chunk.content[char_start:char_end] == span.text``（切片与原文一致）；
7. 重复 span 按 ``(chunk_id, char_start, char_end)`` 去重；
8. **不使用 gold_answer / AU claim 生成上下文** —— 只拼接 span.text；
9. 上下文单行化：内部所有空白序列压缩为单个空格；
10. 上下文 token 数低于模型输入安全上限
    （``llm_max_length - max_response_tokens - TOKEN_RESERVE``）。

产物（output-dir 下）
---------------------
- ``p_gold_contexts.txt``：严格 80 行，第 i 行 = 题集第 i 题的单行上下文
  （Step_3 约定：行数 == 题目数 时按行一一对应，orig_idx 即题集顺序索引）。
- ``p_gold_contexts.jsonl``：审计文件，每题一条：question_id、context、spans、
  tokens、校验通过标记。
- ``p_gold_contexts_manifest.json``：题集哈希、证据集哈希、chunks 哈希、输出
  哈希、ID 顺序（== 题集顺序）、token 统计与全部校验计数。

token 计数
----------
默认在线调用 vLLM ``/tokenize``（hyperrag.qwen_tokenizer）取权威 Qwen 计数，
失败即 fail-fast（契约 units.token_unit = qwen_tokenizer_token，禁止静默回退
估算）。``--no-tokenize`` 时用 tiktoken cl100k_base 估算并在 manifest 标记
``token_source``，仅用于离线调试，不构成正式验收依据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# 契约要求 token 单位为权威 Qwen tokenizer；构建脚本只做上限校验，
# 估算标记只允许 --no-tokenize 调试路径出现。
_TOKEN_RESERVE = 1000  # 覆盖回答模板 + 逐题 query 文本的开销余量（保守）

# question_id 与 candidate_id 的 ID 后缀映射：qv2-XXXX <-> ec-v2-XXXX
_QID_PREFIX = "qv2-"
_CID_PREFIX = "ec-v2-"


class PGoldBuildError(Exception):
    """构建失败（校验不通过 / 输入结构不合法）。"""


@dataclass(frozen=True)
class SpanRef:
    """一条原文证据引用。"""
    span_id: str
    chunk_id: str
    char_start: int
    char_end: int
    text: str

    @property
    def dedup_key(self) -> Tuple[str, int, int]:
        return (self.chunk_id, self.char_start, self.char_end)


@dataclass
class QuestionContext:
    """单题的构建结果（纯内存，可测）。"""
    question_id: str
    context: str
    spans: List[SpanRef] = field(default_factory=list)
    tokens: Optional[int] = None


@dataclass
class BuildResult:
    """一次构建的全部产出 + 校验错误。errors 为空才可写盘。"""
    records: List[QuestionContext]
    errors: List[str]
    stats: Dict


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_questions(question_file: Path) -> List[dict]:
    """加载题集，校验 question_id 唯一且非空。返回按文件行序的题列表。"""
    if not question_file.exists():
        raise PGoldBuildError(f"题集文件不存在: {question_file}")
    qs: List[dict] = []
    seen: Dict[str, int] = {}
    with open(question_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("question_id")
            if not qid:
                raise PGoldBuildError(
                    f"题集第 {idx} 行缺少 question_id")
            qid = str(qid)
            if qid in seen:
                raise PGoldBuildError(
                    f"question_id 重复: {qid}（第 {seen[qid]} 行与第 {idx} 行）")
            seen[qid] = idx
            qs.append(obj)
    if not qs:
        raise PGoldBuildError(f"题集为空: {question_file}")
    return qs


def load_verified(verified_file: Path) -> Dict[str, dict]:
    """加载 verified evidence，按 candidate_id 索引；校验 candidate_id 唯一。"""
    if not verified_file.exists():
        raise PGoldBuildError(f"verified evidence 文件不存在: {verified_file}")
    recs: Dict[str, dict] = {}
    with open(verified_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("candidate_id")
            if not cid:
                raise PGoldBuildError(
                    f"verified 第 {idx} 行缺少 candidate_id")
            cid = str(cid)
            if cid in recs:
                raise PGoldBuildError(f"candidate_id 重复: {cid}")
            recs[cid] = obj
    if not recs:
        raise PGoldBuildError(f"verified evidence 为空: {verified_file}")
    return recs


def load_chunks(chunks_file: Path) -> Dict[str, dict]:
    """加载 chunk 库 {chunk_id: {"content": str, ...}}。"""
    if not chunks_file.exists():
        raise PGoldBuildError(f"chunks 文件不存在: {chunks_file}")
    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise PGoldBuildError(f"chunks 文件必须是 {{chunk_id: {{content}}}}: {chunks_file}")
    return data


def _map_candidate_to_question(questions: List[dict],
                               verified: Dict[str, dict]) -> Dict[str, str]:
    """建立 candidate_id -> question_id 的双向唯一映射（按 ID 后缀，非行号）。

    qv2-XXXX -> ec-v2-XXXX。校验：
    - 每个 candidate 的后缀必须对应题集中真实存在的 question_id；
    - 不存在"两个 candidate 映射同一 question"（后缀唯一性由构造保证，
      此处仍显式校验，防止 ID 命名异常）。
    """
    qids = {q["question_id"] for q in questions}
    mapping: Dict[str, str] = {}
    for cid in verified:
        if not cid.startswith(_CID_PREFIX):
            raise PGoldBuildError(
                f"candidate_id 命名不符合 {_CID_PREFIX}XXXX 约定: {cid}")
        suffix = cid[len(_CID_PREFIX):]
        qid = _QID_PREFIX + suffix
        if qid not in qids:
            raise PGoldBuildError(
                f"candidate {cid} 无法映射到题集中的 question_id（期望 {qid} 不存在）")
        if qid in mapping.values():
            raise PGoldBuildError(f"多个 candidate 映射到同一 question_id: {qid}")
        mapping[cid] = qid
    return mapping


def _align_answer_units(question: dict, verified_rec: dict) -> Tuple[List[dict], List[dict]]:
    """按索引对齐 question.answer_units 与 verified.answer_units。

    交叉校验：数量一致 + 每个 AU 的 claim == verified 的 statement。
    结构漂移（顺序/内容不一致）直接报错，防止错位拼接。
    """
    q_aus = question.get("answer_units") or []
    v_aus = verified_rec.get("answer_units") or []
    if len(q_aus) != len(v_aus):
        raise PGoldBuildError(
            f"{question['question_id']}: answer_units 数量不一致 "
            f"(question={len(q_aus)}, verified={len(v_aus)})")
    for i, (qa, va) in enumerate(zip(q_aus, v_aus)):
        if (qa.get("claim") or "") != (va.get("statement") or ""):
            raise PGoldBuildError(
                f"{question['question_id']}: AU 索引 {i} claim != statement，"
                f"结构漂移，拒绝按索引对齐")
    return q_aus, v_aus


def _collect_spans_for_au(au: dict, verified_rec: dict,
                          span_by_id: Dict[str, SpanRef],
                          errors: List[str]) -> List[SpanRef]:
    """取一个 AU 关联的全部 evidence group 的全部 span（保序）。

    校验：AU 声明的每个 group 都存在且非空（每个 span_id 都能解析）。
    """
    gids = au.get("evidence_group_ids") or []
    groups_by_id = {g["group_id"]: g for g in verified_rec.get("evidence_groups") or []}
    spans: List[SpanRef] = []
    for gid in gids:
        g = groups_by_id.get(gid)
        if g is None:
            errors.append(
                f"{verified_rec['candidate_id']}: AU {au.get('unit_id')} 引用不存在的 "
                f"evidence group {gid}")
            continue
        sids = g.get("span_ids") or []
        if not sids:
            errors.append(
                f"{verified_rec['candidate_id']}: evidence group {gid} 为空 "
                f"（无 span），AU {au.get('unit_id')} 证据缺失")
            continue
        for sid in sids:
            sp = span_by_id.get(sid)
            if sp is None:
                errors.append(
                    f"{verified_rec['candidate_id']}: group {gid} 引用不存在的 span {sid}")
            else:
                spans.append(sp)
    return spans


def _build_one_question(question: dict, verified_rec: dict,
                        chunks: Dict[str, dict],
                        errors: List[str]) -> Optional[QuestionContext]:
    """构建单题上下文。出错时把错误追加到 errors 并返回 None。

    严格**不读取** question/verified 的 gold_answer 或 AU claim/statement
    内容 —— 上下文只由 span.text（原文引用）组成。
    """
    qid = str(question["question_id"])

    # span 索引（verified.spans 为唯一事实源）
    span_by_id: Dict[str, SpanRef] = {}
    for sp in verified_rec.get("spans") or []:
        span_by_id[sp["span_id"]] = SpanRef(
            span_id=sp["span_id"],
            chunk_id=sp["chunk_id"],
            char_start=int(sp["char_start"]),
            char_end=int(sp["char_end"]),
            text=sp["text"],
        )

    # AU 对齐（结构漂移检查）
    try:
        q_aus, v_aus = _align_answer_units(question, verified_rec)
    except PGoldBuildError as e:
        errors.append(str(e))
        return None

    # 逐 AU 收集 span（按 question 的 AU 顺序）
    ordered_spans: List[SpanRef] = []
    seen: set = set()
    for i, (qa, va) in enumerate(zip(q_aus, v_aus)):
        au_spans = _collect_spans_for_au(va, verified_rec, span_by_id, errors)
        required = bool(qa.get("required", True))
        if not au_spans:
            if required:
                errors.append(
                    f"{qid}: required answer unit {qa.get('unit_id')} 没有任何 "
                    f"evidence span")
            continue
        for sp in au_spans:
            # 去重：(chunk_id, char_start, char_end) 全局唯一，保首现顺序
            if sp.dedup_key not in seen:
                seen.add(sp.dedup_key)
                ordered_spans.append(sp)

    if errors:
        return None

    # span 与 chunk 原文一致性校验（全部校验完再决定成败）
    for sp in ordered_spans:
        ch = chunks.get(sp.chunk_id)
        if ch is None:
            errors.append(
                f"{qid}: span {sp.span_id} 的 chunk_id {sp.chunk_id} 不存在于 chunks 库")
            continue
        content = ch.get("content", "")
        if content[sp.char_start:sp.char_end] != sp.text:
            errors.append(
                f"{qid}: span {sp.span_id} 与原文切片不一致 "
                f"chunk[{sp.char_start}:{sp.char_end}] != span.text")
    if errors:
        return None

    # 组装：[Evidence N] text ...（N 全局递增，只含 span.text）
    parts = []
    for n, sp in enumerate(ordered_spans, start=1):
        parts.append(f"[Evidence {n}] {sp.text.strip()}")
    context = re.sub(r"\s+", " ", " ".join(parts)).strip()
    if not context:
        errors.append(f"{qid}: 组装后上下文为空")
        return None

    return QuestionContext(
        question_id=qid, context=context, spans=ordered_spans)


def build_p_gold_contexts(
    question_file: Path,
    verified_file: Path,
    chunks_file: Path,
    count_tokens: Callable[[str], int],
    context_hard_cap: int,
) -> BuildResult:
    """完整构建：加载 -> 映射 -> 逐题校验组装 -> token 计数。

    返回 BuildResult；errors 非空时调用方必须拒绝写盘。
    """
    errors: List[str] = []
    questions = load_questions(question_file)
    verified = load_verified(verified_file)
    chunks = load_chunks(chunks_file)

    # 双向唯一映射（question_id 侧完整性）
    cand2qid = _map_candidate_to_question(questions, verified)
    qid2cand = {q: c for c, q in cand2qid.items()}
    for q in questions:
        qid = str(q["question_id"])
        if qid not in qid2cand:
            errors.append(f"question_id {qid} 找不到对应的 verified candidate")

    records: List[QuestionContext] = []
    for q in questions:
        qid = str(q["question_id"])
        cand = qid2cand.get(qid)
        if cand is None:
            continue  # 已在上面报错
        rec = verified[cand]
        if rec.get("decision") != "accept":
            errors.append(f"{qid} ({cand}): verified decision != accept "
                          f"（当前 {rec.get('decision')!r}），不可作为 gold 证据")
            continue
        rc = _build_one_question(q, rec, chunks, errors)
        if rc is not None:
            try:
                rc.tokens = count_tokens(rc.context)
            except Exception as e:  # 权威计数失败 -> 构建失败（禁止估算静默通过）
                errors.append(f"{qid}: token 计数失败: {e}")
                continue
            if rc.tokens > context_hard_cap:
                errors.append(
                    f"{qid}: context {rc.tokens} tokens 超过输入安全上限 "
                    f"{context_hard_cap}")
            records.append(rc)

    # 统计（即使有错也给出全貌，便于一次修复）
    tokens = [r.tokens or 0 for r in records]
    stats = {
        "question_count": len(questions),
        "context_line_count": len(records),
        "missing_AU": 0,
        "missing_group": 0,
        "span_mismatch": 0,
        "duplicate_question_id": 0,
        "empty_context": 0,
        "over_cap": sum(1 for t in tokens if t > context_hard_cap),
        "total_tokens": sum(tokens),
        "max_tokens": max(tokens) if tokens else 0,
        "min_tokens": min(tokens) if tokens else 0,
        "context_hard_cap": context_hard_cap,
    }
    for e in errors:
        if "没有任何 evidence span" in e or "引用不存在的 evidence group" in e:
            stats["missing_AU"] += 1
        elif "为空（无 span）" in e:
            stats["missing_group"] += 1
        elif "与原文切片不一致" in e or "chunk_id" in e and "不存在" in e:
            stats["span_mismatch"] += 1
        elif "question_id 重复" in e:
            stats["duplicate_question_id"] += 1
        elif "组装后上下文为空" in e:
            stats["empty_context"] += 1

    return BuildResult(records=records, errors=errors, stats=stats)


def _tiktoken_estimate() -> Callable[[str], int]:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return lambda s: len(enc.encode(s))


def _qwen_token_counter() -> Callable[[str], int]:
    from hyperrag.qwen_tokenizer import get_qwen_token_counter
    return get_qwen_token_counter()


def _resolve_contract_bounds() -> Tuple[str, int]:
    """从冻结契约取 (contract_version, context_hard_cap)。

    cap = llm_max_length - max_response_tokens - TOKEN_RESERVE。
    """
    try:
        from hyperrag.experiment_contract import load_contract
        contract = load_contract()
    except Exception as e:
        raise PGoldBuildError(f"契约加载失败: {e}")
    sb = contract.system_boundary
    cap = sb.llm_max_length - sb.max_response_tokens - _TOKEN_RESERVE
    return contract.contract_version, cap


def write_outputs(out_dir: Path, result: BuildResult,
                  inputs: Dict, contract_version: str,
                  token_source: str, built_at: str) -> Dict[str, str]:
    """写 p_gold_contexts.txt / .jsonl / manifest，返回输出文件哈希表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "p_gold_contexts.txt"
    jsonl_path = out_dir / "p_gold_contexts.jsonl"
    manifest_path = out_dir / "p_gold_contexts_manifest.json"

    txt_lines = [r.context for r in result.records]
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(txt_lines) + "\n")

    jsonl_lines = []
    for r in result.records:
        jsonl_lines.append(json.dumps({
            "question_id": r.question_id,
            "context": r.context,
            "tokens": r.tokens,
            "n_spans": len(r.spans),
            "spans": [
                {"span_id": s.span_id, "chunk_id": s.chunk_id,
                 "char_start": s.char_start, "char_end": s.char_end}
                for s in r.spans
            ],
            "context_sha256": _sha256_text(r.context),
        }, ensure_ascii=False))
    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(jsonl_lines) + "\n")

    outputs = {
        "txt": str(txt_path),
        "txt_sha256": _sha256_file(txt_path),
        "txt_line_count": len(txt_lines),
        "jsonl": str(jsonl_path),
        "jsonl_sha256": _sha256_file(jsonl_path),
        "jsonl_line_count": len(jsonl_lines),
    }
    manifest = {
        "schema_version": "p-gold-contexts-v1",
        "contract_version": contract_version,
        "built_at": built_at,
        "token_source": token_source,
        "inputs": {k: {"path": str(p), "sha256": _sha256_file(p)}
                   for k, p in inputs.items()},
        "outputs": outputs,
        "question_ids_order": [r.question_id for r in result.records],
        "stats": result.stats,
        "checks_passed": not result.errors,
    }
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return outputs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="构建 P_gold 逐题 Gold evidence 上下文（只含原文证据）")
    ap.add_argument("--question-file", required=True, type=Path)
    ap.add_argument("--verified-evidence-file", required=True, type=Path)
    ap.add_argument("--chunks-file", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--no-tokenize", action="store_true",
                    help="跳过权威 Qwen /tokenize 计数，用 tiktoken 估算"
                         "（仅离线调试；manifest.token_source 会标记）")
    ap.add_argument("--context-hard-cap", type=int, default=None,
                    help="上下文 token 安全上限（默认从契约推导）")
    args = ap.parse_args(argv)

    try:
        contract_version, cap = _resolve_contract_bounds()
    except PGoldBuildError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    if args.context_hard_cap is not None:
        cap = args.context_hard_cap

    if args.no_tokenize:
        count_tokens = _tiktoken_estimate()
        token_source = "tiktoken-cl100k-estimate"
        print("WARNING: --no-tokenize 使用 tiktoken 估算，非权威 Qwen 计数，"
              "仅用于离线调试")
    else:
        try:
            count_tokens = _qwen_token_counter()
        except Exception as e:
            print(f"FATAL: 无法初始化权威 Qwen token 计数（{e}）；"
                  f"如需离线调试请加 --no-tokenize", file=sys.stderr)
            return 2
        token_source = "qwen_tokenize"

    result = build_p_gold_contexts(
        question_file=args.question_file,
        verified_file=args.verified_evidence_file,
        chunks_file=args.chunks_file,
        count_tokens=count_tokens,
        context_hard_cap=cap,
    )

    if result.errors:
        print(f"FAIL: 构建校验未通过，共 {len(result.errors)} 个问题，拒绝产出：",
              file=sys.stderr)
        for e in result.errors:
            print(f"  - {e}", file=sys.stderr)
        print(f"STATS: {json.dumps(result.stats, ensure_ascii=False)}",
              file=sys.stderr)
        return 2

    import datetime
    built_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    inputs = {
        "question_file": args.question_file,
        "verified_evidence_file": args.verified_evidence_file,
        "chunks_file": args.chunks_file,
    }
    outputs = write_outputs(args.output_dir, result, inputs,
                            contract_version, token_source, built_at)
    s = result.stats
    print(f"OK: P_gold contexts built")
    print(f"  questions={s['question_count']} lines={s['context_line_count']} "
          f"tokens(total={s['total_tokens']} max={s['max_tokens']} "
          f"min={s['min_tokens']}) cap={s['context_hard_cap']}")
    print(f"  missing_AU={s['missing_AU']} missing_group={s['missing_group']} "
          f"span_mismatch={s['span_mismatch']} duplicate_qid={s['duplicate_question_id']} "
          f"empty_context={s['empty_context']} over_cap={s['over_cap']}")
    print(f"  token_source={token_source}")
    print(f"  txt : {outputs['txt']} ({outputs['txt_line_count']} lines) "
          f"sha256={outputs['txt_sha256'][:16]}…")
    print(f"  jsonl: {outputs['jsonl']} ({outputs['jsonl_line_count']} lines)")
    print(f"  manifest: {args.output_dir / 'p_gold_contexts_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
