"""judge_longcat.py - 正式路径成功 Judge（LongCat-2.0，冻结契约 §12）

严格遵循 docs/question_set_v2_frozen_design.md §12.1：
- 模型: meituan-longcat/LongCat-2.0 (SiliconFlow)
- temperature=0，返回严格 JSON:
  {"verdict": "pass|fail|uncertain",
   "answer_units": [{"unit_id": "AU1", "status": "supported|missing|contradicted"}],
   "unsupported_claims": [], "critical_error": false, "evidence_sufficient": true}
- 不向 Judge 暴露路径名称（prompt 仅含 question/evidence/answer）
- 不要求/保存冗长思维链
- JSON 解析失败自动重试一次
- uncertain / unsupported claim / critical error / 判定矛盾 / 解析失败 -> human_review.jsonl
- Judge 输出保存模型名、服务、prompt hash、qrels 版本、时间戳（manifest.json）
- answer units 直接来自冻结问题集（questions_v2_manual_final.jsonl 的 answer_units 字段），
  不依赖 Judge 自造 AU -> 可审计、与校准集共享同一 AU 定义

隔离与可靠性:
- 输入/输出按 repeat/seed/snapshot 隔离:
  judge/longcat/r{repeat}_s{seed}_{snapshot[:8]}/  <-- 输出目录
- 进程级文件锁（O_EXCL 锁文件），跨进程防重复写（根治孤儿进程并发问题）
- 断点续跑: 锁内"查 done + 追加"原子操作

用法:
  # smoke
  python scripts/judge_longcat.py --mode judge --smoke 2 --concurrency 3
  # 全量（后台）
  python scripts/judge_longcat.py --mode judge --concurrency 3
  # 仅聚合
  python scripts/judge_longcat.py --mode summary
"""
import argparse
import errno
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

from openai import OpenAI

from my_config import (LLM_BASE_URL_SILICONFLOW, LLM_API_KEY_SILICONFLOW,
                       LLM_MODEL_SILICONFLOW)
from hyperrag.env import normalize_proxy_env

normalize_proxy_env()

# ---------------------------------------------------------------------------
# 版本 / 常量
# ---------------------------------------------------------------------------
JUDGE_VERSION = "v1.3.0"            # judge 实现版本（每次改 prompt/规则递增）
JSON_SCHEMA_VERSION = "v1"          # 输出 JSON schema 版本
# v2（2026-08-13）：unsupported_claims / critical_error 不再自动派生 fail，
# 而是悬置为 uncertain 并进入人工复核（契约 §12.1 line 867：uncertain/unsupported/
# critical/矛盾 一律进人工复核，不由规则直接判死）。prompt 未变 -> prompt hash 不变，
# 已产出的 480 条原始 LLM 输出可直接离线重裁决（--mode rejudge），无需重新请求 LongCat。
ADJUDICATION_RULE_VERSION = "v2"    # verdict 判定规则版本（冻结点之一）
# v3（2026-08-13）：真正按 evidence_requirements 计算 ER 级覆盖。
# v2 = AU 级 AND（接收 required_units，碰巧与 ER 1:1 时等价）。
# v3 = evidence requirement 组间 AND + alternative source 组内 OR，
# 接收 evidence_requirements（requirement_id/answer_unit_ids/alternative_chunk_ids），
# 返回 {requirement_hits, er_recall, complete_evidence_hit}。
# P2-P4 只搜索 -----Sources----- 区段（§8.3 line 528），P1/P_gold 全文。非冻结项（审计字段）。
COVERAGE_RULE_VERSION = "v3.1"

# 原始 480 条 LLM 调用的脚本版本历史（r0_s42_5c92f17c，2026-08-12 晚 ~ 08-13 凌晨）。
# judge_longcat.py 直到 8c67a0e 才首次提交，此前版本均不可从 git 回溯：
# - 91447b21…a7f3f9d：主跑 480 条 + 首次补跑（MAX_TOKENS=3000），用户快照哈希
# - max_tokens=6000 补跑版：39 条缺口/parse_fail 补跑（修复 JSON 截断），哈希不可回溯
# 现文件（离线裁决器）哈希独立记录为 adjudicator_script_hash，不冒充原始调用环境。
RAW_JUDGE_SCRIPT_HISTORY = [
    {"script_hash": "91447b212e79b9c8f31bd01cbd880f2232bddc3059433502cae39d018a7f3f9d",
     "hash_status": "verified",
     "temperature": 0.0, "max_tokens": 3000,
     "note": "主跑 480 条 + 首次补跑 85 条（未提交，用户快照）"},
    {"script_hash": None,
     "hash_status": "unrecoverable",
     "temperature": 0.0, "max_tokens": 6000,
     "note": "39 条缺口/parse_fail 补跑（MAX_TOKENS 3000->6000 修复截断后，未提交不可回溯）"},
]
RETRY_POLICY = {
    "json_parse_retry": 1,          # JSON 解析失败重试次数（契约：一次，针对解析层）
    "api_max_attempts": 5,          # API 调用最大尝试（基础设施层，不受契约"一次"限制）
    "api_backoff_base": 1.5,        # 指数退避基数（秒）
    "empty_response_retry": 5,      # 空响应重试上限（LongCat 偶发空 content）
}
TEMPERATURE = 0.0
# 3000 在"答案含大量 unsupported_claims"时输出被截断（JSON 不完整→parse fail），
# 2026-08-13 实测 qv2-r0054 等 5 条 parse_fail 均因截断。提高到 6000 后不再截断。
# 注：max_tokens 不在冻结项内（冻结项=judge_model/judge_prompt_hash/json_schema_version/
# adjudication_rule_version/retry_policy），改它不影响契约合规，prompt hash 也不变。
MAX_TOKENS = 6000

SNAPSHOT_DEFAULT = "5c92f17c03ed41adfd4bba6926a3a784418be13c3ec6596830ef15a0868b8c67"
ROUTES = ["P0", "P1", "P2", "P3", "P4", "P_gold"]
CONTRACT_VERSION = "v2.1-v1"

BASE = Path("caches") / "neurology_chunk1000"
PILOT_DIR = BASE / "question_set_v2" / "pilot_v1"
QUESTIONS_FILE = PILOT_DIR / "question_generation" / "questions_v2_manual_final.jsonl"
GOLD_CTX_FILE = PILOT_DIR / "p_gold" / "p_gold_contexts.jsonl"
RESPONSE_DIR = BASE / "response"
LONGCAT_DIR = PILOT_DIR / "judge" / "longcat"

VERDICTS = ("pass", "fail", "uncertain")
AU_STATUS = ("supported", "missing", "contradicted")

# ---------------------------------------------------------------------------
# Judge prompt（v1 —— prompt hash 将随此文本冻结）
# ---------------------------------------------------------------------------
JUDGE_SYS_PROMPT = (
    "You are an evidence-based answer judge. You evaluate whether a candidate answer "
    "fully and faithfully answers a question with respect to the given reference evidence. "
    "You only output a strict JSON object, no other text."
)

JUDGE_PROMPT = """Question:
{question}

Reference evidence (the only ground truth source; do not rely on outside knowledge):
{evidence}

Expected answer units (each unit is a claim a complete answer should address, marked required or optional):
{answer_units}

Candidate answer to judge:
{answer}

For EACH answer unit, determine its status by checking whether the candidate answer states the claim and whether that statement is consistent with the evidence:
- "supported": the answer states the claim and it is consistent with the evidence
- "missing": the answer does not state the claim
- "contradicted": the answer states something that contradicts the claim or the evidence

Also determine:
- "unsupported_claims": list of factual claims made by the answer that are NOT supported by the reference evidence (empty list if none)
- "critical_error": true if the answer contains a factual error that would materially mislead a reader on a key point
- "evidence_sufficient": true if the reference evidence is sufficient to determine every answer unit's status

verdict rules (apply strictly):
- "pass": every REQUIRED answer unit is supported, no answer unit is contradicted, and no critical error
- "fail": any REQUIRED answer unit is contradicted, OR critical_error is true, OR the answer states a claim directly contradicted by the evidence
- "uncertain": any REQUIRED answer unit is missing (the answer is incomplete), OR evidence_sufficient is false, OR the answer's support cannot be determined

Return ONLY a strict JSON object with exactly this schema (no markdown fence, no commentary):
{{"verdict": "pass" | "fail" | "uncertain",
  "answer_units": [{{"unit_id": "<unit_id>", "status": "supported" | "missing" | "contradicted"}}],
  "unsupported_claims": ["<claim or empty>"],
  "critical_error": true | false,
  "evidence_sufficient": true | false}}"""


def judge_prompt_hash() -> str:
    raw = (JUDGE_SYS_PROMPT + "\n" + JUDGE_PROMPT).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# LLM 调用（LongCat, temperature=0, 指数退避重试）
# ---------------------------------------------------------------------------
_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY_SILICONFLOW,
                         base_url=LLM_BASE_URL_SILICONFLOW, timeout=180)
    return _client


def llm_call(prompt: str, system_prompt: str) -> str:
    """LongCat 调用。空响应/异常按指数退避重试（重试带随机注释标记，避开服务端状态）。"""
    import uuid

    last_err = None
    for attempt in range(1, RETRY_POLICY["api_max_attempts"] + 1):
        try:
            call_prompt = prompt
            if attempt > 1:  # 重试时附加随机标记（HTML 注释，不影响模型语义）
                call_prompt = prompt + f"\n<!-- judge-retry:{uuid.uuid4().hex[:8]} -->"
            resp = get_client().chat.completions.create(
                model=LLM_MODEL_SILICONFLOW,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": call_prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError(f"空响应 (attempt {attempt})")  # 偶发空 content -> 重试
            return content
        except Exception as e:  # noqa: BLE001 - 网络/限流/超时/空响应统一重试
            last_err = e
            if attempt < RETRY_POLICY["api_max_attempts"]:
                time.sleep(RETRY_POLICY["api_backoff_base"] ** (attempt - 1))
    raise RuntimeError(f"LongCat API 调用失败: {last_err!r}")


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_questions():
    """qid -> {question, answer_units}。answer_units 仅取冻结问题集定义。"""
    out = {}
    for line in open(QUESTIONS_FILE, encoding="utf-8"):
        r = json.loads(line)
        qid = r["question_id"]
        units = [{"unit_id": u["unit_id"], "claim": u["claim"],
                  "required": bool(u.get("required", True))}
                 for u in r.get("answer_units", [])]
        out[qid] = {"question": r["question"], "answer_units": units}
    return out


def load_gold_ctx():
    out = {}
    for line in open(GOLD_CTX_FILE, encoding="utf-8"):
        r = json.loads(line)
        out[r["question_id"]] = r["context"]
    return out


def load_evidence_spans():
    """qid -> {unit_id: [evidence_spans]}。冻结题集的 evidence_spans 按 AU 组织。"""
    out = {}
    for line in open(QUESTIONS_FILE, encoding="utf-8"):
        r = json.loads(line)
        qid = r["question_id"]
        out[qid] = {es["unit_id"]: es["evidence_spans"]
                    for es in r.get("evidence_spans", [])}
    return out


def load_evidence_requirements():
    """qid -> [evidence_requirements]（ER→answer_unit_ids+alternative_chunk_ids）。

    每条 ER: {requirement_id, answer_unit_ids, alternative_chunk_ids}。
    组间 AND（所有 ER 必须满足）+ 组内 OR（ER 关联的任一 AU 的任一 span 命中即满足）。
    """
    out = {}
    for line in open(QUESTIONS_FILE, encoding="utf-8"):
        r = json.loads(line)
        out[r["question_id"]] = r.get("evidence_requirements", [])
    return out


def normalize_ws(s) -> str:
    """空白归一化（覆盖判定用；原文引文与组装后 context 的换行/缩进可能不同）。"""
    return re.sub(r"\s+", " ", (s or "")).strip()


SOURCE_SECTION_MARK = "-----Sources-----"


def extract_sources_section(context) -> str:
    """返回 context 的 Sources 区段文本。

    P2-P4 的组装 context 含 -----Entities-----/-----Relationships-----/-----Sources-----
    区段（Sources 在最后）；P1/P_gold 是纯 source 文本、无区段标记。
    契约（§8.3 line 528）：只有 Sources 区段可贡献 source evidence hit，
    Entity/Relationship 区段（CSV 描述）不能。无标记时返回整个 context。
    """
    c = context or ""
    m = re.search(rf"^{re.escape(SOURCE_SECTION_MARK)}$", c, re.M)
    if m:
        return c[m.end():]
    return c


def calc_source_evidence_coverage(context, evidence_requirements, spans_by_unit,
                                  search="full"):
    """ER 级证据组覆盖判定（§8.3 line 528 + coverage_rule_version=v3）。

    - evidence requirement 组间 AND：每个 ER 都必须满足（all_hit = all ERs hit）。
    - alternative source 组内 OR：ER 关联的任一 AU 的任一 evidence span quote
      （空白归一化后）作为子串命中搜索文本即满足该 ER。
    - 无 answer_unit_ids 或 AU 无 spans 的 ER 严格判 False（必须有证据）。
    - search="full"（P1/P_gold）：整个 context；search="sources"（P2-P4）：
      仅 -----Sources----- 区段（Entities/Relationships 区段不能贡献 source hit）。

    返回 {"requirement_hits": {er_id: bool}, "er_recall": float,
          "complete_evidence_hit": bool}。
    """
    ctx_n = normalize_ws(
        extract_sources_section(context) if search == "sources" else context)
    requirement_hits = {}
    for er in evidence_requirements:
        er_id = er.get("requirement_id", "?")
        au_ids = er.get("answer_unit_ids", [])
        allowed_chunks = set(er.get("alternative_chunk_ids", []))
        hit = False
        for au_id in au_ids:
            for span in spans_by_unit.get(au_id, []):
                # 只允许 ER 声明的 alternative_chunk_ids 中的 span 贡献命中
                if allowed_chunks and span.get("chunk_id") not in allowed_chunks:
                    continue
                q = normalize_ws(span.get("quote"))
                if q and q in ctx_n:
                    hit = True
                    break
            if hit:
                break
        requirement_hits[er_id] = hit
    total = len(evidence_requirements)
    hit_count = sum(1 for v in requirement_hits.values() if v)
    er_recall = round(hit_count / total, 4) if total else 0.0
    complete = bool(total) and all(requirement_hits.values())
    return {
        "requirement_hits": requirement_hits,
        "er_recall": er_recall,
        "complete_evidence_hit": complete,
    }


def final_answer_correctness(derived, human_review):
    """契约组合：human_review=true 的记录 final 一律 pending（§12.1 line 867：
    矛盾/critical/unsupported/uncertain 均需人工复核，不由规则判死）。
    pre_review 判定（derived）保留为审计字段。uncertain 兜底归 pending。"""
    if human_review:
        return "pending"
    return derived if derived in ("pass", "fail") else "pending"


def combine_route_success(final_ac, cov_hit, has_evidence_gate, human_review):
    """route_success = coverage AND answer_correctness（三值逻辑，§8.3 line 518-520）。

    coverage 失败是确定性 fail，不被 human_review 悬置——证据缺失是客观事实，
    不依赖人工裁决。只有 coverage 通过且 answer_correctness=pending 时才 pending。
    final_ac 已由 final_answer_correctness() 计算（含 human_review→pending），故
    human_review 参数在此不再短路，仅保留签名兼容。

    真值表：
      coverage=False                     → fail
      coverage=True  + answer=pass       → pass
      coverage=True  + answer=fail       → fail
      coverage=True  + answer=pending    → pending
      coverage=None  + answer=fail       → fail
      coverage=None  + 其他              → pending
      (无证据门槛 P0)                     → answer_correctness
    """
    if not has_evidence_gate:
        return final_ac
    if cov_hit is False:
        return "fail"
    if cov_hit is True:
        if final_ac == "pass":
            return "pass"
        if final_ac == "fail":
            return "fail"
        return "pending"
    # cov_hit is None（unknown，如 no_context）
    if final_ac == "fail":
        return "fail"
    return "pending"


def load_result_contexts(path: Path) -> dict:
    """qid -> final context 原文（result.jsonl 顶层 context 字段，组装后喂回答模型的文本）。"""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["question_id"]] = r.get("context") or ""
    return out


def result_file(route: str, repeat: int, seed: int, snapshot: str) -> Path:
    return RESPONSE_DIR / (
        f"fixed_{route}_r{repeat}_s{seed}_{CONTRACT_VERSION}_{snapshot}_result.jsonl")


def load_results(path: Path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["question_id"]] = r["result"]
    return out


def canonical_order():
    return [json.loads(l)["question_id"] for l in open(GOLD_CTX_FILE, encoding="utf-8")]


# ---------------------------------------------------------------------------
# 输出目录 / 解析 / 文件锁 / 落盘
# ---------------------------------------------------------------------------
def out_dir(repeat: int, seed: int, snapshot: str) -> Path:
    d = LONGCAT_DIR / f"r{repeat}_s{seed}_{snapshot[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_json_obj(text: str):
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        try:
            start, end = text.index("{"), text.rindex("}")
            return json.loads(text[start:end + 1])
        except Exception:
            return None


def validate_schema(data) -> tuple:
    """返回 (ok, error_msg)。校验契约 JSON 结构。"""
    if not isinstance(data, dict):
        return False, "顶层非 JSON 对象"
    v = data.get("verdict")
    if v not in VERDICTS:
        return False, f"verdict 非法: {v!r}"
    aus = data.get("answer_units")
    if not isinstance(aus, list):
        return False, "answer_units 非数组"
    for au in aus:
        if not isinstance(au, dict) or "unit_id" not in au or au.get("status") not in AU_STATUS:
            return False, f"answer_unit 非法: {au!r}"
    for fld in ("unsupported_claims",):
        if not isinstance(data.get(fld), list):
            return False, f"{fld} 非数组"
    for fld in ("critical_error", "evidence_sufficient"):
        if not isinstance(data.get(fld), bool):
            return False, f"{fld} 非布尔"
    return True, ""


class FileLock:
    """跨进程文件锁（O_EXCL 原子创建；超时后强删 stale 锁）。"""

    def __init__(self, path: Path, timeout: float = 120.0, poll: float = 0.2):
        self.path = Path(path)
        self.timeout = timeout
        self.poll = poll
        self.fd = None

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                if time.time() - t0 > self.timeout:  # stale 锁强删
                    try:
                        os.unlink(str(self.path))
                    except OSError:
                        pass
                    t0 = time.time()
                time.sleep(self.poll)

    def __exit__(self, *exc):
        try:
            if self.fd is not None:
                os.close(self.fd)
        except OSError:
            pass
        try:
            os.unlink(str(self.path))
        except OSError:
            pass


def read_done_ok(out: Path) -> set:
    """已成功解析的 qid（parse_ok=true）。仅用于并发写入幂等守卫。"""
    if not out.exists():
        return set()
    ids = set()
    for line in open(out, encoding="utf-8"):
        try:
            r = json.loads(line)
            if r.get("parse_ok"):
                ids.add(r["question_id"])
        except Exception:
            continue
    return ids


def read_done_any(out: Path) -> set:
    """已落盘任何记录的 qid（无论 parse_ok 与否）。

    用于 resume 跳过判定：确定性的 parse_failed cell（如 LongCat 输出撞 token
    上限被截断）不应在每次 resume 时被反复重跑——它必然产出同一份无效结果，
    重试没有意义，应作为一等公民的 judge_error 结果留存。
    """
    if not out.exists():
        return set()
    ids = set()
    for line in open(out, encoding="utf-8"):
        try:
            r = json.loads(line)
            ids.add(r["question_id"])
        except Exception:
            continue
    return ids


def write_record(out: Path, qid: str, rec: dict) -> bool:
    """锁内 '查 done_ok + 同 qid 去重替换写' 原子操作。

    返回是否实际写入（False=已被其他进程成功写过 parse_ok 记录）。
    同 qid 旧记录（含 parse_failed）会被替换，避免 resume 累积重复行（#bloat 修复）。
    """
    lock = out.with_suffix(out.suffix + ".lock")
    with FileLock(lock):
        if qid in read_done_ok(out):
            return False
        kept = []
        if out.exists():
            for line in open(out, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    if r.get("question_id") == qid:
                        continue  # 丢弃同 qid 旧记录（含 parse_failed）
                except Exception:
                    pass
                kept.append(line if line.endswith("\n") else line + "\n")
        kept.append(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(out, "w", encoding="utf-8") as f:
            f.writelines(kept)
        return True


# ---------------------------------------------------------------------------
# 单题判定
# ---------------------------------------------------------------------------
def adjudicate(data: dict, required_units: set) -> tuple:
    """由 AU 状态推导 verdict（adjudication_rule_version=v2）。

    返回 (verdict, conflict_flag)。conflict_flag=True 表示 Judge 输出内部矛盾，需人工复核。

    v2 规则（对照冻结契约 §8.3 / §12.1）：
    - 任一 required AU contradicted          -> fail（§8.3 line 527 冲突即失败）
    - critical_error 或 unsupported_claims 非空 -> 不自动 fail；悬置 uncertain，进人工复核
    - required missing 或 evidence 不足      -> uncertain
    - 全部 required supported               -> pass
    - derived != Judge 输出 verdict          -> conflict -> 人工复核
    """
    aus = data.get("answer_units") or []
    status_by_id = {au.get("unit_id"): au.get("status") for au in aus}
    req_status = {u: status_by_id.get(u) for u in required_units}
    any_missing_status = any(s == "missing" for s in req_status.values())
    any_contradicted = any(s == "contradicted" for s in req_status.values())
    critical = bool(data.get("critical_error"))
    unsupported = data.get("unsupported_claims") or []
    ev_sufficient = bool(data.get("evidence_sufficient"))

    derived = None
    if any_contradicted:
        derived = "fail"
    elif critical or unsupported:
        # v2：不自动判死，悬置待人工复核（理由已由 judge_one/rejudge 记录到 human_review）
        derived = "uncertain"
    elif any_missing_status or not ev_sufficient:
        derived = "uncertain"
    else:
        if all(s == "supported" for s in req_status.values()) and req_status:
            derived = "pass"
        else:
            derived = "uncertain"  # 异常兜底

    # 矛盾检测：Judge 输出 verdict 与推导不一致 -> 人工复核
    conflict = (derived != data.get("verdict"))
    return derived, conflict


def judge_one(qid: str, questions: dict, gold_ctx: dict, answer: str) -> dict:
    q = questions[qid]
    units_lines = []
    for u in q["answer_units"]:
        tag = "required" if u["required"] else "optional"
        units_lines.append(f"- {u['unit_id']} ({tag}): {u['claim']}")
    units_text = "\n".join(units_lines) if units_lines else "(no answer units defined)"

    prompt = JUDGE_PROMPT.format(
        question=q["question"], evidence=gold_ctx[qid],
        answer_units=units_text, answer=answer,
    )

    data = None
    raw = ""
    for attempt in range(RETRY_POLICY["json_parse_retry"] + 1):
        raw = llm_call(prompt, JUDGE_SYS_PROMPT)
        data = extract_json_obj(raw)
        if data is not None:
            ok, err = validate_schema(data)
            if ok:
                break
        data = None  # 解析/schema 失败 -> 重试一次
    if data is None:
        return {"question_id": qid, "route": None, "parse_ok": False,
                "response": raw, "verdict": None, "answer_units": None,
                "unsupported_claims": None, "critical_error": None,
                "evidence_sufficient": None, "derived_verdict": None,
                "conflict": True, "human_review": True,
                "human_review_reason": "json_parse_failed"}

    required_units = {u["unit_id"] for u in q["answer_units"] if u["required"]}
    derived, conflict = adjudicate(data, required_units)

    # 人工复核条件（契约 12.1）
    reasons = []
    if data["verdict"] == "uncertain":
        reasons.append("verdict_uncertain")
    if data.get("unsupported_claims"):
        reasons.append("unsupported_claims")
    if data.get("critical_error"):
        reasons.append("critical_error")
    if conflict:
        reasons.append("verdict_conflict")

    return {
        "question_id": qid,
        "parse_ok": True,
        "response": raw,
        "verdict": data["verdict"],
        "answer_units": data["answer_units"],
        "unsupported_claims": data["unsupported_claims"],
        "critical_error": data["critical_error"],
        "evidence_sufficient": data["evidence_sufficient"],
        "derived_verdict": derived,
        "conflict": conflict,
        "human_review": bool(reasons),
        "human_review_reason": reasons,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_judge(routes, repeat, seed, snapshot, smoke, concurrency):
    questions = load_questions()
    gold_ctx = load_gold_ctx()
    order = canonical_order()
    if smoke:
        order = order[:smoke]
    od = out_dir(repeat, seed, snapshot)

    total_planned = 0
    for route in routes:
        rf = result_file(route, repeat, seed, snapshot)
        if not rf.exists():
            print(f"[judge] 跳过 {route}: 输入不存在 {rf}")
            continue
        answers = load_results(rf)
        out = od / f"{route}_verdict.jsonl"
        done_any = read_done_any(out)
        qids = [q for q in order if q in answers and q in questions and q in gold_ctx
                and q not in done_any]
        total_planned += len(qids)

        def work(qid):
            rec = judge_one(qid, questions, gold_ctx, answers[qid])
            rec["route"] = route
            wrote = write_record(out, qid, rec)
            if wrote:
                hr_out = od / "human_review.jsonl"
                if rec.get("human_review"):
                    write_record(hr_out, f"{route}|{qid}", rec)
            return qid, wrote

        print(f"[judge] {route}: 计划 {len(qids)} 题 -> {out}")
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(work, q): q for q in qids}
            for fut in tqdm(as_completed(futs), total=len(qids), desc=f"judge/{route}"):
                try:
                    fut.result()
                except Exception as e:
                    print(f"[judge/{route}] 失败 {futs[fut]}: {e!r}")
    print(f"[judge] 完成，共计划 {total_planned} 题")


def rejudge_records(repeat, seed, snapshot, routes):
    """离线重裁决存量记录（adjudication_rule_version v1 -> v2）。

    原始 LLM 输出（verdict/answer_units/unsupported_claims/critical_error/
    evidence_sufficient）已完整存盘，无需重新请求 LongCat。仅重算：
    derived_verdict / conflict / human_review / human_review_reason，
    并重建 human_review.jsonl，更新 manifest 的 rule_version 与时间戳。
    单进程离线操作，直接全量重建文件（不做追加）。
    """
    od = out_dir(repeat, seed, snapshot)
    questions = load_questions()
    total = 0
    for route in routes:
        out = od / f"{route}_verdict.jsonl"
        if not out.exists():
            print(f"[rejudge] 跳过 {route}: 无记录文件")
            continue
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        updated = []
        for r in rows:
            if not r.get("parse_ok"):  # parse_failed 无 LLM 输出，原样保留
                updated.append(r)
                continue
            qid = r["question_id"]
            data = {
                "verdict": r.get("verdict"),
                "answer_units": r.get("answer_units") or [],
                "unsupported_claims": r.get("unsupported_claims") or [],
                "critical_error": bool(r.get("critical_error")),
                "evidence_sufficient": bool(r.get("evidence_sufficient")),
            }
            required = {u["unit_id"] for u in questions[qid]["answer_units"]
                        if u["required"]}
            derived, conflict = adjudicate(data, required)
            reasons = []
            if r.get("verdict") == "uncertain":
                reasons.append("verdict_uncertain")
            if r.get("unsupported_claims"):
                reasons.append("unsupported_claims")
            if r.get("critical_error"):
                reasons.append("critical_error")
            if conflict:
                reasons.append("verdict_conflict")
            r2 = dict(r)
            r2["derived_verdict"] = derived
            r2["conflict"] = conflict
            r2["human_review"] = bool(reasons)
            r2["human_review_reason"] = reasons
            r2["rejudged_from_rule"] = "v1"
            r2["rejudged_at"] = datetime.now(timezone.utc).isoformat()
            updated.append(r2)
        with open(out, "w", encoding="utf-8") as f:
            for r in updated:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(updated)
        print(f"[rejudge] {route}: {len(updated)} 条已用 v2 规则重裁决")

    # 重建 human_review.jsonl（按 qid 去重，优先 parse_ok）
    hr = od / "human_review.jsonl"
    hr_rows = []
    seen = {}
    for route in routes:
        out = od / f"{route}_verdict.jsonl"
        if not out.exists():
            continue
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("human_review"):
                continue
            key = f"{route}|{r['question_id']}"
            if key not in seen or (r.get("parse_ok") and not seen[key].get("parse_ok")):
                seen[key] = r
    hr_rows = sorted(seen.values(), key=lambda r: (r["route"], r["question_id"]))
    with open(hr, "w", encoding="utf-8") as f:
        for r in hr_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[rejudge] human_review.jsonl 重建完成: {len(hr_rows)} 条")
    write_manifest(repeat, seed, snapshot, routes, mode="rejudge")
    print(f"[rejudge] 完成，共重裁决 {total} 条；manifest 已更新 rule_version=v2，"
          f"raw 部分保留原始 480 次调用环境")


def merge_raw_judge_fields(old, mode, script_hash, temperature, max_tokens):
    """合并 manifest 的 raw_judge_* 字段（原始 480 次 API 调用环境，与裁决器分离）。

    - mode="judge"：追加当前脚本哈希与本次调用配置（去重）。
    - mode="rejudge"：保留旧 raw 历史；若从未记录则补 RAW_JUDGE_SCRIPT_HISTORY
      （91447b21 主跑版完整 SHA-256 + 6000 补跑版 hash_status=unrecoverable）。
    - raw_judge_script_hashes 只含可校验的实际哈希（跳过 null/unrecoverable）。
    - raw_generation_configs 保留完整条目（含 hash_status 字段）。
    返回 (raw_prompt_hash, raw_hashes, raw_configs)。
    """
    if mode == "judge":
        raw_hashes = list(old.get("raw_judge_script_hashes") or [])
        if script_hash not in raw_hashes:
            raw_hashes.append(script_hash)
        raw_configs = list(old.get("raw_generation_configs") or [])
        raw_configs.append({
            "script_hash": script_hash, "hash_status": "verified",
            "temperature": temperature, "max_tokens": max_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "judge 调用"})
        raw_prompt = old.get("raw_judge_prompt_hash") or judge_prompt_hash()
    else:  # rejudge：不追加，保留历史；无历史则补已知记录
        raw_hashes = old.get("raw_judge_script_hashes")
        raw_configs = old.get("raw_generation_configs")
        raw_prompt = old.get("raw_judge_prompt_hash")
        if not raw_hashes:
            # 只取有实际哈希的（跳过 null/unrecoverable）
            raw_hashes = [h["script_hash"] for h in RAW_JUDGE_SCRIPT_HISTORY
                          if h.get("script_hash")]
        if not raw_configs:
            raw_configs = [dict(h) for h in RAW_JUDGE_SCRIPT_HISTORY]
        if not raw_prompt:
            raw_prompt = judge_prompt_hash()
    return raw_prompt, raw_hashes, raw_configs


def write_manifest(repeat, seed, snapshot, routes, mode="judge", concurrency=None):
    od = out_dir(repeat, seed, snapshot)
    result_hashes = {}
    for route in routes:
        rf = result_file(route, repeat, seed, snapshot)
        result_hashes[route] = (sha256_file(rf) if rf.exists() else None)
    script_hash = sha256_file(Path(__file__).resolve())
    # 保留旧 manifest（raw 部分跨 judge/rejudge 运行继承，不互相覆盖）
    old = {}
    mp = od / "manifest.json"
    if mp.exists():
        try:
            old = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    raw_prompt, raw_hashes, raw_configs = merge_raw_judge_fields(
        old, mode, script_hash, TEMPERATURE, MAX_TOKENS)
    if mode == "judge" and concurrency is not None and raw_configs:
        raw_configs[-1]["concurrency"] = concurrency
    manifest = {
        "judge_model": LLM_MODEL_SILICONFLOW,
        "service": LLM_BASE_URL_SILICONFLOW,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "judge_version": JUDGE_VERSION,
        "judge_prompt_hash": judge_prompt_hash(),
        "json_schema_version": JSON_SCHEMA_VERSION,
        "adjudication_rule_version": ADJUDICATION_RULE_VERSION,
        "coverage_rule_version": COVERAGE_RULE_VERSION,
        "retry_policy": RETRY_POLICY,
        "judge_script": str(Path(__file__).resolve()),
        # ---- 原始 480 次 API 调用环境（不随裁决器修改而变）----
        "raw_judge_prompt_hash": raw_prompt,
        "raw_judge_script_hashes": raw_hashes,
        "raw_generation_configs": raw_configs,
        # ---- 离线裁决器（现文件，独立哈希）----
        "adjudicator_script_hash": script_hash,
        "qrels_version": "questions_v2_manual_final+verified_evidence",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repeat": repeat,
        "seed": seed,
        "snapshot": snapshot,
        "inputs": {
            "questions_file": str(QUESTIONS_FILE),
            "questions_file_hash": sha256_file(QUESTIONS_FILE),
            "gold_context_file": str(GOLD_CTX_FILE),
            "gold_context_file_hash": sha256_file(GOLD_CTX_FILE),
            "result_files": result_hashes,
        },
    }
    (od / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_summary(repeat, seed, snapshot):
    od = out_dir(repeat, seed, snapshot)
    questions = load_questions()
    evidence = load_evidence_spans()
    evidence_reqs = load_evidence_requirements()
    summary = {"repeat": repeat, "seed": seed, "snapshot": snapshot}
    routes_sum = {}
    for route in ROUTES:
        out = od / f"{route}_verdict.jsonl"
        if not out.exists():
            continue
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        # 按 qid 去重：优先 parse_ok，其次保留最后一条（parse_failed 重跑可能留双记录）
        best = {}
        for r in rows:
            q = r["question_id"]
            if q not in best or (r.get("parse_ok") and not best[q].get("parse_ok")):
                best[q] = r
        rows = list(best.values())
        parsed = [r for r in rows if r.get("parse_ok")]
        vc = {}
        ac = {"pass": 0, "fail": 0, "pending": 0}
        ac_pre = {"pass": 0, "fail": 0, "pending": 0}
        cov = {"pass": 0, "fail": 0, "no_context": 0}
        rs = {"pass": 0, "fail": 0, "pending": 0}
        rf = result_file(route, repeat, seed, snapshot)
        contexts = load_result_contexts(rf) if rf.exists() else {}
        has_gate = route != "P0"
        for r in parsed:
            v = r["verdict"]
            vc[v] = vc.get(v, 0) + 1
            # answer_correctness（§8.3：最终回答必须覆盖所有必要答案点，不得冲突）
            # 契约 §12.1：human_review=true -> final=pending（矛盾/critical/unsupported/
            # uncertain 均需人工复核，不由规则判死）；pre_review 判定保留为审计。
            d = r.get("derived_verdict")
            hr = bool(r.get("human_review"))
            ac_pre["pass" if d == "pass" else ("fail" if d == "fail" else "pending")] += 1
            final = final_answer_correctness(d, hr)
            ac[final] += 1
            # source_evidence_coverage（仅 P1-P4 与 P_gold 有证据门槛；P0 无）
            if not has_gate:
                # P0 无证据门槛：route_success = final_answer_correctness
                rs[combine_route_success(final, None, False, hr)] += 1
                continue
            ctx = contexts.get(r["question_id"])
            if ctx is None:
                cov["no_context"] += 1
                # coverage=unknown(None)：answer=fail→fail，否则 pending
                rs[combine_route_success(final, None, True, hr)] += 1
                continue
            ers = evidence_reqs.get(r["question_id"], [])
            # P1/P_gold 纯 source 全文即 Sources；P2-P4 只搜索 -----Sources----- 区段
            cov_result = calc_source_evidence_coverage(
                ctx, ers, evidence.get(r["question_id"], {}),
                search="sources" if route in ("P2", "P3", "P4") else "full")
            cov_hit = cov_result["complete_evidence_hit"]
            cov["pass" if cov_hit else "fail"] += 1
            rs[combine_route_success(final, cov_hit, True, hr)] += 1
        routes_sum[route] = {
            "n_total": len(rows), "n_parsed": len(parsed),
            "parse_rate": round(len(parsed) / len(rows), 4) if rows else 0.0,
            "verdicts": vc,
            "answer_correctness": ac,
            "answer_correctness_pre_review": ac_pre,
            "source_evidence_coverage": cov,
            "route_success": rs,
            "n_human_review": sum(1 for r in rows if r.get("human_review")),
        }
    summary["routes"] = routes_sum
    hr = od / "human_review.jsonl"
    summary["human_review_total"] = (sum(1 for _ in open(hr, encoding="utf-8"))
                                     if hr.exists() else 0)
    (od / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description="正式 Judge（LongCat）")
    ap.add_argument("--mode", choices=["judge", "rejudge", "summary"], default="judge")
    ap.add_argument("--routes", nargs="+", default=ROUTES)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snapshot", default=SNAPSHOT_DEFAULT)
    ap.add_argument("--smoke", type=int, default=0, help="每任务只跑前 N 题（0=全量）")
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    if args.mode == "judge":
        run_judge(args.routes, args.repeat, args.seed, args.snapshot,
                  args.smoke, args.concurrency)
        write_manifest(args.repeat, args.seed, args.snapshot, args.routes,
                       mode="judge", concurrency=args.concurrency)
    elif args.mode == "rejudge":
        rejudge_records(args.repeat, args.seed, args.snapshot, args.routes)
    elif args.mode == "summary":
        build_summary(args.repeat, args.seed, args.snapshot)


if __name__ == "__main__":
    from tqdm import tqdm  # noqa: E402
    main()
