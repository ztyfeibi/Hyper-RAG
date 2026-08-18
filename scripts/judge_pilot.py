"""judge_pilot.py - Pilot 六路径（P0-P4 + P_gold）质量判定（Judge）

两套互补口径：
  A. 五维打分（absolute）：Comprehensiveness / Diversity / Empowerment / Logical / Readability
     6 路径 x 80 题，统一以该题 gold context（p_gold_contexts.jsonl）作参考文档，横向可比。
  B. Pairwise 对比（relative）：P0-P4 各 vs P_gold，逐题二选一（8 维 + 总体 winner），
     双向取平均消除答案位置偏差（fwd: candidate 在 Answer1；rev: candidate 在 Answer2）。

只新增不改动：不触碰 Step_3、evaluate/*、六路径 result 产物。
断点续跑：每题结果即时落盘 judge/scoring|selection/*.jsonl，重启自动跳过已完成 question_id。

用法示例：
  # smoke（每任务前 3 题，并发 4）
  python scripts/judge_pilot.py --mode all --smoke 3 --concurrency 4
  # 全量（后台，并发 4）
  python scripts/judge_pilot.py --mode all --concurrency 4
  # 只聚合已落盘结果
  python scripts/judge_pilot.py --mode summary
"""
import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

import numpy as np
from tqdm import tqdm
from openai import OpenAI

from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from hyperrag.env import normalize_proxy_env

normalize_proxy_env()

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SNAPSHOT_DEFAULT = "5c92f17c03ed41adfd4bba6926a3a784418be13c3ec6596830ef15a0868b8c67"
ROUTES = ["P0", "P1", "P2", "P3", "P4", "P_gold"]
CANDIDATE_ROUTES = ["P0", "P1", "P2", "P3", "P4"]

BASE = Path("caches") / "neurology_chunk1000"
RESPONSE_DIR = BASE / "response"
PILOT_DIR = BASE / "question_set_v2" / "pilot_v1"
GOLD_CTX_FILE = PILOT_DIR / "p_gold" / "p_gold_contexts.jsonl"
JUDGE_DIR = PILOT_DIR / "judge"

SCORING_METRICS = [
    "Comprehensiveness", "Diversity", "Empowerment", "Logical", "Readability",
]
SELECTION_METRICS = [
    "Comprehensiveness", "Empowerment", "Accuracy", "Relevance",
    "Coherence", "Clarity", "Logical", "Flexibility",
]

SCORING_SYS_PROMPT = (
    "---Role---\n"
    "You are an expert tasked with evaluating answers to the questions by using the "
    "relevant documents based on five criteria: **Comprehensiveness**, **Diversity**, "
    "**Empowerment**, **Logical**, and **Readability**."
)

SCORING_PROMPT = """You will evaluate the answer to the question by using the relevant documents based on five criteria: **Comprehensiveness**, **Diversity**, **Empowerment**, **Logical**, and **Readability**.

- **Comprehensiveness** -
Measure whether the answer comprehensively covers all key aspects of the question and whether there are omissions.
Level   | score range | description
Level 1 | 0-20   | The answer is extremely one-sided, leaving out key parts or important aspects of the question.
Level 2 | 20-40  | The answer has some content, but it misses many important aspects of the question and is not comprehensive enough.
Level 3 | 40-60  | The answer is more comprehensive, covering the main aspects of the question, but there are still some omissions.
Level 4 | 60-80  | The answer is comprehensive, covering most aspects of the question, with few omissions.
Level 5 | 80-100 | The answer is extremely comprehensive, covering all aspects of the question with no omissions, enabling the reader to gain a complete understanding.

- **Diversity** -
Measure the richness of the answer content, including not only the direct answer to the question, but also the background knowledge related to the question, extended information, case studies, etc.
Level   | score range | description
Level 1 | 0-20   | The answer is extremely sparse, providing only direct answers to questions without additional information or expansion of relevant knowledge.
Level 2 | 20-40  | The answer provides a direct answer to the question, but contains only a small amount of relevant knowledge expansion, the content is relatively thin.
Level 3 | 40-60  | In addition to the direct answers, the answer also provides some relevant background knowledge or supplementary information.
Level 4 | 60-80  | The answer is rich in content, not only answering the question, but also providing more relevant background knowledge, supplementary information or expanded content, so that readers can understand the question more comprehensively.
Level 5 | 80-100 | In addition to the direct answers, the answer also provides a lot of relevant knowledge, expanded content and in-depth analysis, so that readers can get a comprehensive and in-depth understanding.

- **Empowerment** -
Measure the credibility of the answer and whether it convinces the reader that it is correct. High confidence answers often cite authoritative sources or provide sufficient evidence.
Level   | score range | description
Level 1 | 0-20   | The answer lacks credibility, contains obvious errors or false information, and fails to convince the reader.
Level 2 | 20-40  | The answer has some credibility, but some of the information is not accurate or lacks support, which may cause readers to doubt.
Level 3 | 40-60  | The answer is credible and provides some supporting information, but there are still some areas that are not clear or authoritative.
Level 4 | 60-80  | The answer is highly credible, providing sufficient supporting information (such as quotes, data, etc.), so that readers can be more convinced.
Level 5 | 80-100 | The answer is highly credible, providing sufficient and authoritative supporting information, so that the reader is completely convinced of their correctness.

- **Logical** -
Measure whether the answer is coherent, clear, and easy to understand.
Level   | score range | description
Level 1 | 0-20   | The answer is illogical, incoherent, and difficult to understand.
Level 2 | 20-40  | The answer has some logic, but it is incoherent and difficult to understand in parts.
Level 3 | 40-60  | The answer is logically clear and the sentences are basically coherent, but there are still a few logical loopholes or unclear places.
Level 4 | 60-80  | The answer is logical, coherent, and easy to understand.
Level 5 | 80-100 | The answer is extremely logical, fluent and well-organized, making it easy for the reader to follow the author's thoughts.

- **Readability** -
Measure whether the answer is well organized, clear in format, and easy to read.
Level   | score range | description
Level 1 | 0-20   | The format of the answer is confused, the writing is poorly organized and difficult to read.
Level 2 | 20-40  | There are some problems in the format of the answer, the organizational structure of the text is not clear enough, and it is difficult to read.
Level 3 | 40-60  | The format of the answer is basically clear, the writing structure is good, but there is still room for improvement.
Level 4 | 60-80  | The format of the answer is clear, the writing is well organized and the reading is smooth.
Level 5 | 80-100 | The format of the answer is very clear, the writing structure is great, the reading experience is excellent, the format is standardized and easy to understand.

For each indicator, please give the problem a corresponding Level based on the description of the indicator, and then give a score according to the score range of the level.

Here are the relevant documents:
{reference}

Here are the questions:
{query}

Here are the answers:
{answer}

Evaluate the answer using the five criteria listed above. For each criterion, provide a summary description, give a Level based on the description of the indicator, and then give a score based on the score range of the level.

Output your evaluation in the following JSON format (valid JSON, no markdown fence):

{{
    "Comprehensiveness": {{
        "Explanation": "Provide explanation here",
        "Level": 4,
        "Score": 78
    }},
    "Diversity": {{
        "Explanation": "Provide explanation here",
        "Level": 4,
        "Score": 75
    }},
    "Empowerment": {{
        "Explanation": "Provide explanation here",
        "Level": 4,
        "Score": 80
    }},
    "Logical": {{
        "Explanation": "Provide explanation here",
        "Level": 4,
        "Score": 82
    }},
    "Readability": {{
        "Explanation": "Provide explanation here",
        "Level": 4,
        "Score": 85
    }}
}}"""

SELECTION_SYS_PROMPT = (
    "---Role---\n"
    "You will evaluate two answers to the same question based on eight criteria: "
    "**Comprehensiveness**, **Empowerment**, **Accuracy**, **Relevance**, **Coherence**, "
    "**Clarity**, **Logical**, and **Flexibility**."
)

SELECTION_PROMPT = """You will evaluate two answers to the same question by using the relevant documents based on eight criteria: **Comprehensiveness**, **Empowerment**, **Accuracy**, **Relevance**, **Coherence**, **Clarity**, **Logical**, and **Flexibility**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?
- **Accuracy**: How well does the answer align with factual truth and avoid hallucination based on the retrieved context?
- **Relevance**: How precisely does the answer address the core aspects of the question without including unnecessary information?
- **Coherence**: How well does the system integrate and synthesize information from multiple sources into a logically flowing response?
- **Clarity**: How well does the system provide complete information while avoiding unnecessary verbosity and redundancy?
- **Logical**: How well does the system maintain consistent logical arguments without contradicting itself across the response?
- **Flexibility**: How well does the system handle various question formats, tones, and levels of complexity?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why.

Here are the questions:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the eight criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format (valid JSON, no markdown fence):

{{
    "Comprehensiveness": {{
        "Winner": "Answer 1",
        "Explanation": "Provide explanation here"
    }},
    "Empowerment": {{
        "Winner": "Answer 1",
        "Explanation": "Provide explanation here"
    }},
    "Accuracy": {{
        "Winner": "Answer 1",
        "Explanation": "Provide explanation here"
    }},
    "Relevance": {{
        "Winner": "Answer 1",
        "Explanation": "Provide explanation here"
    }},
    "Coherence": {{
        "Winner": "Answer 1",
        "Explanation": "Provide explanation here"
    }},
    "Clarity": {{
        "Winner": "Answer 1",
        "Explanation": "Provide explanation here"
    }},
    "Logical": {{
        "Winner": "Answer 1",
        "Explanation": "Provide explanation here"
    }},
    "Flexibility": {{
        "Winner": "Answer 1",
        "Explanation": "Provide explanation here"
    }}
}}"""

# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------
def llm_model_func(prompt, system_prompt=None, temperature=0.1, max_tokens=3000, **kwargs) -> str:
    openai_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        **kwargs,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def result_file(route: str, snapshot: str) -> Path:
    return RESPONSE_DIR / f"fixed_{route}_r0_s42_v2.1-v1_{snapshot}_result.jsonl"


def load_result_jsonl(path: Path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return {r["question_id"]: r for r in rows}


def load_gold_ctx() -> dict:
    rows = [json.loads(l) for l in open(GOLD_CTX_FILE, encoding="utf-8")]
    return {r["question_id"]: r["context"] for r in rows}


def canonical_order() -> list:
    """规范题序：p_gold_contexts.jsonl 行序（= 题集顺序）。"""
    return [json.loads(l)["question_id"] for l in open(GOLD_CTX_FILE, encoding="utf-8")]


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
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


def parse_scoring(response: str):
    """返回 (raw_data, scores_dict or None)。scores 5 维均为 0-100 float。"""
    data = extract_json_obj(response)
    if isinstance(data, dict):
        scores = {}
        ok = True
        for k in SCORING_METRICS:
            v = data.get(k)
            if isinstance(v, dict):
                s = v.get("Score")
            elif isinstance(v, (int, float)):
                s = v
            else:
                s = None
            try:
                scores[k] = float(s)
            except (TypeError, ValueError):
                scores[k] = None
                ok = False
        if ok:
            return data, scores
    # 正则兜底
    nums = re.findall(r'"Score":\s*(?:"?(\d+(?:\.\d+)?)"?)', response or "")
    if len(nums) >= len(SCORING_METRICS):
        scores = {k: float(n) for k, n in zip(SCORING_METRICS, nums)}
        return data, scores
    return data, None


def parse_selection(response: str):
    """返回 (raw_data, winners_dict or None)。winners: metric -> 1(Answer1赢)/2(Answer2赢)。"""
    data = extract_json_obj(response)
    winners = {}
    if isinstance(data, dict):
        ok = True
        for k in SELECTION_METRICS:
            v = data.get(k)
            w = v.get("Winner") if isinstance(v, dict) else None
            if w is None:
                winners[k] = None
                ok = False
            elif re.search(r"answer\s*1", str(w), re.I):
                winners[k] = 1
            elif re.search(r"answer\s*2", str(w), re.I):
                winners[k] = 2
            else:
                winners[k] = None
                ok = False
        if ok:
            return data, winners
    # 正则兜底
    ws = re.findall(r'"Winner":\s*"([^"]+)"', response or "")
    if len(ws) >= len(SELECTION_METRICS):
        for k, w in zip(SELECTION_METRICS, ws):
            if re.search(r"answer\s*1", w, re.I):
                winners[k] = 1
            elif re.search(r"answer\s*2", w, re.I):
                winners[k] = 2
            else:
                winners[k] = None
    return data, winners


# ---------------------------------------------------------------------------
# 落盘（断点续跑）
# ---------------------------------------------------------------------------
_write_lock = threading.Lock()


def append_row(path: Path, row: dict):
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    for l in open(path, encoding="utf-8"):
        try:
            ids.add(json.loads(l)["question_id"])
        except Exception:
            continue
    return ids


# ---------------------------------------------------------------------------
# 任务执行
# ---------------------------------------------------------------------------
def run_scoring(routes, snapshot, smoke, concurrency, temperature, max_tokens):
    gold_ctx = load_gold_ctx()
    order = canonical_order()
    if smoke:
        order = order[:smoke]
    total_planned = 0
    for route in routes:
        rf = result_file(route, snapshot)
        out = JUDGE_DIR / "scoring" / f"{route}_scoring.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = done_ids(out)
        results = load_result_jsonl(rf)
        qids = [q for q in order if q in results and q not in done and q in gold_ctx]
        total_planned += len(qids)

        def work(qid):
            row = results[qid]
            prompt = SCORING_PROMPT.format(
                reference=gold_ctx[qid], query=row["query"], answer=row["result"]
            )
            resp = llm_model_func(prompt, SCORING_SYS_PROMPT,
                                  temperature=temperature, max_tokens=max_tokens)
            _, scores = parse_scoring(resp)
            rec = {
                "question_id": qid, "route": route, "query": row["query"],
                "response": resp, "scores": scores, "parse_ok": scores is not None,
            }
            append_row(out, rec)
            return qid, scores is not None

        print(f"[scoring] {route}: 计划 {len(qids)} 题 -> {out}")
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(work, q): q for q in qids}
            for fut in tqdm(as_completed(futs), total=len(qids), desc=f"scoring/{route}"):
                try:
                    fut.result()
                except Exception as e:
                    print(f"[scoring/{route}] 失败 {futs[fut]}: {e!r}")
    print(f"[scoring] 完成，共计划 {total_planned} 题")


def run_selection(snapshot, smoke, concurrency, temperature, max_tokens):
    order = canonical_order()
    if smoke:
        order = order[:smoke]
    total_planned = 0
    gold_results = load_result_jsonl(result_file("P_gold", snapshot))
    for route in CANDIDATE_ROUTES:
        results = load_result_jsonl(result_file(route, snapshot))
        for direction, a_route, b_route, out_name in (
            ("fwd", route, "P_gold", f"{route}_vs_P_gold_fwd.jsonl"),
            ("rev", "P_gold", route, f"{route}_vs_P_gold_rev.jsonl"),
        ):
            out = JUDGE_DIR / "selection" / out_name
            out.parent.mkdir(parents=True, exist_ok=True)
            done = done_ids(out)
            a_res, b_res = (results, gold_results) if a_route == route else (gold_results, results)
            qids = [q for q in order if q in a_res and q in b_res and q not in done]
            total_planned += len(qids)

            def work(qid):
                a_row, b_row = a_res[qid], b_res[qid]
                prompt = SELECTION_PROMPT.format(
                    query=a_row["query"], answer1=a_row["result"], answer2=b_row["result"]
                )
                resp = llm_model_func(prompt, SELECTION_SYS_PROMPT,
                                      temperature=temperature, max_tokens=max_tokens)
                _, winners = parse_selection(resp)
                # fwd: candidate 在 Answer1；rev: candidate 在 Answer2
                cand_wins = {}
                if winners:
                    for k, w in winners.items():
                        if w is None:
                            cand_wins[k] = None
                        elif direction == "fwd":
                            cand_wins[k] = (w == 1)
                        else:
                            cand_wins[k] = (w == 2)
                rec = {
                    "question_id": qid, "a_route": a_route, "b_route": b_route,
                    "direction": direction, "response": resp,
                    "winners": winners, "candidate_wins": cand_wins,
                    "parse_ok": winners is not None,
                }
                append_row(out, rec)
                return qid, winners is not None

            print(f"[selection] {a_route} vs {b_route} ({direction}): 计划 {len(qids)} 题 -> {out}")
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(work, q): q for q in qids}
                for fut in tqdm(as_completed(futs), total=len(qids), desc=f"selection/{out_name}"):
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"[selection/{out_name}] 失败 {futs[fut]}: {e!r}")
    print(f"[selection] 完成，共计划 {total_planned} 题")


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------
def build_summary():
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    summary_dir = JUDGE_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # --- scoring ---
    scoring_summary = {}
    all_scoring = {}
    for route in ROUTES:
        f = JUDGE_DIR / "scoring" / f"{route}_scoring.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        ok_rows = [r for r in rows if r.get("parse_ok")]
        all_scoring[route] = ok_rows
        metric_vals = {m: [] for m in SCORING_METRICS}
        for r in ok_rows:
            for m in SCORING_METRICS:
                v = (r.get("scores") or {}).get(m)
                if v is not None:
                    metric_vals[m].append(v)
        scoring_summary[route] = {
            "n_total": len(rows),
            "n_parsed": len(ok_rows),
            "parse_rate": round(len(ok_rows) / len(rows), 4) if rows else 0.0,
            "metrics": {
                m: (round(float(np.mean(v)), 2) if v else None)
                for m, v in metric_vals.items()
            },
            "average": (
                round(float(np.mean([v for m in SCORING_METRICS for v in metric_vals[m]])), 2)
                if any(metric_vals.values()) else None
            ),
        }
    with open(summary_dir / "scoring_summary.json", "w", encoding="utf-8") as f:
        json.dump(scoring_summary, f, ensure_ascii=False, indent=2)

    # --- selection（双向平均）---
    selection_summary = {}
    for route in CANDIDATE_ROUTES:
        fwd_f = JUDGE_DIR / "selection" / f"{route}_vs_P_gold_fwd.jsonl"
        rev_f = JUDGE_DIR / "selection" / f"{route}_vs_P_gold_rev.jsonl"
        if not fwd_f.exists() or not rev_f.exists():
            continue
        fwd_rows = [json.loads(l) for l in open(fwd_f, encoding="utf-8")]
        rev_rows = [json.loads(l) for l in open(rev_f, encoding="utf-8")]
        n = min(len(fwd_rows), len(rev_rows))
        if n == 0:
            continue
        by_qid = {}
        for r in fwd_rows:
            if r.get("parse_ok"):
                by_qid.setdefault(r["question_id"], {})["fwd"] = r.get("candidate_wins") or {}
        for r in rev_rows:
            if r.get("parse_ok"):
                by_qid.setdefault(r["question_id"], {})["rev"] = r.get("candidate_wins") or {}
        # 只统计双向都解析成功的题
        paired = {q: v for q, v in by_qid.items() if "fwd" in v and "rev" in v}
        metric_win = {m: [] for m in SELECTION_METRICS}
        overall_win = []
        for q, v in paired.items():
            for m in SELECTION_METRICS:
                a, b = v["fwd"].get(m), v["rev"].get(m)
                if a is not None and b is not None:
                    metric_win[m].append((a + b) / 2)  # 1=candidate赢, 0=candidate输, 0.5=tie 平均
            a = v["fwd"].get("Overall") if "Overall" in v["fwd"] else None
            b = v["rev"].get("Overall") if "Overall" in v["rev"] else None
        # 总体 winner 用八维平均投票近似（selection prompt 无独立 Overall 字段时）
        # 若 LLM 输出了 Overall 字段则优先
        overall_by_qid = {}
        for q, v in paired.items():
            cand = 0.0
            cnt = 0
            for m in SELECTION_METRICS:
                a, b = v["fwd"].get(m), v["rev"].get(m)
                if a is not None and b is not None:
                    cand += (a + b) / 2
                    cnt += 1
            if cnt:
                overall_by_qid[q] = cand / cnt
        selection_summary[route] = {
            "n_fwd": len(fwd_rows), "n_rev": len(rev_rows),
            "n_paired_parsed": len(paired),
            "metrics_win_rate": {
                m: (round(float(np.mean(v)), 4) if v else None)
                for m, v in metric_win.items()
            },
            "overall_win_rate": (
                round(float(np.mean(list(overall_by_qid.values()))), 4)
                if overall_by_qid else None
            ),
            "avg_8dim_win_rate": (
                round(float(np.mean([v for m in SELECTION_METRICS for v in metric_win[m]])), 4)
                if any(metric_win.values()) else None
            ),
        }
    with open(summary_dir / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(selection_summary, f, ensure_ascii=False, indent=2)

    return scoring_summary, selection_summary


def build_report(scoring_summary, selection_summary):
    lines = []
    lines.append("# Pilot 六路径质量判定报告（Judge）\n")
    lines.append(f"- judge 模型: {LLM_MODEL} @ {LLM_BASE_URL}")
    lines.append(f"- 判定维度: 五维打分（论文口径）+ 8 维 pairwise（双向平均）")
    lines.append(f"- 参考文档: 每题统一使用 gold context（p_gold_contexts.jsonl）\n")
    lines.append("## 一、五维打分（0-100，越高越好）\n")
    lines.append("| 路径 | 题数(解析/总) | Comprehensiveness | Diversity | Empowerment | Logical | Readability | 平均 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for route in ROUTES:
        s = scoring_summary.get(route)
        if not s:
            continue
        m = s["metrics"]
        row = [route, f"{s['n_parsed']}/{s['n_total']}"]
        row += [f"{m[k]:.1f}" if m.get(k) is not None else "-" for k in SCORING_METRICS]
        row.append(f"{s['average']:.1f}" if s["average"] is not None else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n## 二、Pairwise 胜率（候选路径 vs P_gold，双向平均，越高越接近 gold）\n")
    lines.append("| 候选路径 | 配对题数 | 8 维平均胜率 | 总体胜率 |")
    lines.append("|---|---|---|---|")
    for route in CANDIDATE_ROUTES:
        s = selection_summary.get(route)
        if not s:
            continue
        lines.append(
            f"| {route} | {s['n_paired_parsed']} | "
            f"{s['avg_8dim_win_rate']:.2%}" if s["avg_8dim_win_rate"] is not None else "-" + " | "
            f"{s['overall_win_rate']:.2%}" if s["overall_win_rate"] is not None else "-" + " |"
        )
    lines.append("\n## 三、结论要点\n")
    lines.append("（待人工补充：打分最高的路径是否 P_gold、各路径与 gold 差距排序等）\n")
    report_path = JUDGE_DIR / "judge_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告已写入: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pilot 六路径质量判定（Judge）")
    parser.add_argument("--mode", choices=["scoring", "selection", "summary", "all"],
                        default="all", help="执行模式（默认 all）")
    parser.add_argument("--routes", nargs="+", default=ROUTES,
                        help=f"scoring 的路径列表（默认 {' '.join(ROUTES)}）")
    parser.add_argument("--snapshot", default=SNAPSHOT_DEFAULT,
                        help="system snapshot id（默认当前有效快照）")
    parser.add_argument("--smoke", type=int, default=0,
                        help="每任务只跑前 N 题（0=全量）")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="并发调用数（默认 1，串行；vLLM 建议 2-4）")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=3000)
    args = parser.parse_args()

    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode in ("scoring", "all"):
        run_scoring(args.routes, args.snapshot, args.smoke,
                    args.concurrency, args.temperature, args.max_tokens)
    if args.mode in ("selection", "all"):
        run_selection(args.snapshot, args.smoke,
                      args.concurrency, args.temperature, args.max_tokens)
    if args.mode in ("summary", "all"):
        s_s, s_sel = build_summary()
        build_report(s_s, s_sel)


if __name__ == "__main__":
    main()
