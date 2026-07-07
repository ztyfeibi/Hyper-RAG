"""验证 LLM 调用是否能正确禁用 thinking 输出，并返回可解析的结果。

测试三个场景：
1. extra_body={"enable_thinking": False} （当前代码的做法）
2. chat_template_kwargs={"enable_thinking": False} （vLLM 的另一种传法）
3. 不传任何参数（默认行为）

对每个场景检查：
- 返回中是否包含 <think>...</think> 标记
- 去掉 thinking 后能否正确 JSON 解析
- 能否正确解析 entity_extraction 格式的记录
"""
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI
from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def strip_thinking(text: str) -> str:
    """去掉 LLM 返回中的 thinking 部分。

    支持两种格式：
    - <think>...</think>  (Qwen3 标准格式)
    - Thinking Process:... </think> (非标准格式)
    """
    # 标准格式：<think>...</think>
    m = re.search(r"</think>\s*", text)
    if m:
        return text[m.end():]
    # 非标准：以 Thinking 开头但没有闭合标签
    if text.strip().startswith("Thinking"):
        # 找第一个 ( 或 { 开头的行
        for marker in ["(", "{"]:
            idx = text.find(marker)
            if idx != -1:
                return text[idx:]
    return text


async def test_simple_json(client: dict, label: str, **create_kwargs):
    """测试简单 JSON 输出能否被正确解析。"""
    prompt = (
        'You are a helpful assistant. Return ONLY a JSON object, no extra text.\n'
        'Output the content in the following structure:\n'
        '{"Question": "What is the main symptom of migraine?"}'
    )

    try:
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            **create_kwargs,
        )
        content = resp.choices[0].message.content

        has_think = "<think>" in content or "</think>" in content
        stripped = strip_thinking(content)
        stripped_has_think = "<think>" in stripped or "</think>" in stripped

        # 尝试 JSON 解析原始内容
        brace = content.find("{")
        json_ok_raw = False
        if brace != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(content[brace:])
                json_ok_raw = isinstance(obj, dict) and "Question" in obj
            except json.JSONDecodeError:
                pass

        # 尝试 JSON 解析去掉 thinking 后的内容
        brace2 = stripped.find("{")
        json_ok_stripped = False
        if brace2 != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(stripped[brace2:])
                json_ok_stripped = isinstance(obj, dict) and "Question" in obj
            except json.JSONDecodeError:
                pass

        print(f"\n[{label}]")
        print(f"  原始返回长度: {len(content)} chars")
        print(f"  原始包含 thinking: {has_think}")
        print(f"  去除后包含 thinking: {stripped_has_think}")
        print(f"  原始可JSON解析: {json_ok_raw}")
        print(f"  去除后可JSON解析: {json_ok_stripped}")
        if json_ok_stripped:
            print(f"  Question: {obj['Question'][:80]}")
        return json_ok_stripped
    except Exception as e:
        print(f"\n[{label}]")
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False


async def test_entity_extraction(client: dict, label: str, **create_kwargs):
    """测试 Step_1 实际的 entity_extraction 格式能否被正确解析。"""
    prompt = (
        'Information: "偏头痛是一种常见的原发性头痛性疾病，表现为反复发作的中重度头痛，'
        '常伴有恶心、呕吐，对光和声音敏感。"\n'
        "################\n"
        "Given the information above, please extract the entities and relationships, "
        "as well as the high-order relationships.\n\n"
        "Output the content in the following structure:\n"
        '("entity" | "entity_type" | "entity_description")\n'
        '("relationship" | "entity1" | "entity2" | "edge_type" | "relationship_description")\n\n'
        "Examples:\n"
        '("entity" | "偏头痛" | "DISEASE" | "一种常见的原发性头痛性疾病")\n'
        '("relationship" | "偏头痛" | "头痛" | "HAS_SYMPTOM" | "偏头痛表现为头痛")\n\n'
        "Output ONLY the records, one per line. End with <|COMPLETE|>"
    )

    try:
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            **create_kwargs,
        )
        content = resp.choices[0].message.content

        has_think = "<think>" in content or "</think>" in content
        stripped = strip_thinking(content)

        # 模拟 indexing.py 的解析逻辑
        record_delimiter = "\n"
        completion_delimiter = "<|COMPLETE|>"
        records = re.split(
            "|".join(re.escape(m) for m in [record_delimiter, completion_delimiter]),
            stripped,
        )
        records = [r.strip() for r in records if r.strip()]

        # 统计能解析出 (entity ...) 或 (relationship ...) 的记录数
        entity_count = 0
        rel_count = 0
        for record in records:
            m = re.search(r"\((.*)\)", record)
            if m is None:
                continue
            inner = m.group(1)
            parts = [p.strip().strip('"') for p in inner.split("|")]
            if len(parts) >= 2 and parts[0] == "entity":
                entity_count += 1
            elif len(parts) >= 3 and parts[0] == "relationship":
                rel_count += 1

        print(f"\n[{label}]")
        print(f"  原始返回长度: {len(content)} chars")
        print(f"  原始包含 thinking: {has_think}")
        print(f"  去除后记录数: {len(records)}")
        print(f"  解析出 entities: {entity_count}")
        print(f"  解析出 relationships: {rel_count}")
        if entity_count > 0 or rel_count > 0:
            print(f"  前3条记录:")
            for r in records[:3]:
                print(f"    {r[:120]}")
        return entity_count > 0 or rel_count > 0
    except Exception as e:
        print(f"\n[{label}]")
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False


async def main():
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    print("=" * 60)
    print(f"模型: {LLM_MODEL}")
    print(f"服务: {LLM_BASE_URL}")
    print("=" * 60)

    results = {}

    # 测试1: extra_body enable_thinking=False (当前代码的做法)
    r1a = await test_simple_json(
        client, "1a) JSON + extra_body enable_thinking=False",
        extra_body={"enable_thinking": False},
    )
    r1b = await test_entity_extraction(
        client, "1b) Entity + extra_body enable_thinking=False",
        extra_body={"enable_thinking": False},
    )
    results["extra_body"] = (r1a, r1b)

    # 测试2: chat_template_kwargs enable_thinking=False
    r2a = await test_simple_json(
        client, "2a) JSON + chat_template_kwargs enable_thinking=False",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    r2b = await test_entity_extraction(
        client, "2b) Entity + chat_template_kwargs enable_thinking=False",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    results["chat_template_kwargs"] = (r2a, r2b)

    # 测试3: 不传任何参数 + strip_thinking 后处理
    r3a = await test_simple_json(
        client, "3a) JSON + 无参数 + strip_thinking后处理",
    )
    r3b = await test_entity_extraction(
        client, "3b) Entity + 无参数 + strip_thinking后处理",
    )
    results["no_param_strip"] = (r3a, r3b)

    # 汇总
    print("\n" + "=" * 60)
    print("汇总:")
    print(f"  1) extra_body enable_thinking=False:      JSON={r1a}, Entity={r1b}")
    print(f"  2) chat_template_kwargs enable_thinking:   JSON={r2a}, Entity={r2b}")
    print(f"  3) 无参数 + strip_thinking后处理:           JSON={r3a}, Entity={r3b}")


if __name__ == "__main__":
    asyncio.run(main())
