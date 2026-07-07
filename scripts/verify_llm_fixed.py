"""验证修复后的 LLM 调用：确认 thinking 被正确禁用且结果可解析。"""
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI
from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


async def main():
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    print("=" * 60)
    print("验证修复后: chat_template_kwargs enable_thinking=False")
    print("模型: {}  服务: {}".format(LLM_MODEL, LLM_BASE_URL))
    print("=" * 60)

    # 测试1: 简单 JSON
    prompt1 = (
        'You are a helpful assistant. Return ONLY a JSON object, no extra text.\n'
        'Output the content in the following structure:\n'
        '{"Question": "What is the main symptom of migraine?"}'
    )
    resp1 = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt1}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    c1 = resp1.choices[0].message.content
    has_think1 = "<think>" in c1 or "</think>" in c1
    print("\n1) 简单 JSON 输出:")
    print("   返回长度: {} chars (修复前 2374 chars)".format(len(c1)))
    print("   包含 thinking: {} (修复前 True)".format(has_think1))
    print("   完整返回: {}".format(c1))
    brace = c1.find("{")
    json_ok1 = False
    if brace != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(c1[brace:])
            json_ok1 = "Question" in obj
            print("   JSON 解析: OK, Question = {}".format(obj.get("Question", "")[:80]))
        except json.JSONDecodeError as e:
            print("   JSON 解析失败: {}".format(e))

    # 测试2: 实体抽取格式
    prompt2 = (
        'Information: "偏头痛是一种常见的原发性头痛性疾病，表现为反复发作的中重度头痛，'
        '常伴有恶心、呕吐。"\n'
        "################\n"
        "Given the information above, please extract the entities and relationships.\n\n"
        "Output the content in the following structure:\n"
        '("entity" | "entity_type" | "entity_description")\n'
        '("relationship" | "entity1" | "entity2" | "edge_type" | "description")\n\n'
        "Examples:\n"
        '("entity" | "偏头痛" | "DISEASE" | "一种常见的原发性头痛性疾病")\n'
        '("relationship" | "偏头痛" | "头痛" | "HAS_SYMPTOM" | "偏头痛表现为头痛")\n\n'
        "Output ONLY the records, one per line. End with <|COMPLETE|>"
    )
    resp2 = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt2}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    c2 = resp2.choices[0].message.content
    has_think2 = "<think>" in c2 or "</think>" in c2
    print("\n2) 实体抽取输出:")
    print("   返回长度: {} chars (修复前 17509 chars)".format(len(c2)))
    print("   包含 thinking: {} (修复前 True)".format(has_think2))

    records = [r.strip() for r in re.split(r"\n|<\|COMPLETE\|>", c2) if r.strip()]
    ent_cnt = 0
    rel_cnt = 0
    for record in records:
        m = re.search(r"\((.*)\)", record)
        if m:
            parts = [p.strip().strip('"') for p in m.group(1).split("|")]
            if len(parts) >= 2 and parts[0] == "entity":
                ent_cnt += 1
            elif len(parts) >= 3 and parts[0] == "relationship":
                rel_cnt += 1
    print("   记录数: {}, entities: {}, relationships: {}".format(len(records), ent_cnt, rel_cnt))
    print("   前5条记录:")
    for r in records[:5]:
        print("     " + r[:120])

    print("\n" + "=" * 60)
    print("结论:")
    print("  - thinking 已禁用: {}".format(not has_think1 and not has_think2))
    print("  - JSON 可解析: {}".format(json_ok1))
    print("  - 实体抽取可解析: {}".format(ent_cnt > 0 and rel_cnt > 0))
    print("  - 返回长度大幅缩短 (JSON: 2374->{}, Entity: 17509->{})".format(len(c1), len(c2)))
    step1_ok = ent_cnt > 0 and rel_cnt > 0 and not has_think2
    print("  - Step_1 建库可用: {}".format(step1_ok))

    if step1_ok:
        print("\n  ✅ LLM 调用验证通过，可以开始建库！")
    else:
        print("\n  ❌ 仍有问题，需要进一步排查")


if __name__ == "__main__":
    asyncio.run(main())
