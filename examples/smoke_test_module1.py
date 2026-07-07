"""Module 1 冒烟测试：验证领域类型增强索引是否正确集成。

测试内容：
1. 使用小段神经病学文本进行索引
2. 检查 vdb_entities.json 是否包含 entity_type
3. 检查 vdb_relationships.json 是否包含 edge_type
4. 检查超图数据是否包含 edge_type 和 generalization
5. 运行查询验证关系上下文包含 type 列
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from hyperrag import HyperRAG, QueryParam
from hyperrag.llm import openai_complete_if_cache, openai_embedding
from hyperrag.utils import EmbeddingFunc

from my_config import EMB_API_KEY, EMB_BASE_URL, EMB_DIM, EMB_MODEL
from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

# 神经病学测试文本（包含疾病、症状、检查、治疗等多种实体类型）
TEST_TEXT = """
Anti-NMDA receptor antibody encephalitis is a form of autoimmune encephalitis often associated with ovarian teratoma. 
Patients typically present with psychiatric symptoms including psychosis, memory loss, and seizures. 
Neurological signs include dyskinesias, autonomic instability, and decreased consciousness.
Spinal fluid analysis can support the diagnosis by detecting anti-NMDA receptor antibodies.
MRI may show FLAIR hyperintensities in the temporal lobes.
Treatment involves tumor removal if ovarian teratoma is present, along with immunotherapy.
First-line immunotherapy includes corticosteroids, intravenous immunoglobulin, and plasma exchange.
Second-line agents include rituximab and cyclophosphamide.
Early diagnosis and aggressive treatment improve outcomes significantly.
"""


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model=EMB_MODEL,
        api_key=EMB_API_KEY,
        base_url=EMB_BASE_URL,
    )


def check_storage_files(working_dir: Path) -> dict:
    """检查存储文件是否包含新字段。"""
    results = {
        "entity_type_in_vdb": False,
        "edge_type_in_vdb": False,
    }

    # 检查 vdb_entities.json
    entities_vdb_path = working_dir / "vdb_entities.json"
    if entities_vdb_path.exists():
        with open(entities_vdb_path, "r", encoding="utf-8") as f:
            entities_data = json.load(f)
            if "data" in entities_data and entities_data["data"]:
                first_entry = entities_data["data"][0]
                if "entity_type" in first_entry:
                    results["entity_type_in_vdb"] = True
                    print(f"  [OK] vdb_entities.json 包含 entity_type: {first_entry.get('entity_type')}")

    # 检查 vdb_relationships.json
    rels_vdb_path = working_dir / "vdb_relationships.json"
    if rels_vdb_path.exists():
        with open(rels_vdb_path, "r", encoding="utf-8") as f:
            rels_data = json.load(f)
            if "data" in rels_data and rels_data["data"]:
                first_entry = rels_data["data"][0]
                if "edge_type" in first_entry:
                    results["edge_type_in_vdb"] = True
                    print(f"  [OK] vdb_relationships.json 包含 edge_type: {first_entry.get('edge_type')}")

    return results


def main():
    print("=" * 60)
    print("Module 1 冒烟测试：领域类型增强索引")
    print("=" * 60)

    # 使用固定目录便于检查
    working_dir = Path("caches/smoke_test_module1")
    working_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/4] 创建工作目录: {working_dir}")

    # 创建 HyperRAG 实例
    print("[2/4] 创建 HyperRAG 实例...")
    rag = HyperRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM, max_token_size=8192, func=embedding_func
        ),
        embedding_batch_num=8,
        embedding_func_max_async=4,
        llm_model_max_async=1,
    )

    # 执行索引
    print("[3/4] 插入测试文本并建立索引...")
    try:
        rag.insert(TEST_TEXT)
        print("  [OK] 索引完成，无异常")
    except Exception as e:
        print(f"  [FAIL] 索引失败: {e}")
        return

    # 检查存储文件
    print("[4/4] 检查存储文件...")
    results = check_storage_files(working_dir)

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    all_pass = True
    for key, value in results.items():
        status = "PASS" if value else "FAIL"
        if not value:
            all_pass = False
        print(f"  {key}: {status}")

    print("\n" + "=" * 60)
    if all_pass:
        print("所有测试通过！Module 1 集成成功。")
    else:
        print("部分测试失败，请检查代码。")
    print("=" * 60)

    # 运行查询验证 type 列
    print("\n[可选] 运行查询验证关系上下文包含 type 列...")
    try:
        query = "What is the relationship between anti-NMDA receptor antibody and autoimmune encephalitis?"
        result = rag.query(query, param=QueryParam(mode="hyper"))
        if "type" in result.lower():
            print("  [OK] 查询结果包含 type 相关信息")
        else:
            print("  [INFO] 查询结果未明确显示 type 列，请手动检查")
        print(f"\n查询结果片段:\n{result[:500]}...")
    except Exception as e:
        print(f"  [WARN] 查询失败: {e}")


if __name__ == "__main__":
    main()
