import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# reproduce/ 目录里的脚本需要导入项目根目录下的 hyperrag 包。
# 直接运行 `python reproduce/Step_1.py` 时，默认 import 路径不一定包含项目根目录，
# 所以这里把父目录加入 sys.path。
sys.path.append(str(Path(__file__).resolve().parent.parent))

from hyperrag import HyperRAG
from hyperrag.llm import openai_complete_if_cache, openai_embedding
from hyperrag.utils import EmbeddingFunc
from my_config import EMB_API_KEY, EMB_BASE_URL, EMB_DIM, EMB_MODEL
from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
try:
    from .pipeline_defaults import DATA_NAME as DEFAULT_DATA_NAME
except ImportError:
    from pipeline_defaults import DATA_NAME as DEFAULT_DATA_NAME


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """给 HyperRAG 注入的 LLM 调用函数。

    HyperRAG 在实体抽取、关系抽取、摘要和最终回答时会调用这个函数。
    具体模型、base_url、api_key 都来自 my_config.py。
    """
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
    """给 HyperRAG 注入的 embedding 函数。

    Step_1 建库时会把 chunks、实体、关系写入向量库；
    Step_3 查询时也会用同一套 embedding 配置做相似度检索。
    """
    return await openai_embedding(
        texts,
        model=EMB_MODEL,
        api_key=EMB_API_KEY,
        base_url=EMB_BASE_URL,
    )


def insert_text(rag, file_path, retries=0, max_retries=3, limit=None, batch_size=None):
    """读取 Step_0 生成的 context 文件，并调用 HyperRAG.insert 建索引。

    file_path 默认形如：
    caches/<data_name>/contexts/<data_name>_unique_contexts.json

    rag.insert 内部会完成：
    1. 文本切块；
    2. chunk 向量入库；
    3. LLM 抽取实体和低阶/高阶超边；
    4. 写入实体向量库、关系向量库和 hypergraph hgdb 文件。

    limit: 若不为 None，只取前 limit 条 context 拼成文本，用于小规模冒烟测试。

    batch_size: 若不为 None，将 context 列表分批插入，每批 batch_size 条。
    分批插入可避免一次性处理过多 chunk 导致内存溢出（2317 chunk 全量会崩）。
    每批结束后 HyperRAG 自动落盘（_insert_done），崩溃只丢当前批。
    断点续跑：已处理的 doc/chunk 会被 filter_keys 自动跳过。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        contexts = json.load(f)

    if limit is not None:
        contexts = contexts[:limit]
        print(f"[Smoke test] Using first {limit} contexts")

    if batch_size is None:
        # 不分批：全部拼成一个字符串，单次 insert（仅适用于小规模）
        unique_contexts = "".join(contexts)
        print(f"Inserting {len(contexts)} contexts as single doc "
              f"({len(unique_contexts)} chars)")
        while retries < max_retries:
            try:
                rag.insert(unique_contexts)
                break
            except Exception as e:
                retries += 1
                print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
                time.sleep(30)
        if retries == max_retries:
            raise RuntimeError(
                f"Insertion failed after {max_retries} retries; "
                "stop to avoid producing an incomplete cache"
            )
    else:
        # 分批插入：每批 batch_size 条 context，独立 insert + 落盘
        total_batches = (len(contexts) + batch_size - 1) // batch_size
        print(f"[Batch mode] {len(contexts)} contexts -> {total_batches} batches "
              f"({batch_size} contexts/batch)")
        for i in range(0, len(contexts), batch_size):
            batch_num = i // batch_size + 1
            batch_contexts = contexts[i:i + batch_size]
            batch_text = "".join(batch_contexts)
            print(f"\n{'='*60}")
            print(f"[Batch {batch_num}/{total_batches}] {len(batch_contexts)} contexts, "
                  f"{len(batch_text)} chars")
            print(f"{'='*60}")
            while retries < max_retries:
                try:
                    rag.insert(batch_text)
                    break
                except Exception as e:
                    retries += 1
                    print(f"Batch {batch_num} failed, retrying "
                          f"({retries}/{max_retries}), error: {e}")
                    time.sleep(30)
            if retries == max_retries:
                raise RuntimeError(
                    f"Batch {batch_num} failed after {max_retries} retries; "
                    "stop to avoid producing an incomplete cache"
                )
            retries = 0  # reset for next batch
        print(f"\n{'='*60}")
        print(f"All {total_batches} batches completed")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 Step_0 生成的 context JSON 写入 HyperRAG 索引")
    parser.add_argument(
        "--data-name",
        type=str,
        default=DEFAULT_DATA_NAME,
        help=f"工作目录 caches/<name>（默认 {DEFAULT_DATA_NAME!r}）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只取前 N 条 context 做小规模冒烟测试（默认全量）",
    )
    parser.add_argument(
        "--source-data-name",
        type=str,
        default=None,
        help="读取 context 文件的 data_name（默认与 --data-name 相同）。"
             "用于重建 chunk 时复用已有 context 文件，例如"
             " --data-name=neurology_chunk1000 --source-data-name=neurology",
    )
    parser.add_argument(
        "--chunk-token-size",
        type=int,
        default=2400,
        help="每个 chunk 的 token 上限（默认 2400）",
    )
    parser.add_argument(
        "--chunk-overlap-token-size",
        type=int,
        default=120,
        help="相邻 chunk 之间的重叠 token 数（默认 120）",
    )
    parser.add_argument(
        "--gleaning",
        type=int,
        default=0,
        help="实体抽取最大 gleaning 轮数（默认 0，即关闭）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="分批插入的 context 条数（默认 None=不分批）。"
             "全量 chunk=1000 时 2317 chunk 一次性处理会内存溢出，"
             "建议设 3000（约 580 chunk/批，4-5 批）",
    )
    args = parser.parse_args()
    data_name = args.data_name
    source_data_name = args.source_data_name or data_name

    # HyperRAG 的所有持久化产物都会落在这个目录下：
    # kv_store_full_docs.json、kv_store_text_chunks.json、vdb_*.json、
    # hypergraph_chunk_entity_relation.hgdb、HyperRAG.log 等。
    WORKING_DIR = Path("caches") / data_name
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    # 实例化核心方法类。Step_1 的职责不是实现算法，而是把配置和输入语料
    # 交给 hyperrag.HyperRAG，让它完成建库。
    rag = HyperRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM, max_token_size=8192, func=embedding_func
        ),
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap_token_size,
        # 内网 vLLM 无 429 限流风险，适度提高并发加速建库
        llm_model_max_async=4,
        embedding_func_max_async=4,
        entity_extract_max_gleaning=args.gleaning,
    )

    # 读取 Step_0 的输出，并开始构建 HyperRAG 所需的全部索引和超图数据。
    insert_text(
        rag,
        f"caches/{source_data_name}/contexts/{source_data_name}_unique_contexts.json",
        limit=args.limit,
        batch_size=args.batch_size,
    )

