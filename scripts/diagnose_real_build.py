"""用真实 HyperRAG 实例跑 5 个 chunk，捕获实际异常。"""
import asyncio
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperrag import HyperRAG
from hyperrag.llm import openai_complete_if_cache, openai_embedding
from hyperrag.utils import EmbeddingFunc
from my_config import EMB_API_KEY, EMB_BASE_URL, EMB_DIM, EMB_MODEL
from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        **kwargs,
    )


async def embedding_func(texts):
    return await openai_embedding(
        texts,
        model=EMB_MODEL,
        api_key=EMB_API_KEY,
        base_url=EMB_BASE_URL,
    )


async def main():
    rag = HyperRAG(
        working_dir=Path("caches/neurology"),
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM, max_token_size=8192, func=embedding_func
        ),
        chunk_token_size=2400,
        chunk_overlap_token_size=120,
        llm_model_max_async=4,
        embedding_func_max_async=4,
    )

    # 用 50 个 context 做中等规模测试（会产生多个 chunk）
    with open("caches/neurology/contexts/neurology_unique_contexts.json", "r", encoding="utf-8") as f:
        contexts = json.load(f)
    test_text = "".join(contexts[:50])
    print(f"Test text length: {len(test_text)} chars")

    try:
        print("Starting insert...")
        await rag.ainsert(test_text)
        print("Insert completed successfully!")
    except Exception as e:
        print(f"\nEXCEPTION: {type(e).__name__}: {e}")
        print(f"\nFull traceback:\n{traceback.format_exc()}")

    # 检查结果
    print(f"\nHypergraph vertices: {rag.chunk_entity_relation_hypergraph._hg.num_v}")
    print(f"Hypergraph hyperedges: {rag.chunk_entity_relation_hypergraph._hg.num_e}")


if __name__ == "__main__":
    asyncio.run(main())
