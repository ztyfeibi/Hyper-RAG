"""冒烟测试：验证建库后 naive 和 hyper 模式查询链路是否完整。

用一个问题分别测试 naive 和 hyper 模式，检查：
1. embedding 查询是否正常返回
2. kv_store 回查是否正常
3. LLM 回答生成是否正常
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from hyperrag import HyperRAG
from hyperrag.llm import openai_complete_if_cache, openai_embedding
from hyperrag.utils import EmbeddingFunc
from my_config import EMB_API_KEY, EMB_BASE_URL, EMB_DIM, EMB_MODEL
from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        LLM_MODEL, prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        **kwargs,
    )


async def embedding_func(texts):
    return await openai_embedding(
        texts, model=EMB_MODEL, api_key=EMB_API_KEY, base_url=EMB_BASE_URL,
    )


async def main():
    from pathlib import Path as P
    rag = HyperRAG(
        working_dir=P("caches/neurology"),
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM, max_token_size=8192, func=embedding_func,
        ),
        chunk_token_size=2400,
        chunk_overlap_token_size=120,
        llm_model_max_async=1,
        embedding_func_max_async=4,
    )

    test_question = "What is migraine and what are its main symptoms?"

    print("=" * 60)
    print("冒烟测试: 查询链路验证")
    print("问题: {}".format(test_question))
    print("=" * 60)

    # 1. naive 模式
    print("\n--- 1) naive 模式 ---")
    try:
        from hyperrag.base import QueryParam
        param = QueryParam(mode="naive")
        result = await rag.aquery(test_question, param)
        print("  状态: OK")
        print("  回答长度: {} chars".format(len(result)))
        print("  回答前200字: {}".format(result[:200]))
    except Exception as e:
        print("  状态: FAILED")
        print("  错误: {}: {}".format(type(e).__name__, e))

    # 2. hyper 模式
    print("\n--- 2) hyper 模式 ---")
    try:
        from hyperrag.base import QueryParam
        param = QueryParam(mode="hyper")
        result = await rag.aquery(test_question, param)
        print("  状态: OK")
        print("  回答长度: {} chars".format(len(result)))
        print("  回答前200字: {}".format(result[:200]))
    except Exception as e:
        print("  状态: FAILED")
        print("  错误: {}: {}".format(type(e).__name__, e))

    # 3. hyper-lite 模式
    print("\n--- 3) hyper-lite 模式 ---")
    try:
        from hyperrag.base import QueryParam
        param = QueryParam(mode="hyper-lite")
        result = await rag.aquery(test_question, param)
        print("  状态: OK")
        print("  回答长度: {} chars".format(len(result)))
        print("  回答前200字: {}".format(result[:200]))
    except Exception as e:
        print("  状态: FAILED")
        print("  错误: {}: {}".format(type(e).__name__, e))

    print("\n" + "=" * 60)
    print("冒烟测试完成")


if __name__ == "__main__":
    asyncio.run(main())
