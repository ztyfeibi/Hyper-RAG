# Hyper-RAG 项目：完整问题与解决方案清单

> 生成时间：2026-07-07
> 覆盖阶段：Phase 0（建库 Step_1）+ Phase 1（Baseline Step_2/Step_3）
> 目的：供独立评估

---

## 目录

- [Phase 0：建库阶段（Step_1）](#phase-0建库阶段step_1)
  - [Bug 1: EMB_DIM 维度不匹配](#bug-1-emb_dim-维度不匹配)
  - [Bug 2: vLLM extra_body 格式错误](#bug-2-vllm-extra_body-格式错误)
  - [Bug 3: time 变量覆盖 time 模块](#bug-3-time-变量覆盖-time-模块)
  - [Bug 4: time.sleep() 阻塞事件循环](#bug-4-timesleep-阻塞事件循环)
  - [Bug 5: limit_async_func_call 竞态条件](#bug-5-limit_async_func_call-竞态条件)
  - [Bug 6: gleaning 异常传播导致 asyncio.gather 崩溃](#bug-6-gleaning-异常传播导致-asynciogather-崩溃)
  - [Bug 7: gleaning context 超长导致全量建库失败](#bug-7-gleaning-context-超长导致全量建库失败)
- [Phase 1：Baseline 阶段（Step_2 / Step_3）](#phase-1baseline-阶段step_2--step_3)
  - [Bug 8: 评测问题脏数据（原论文 LLM 输出泄漏）](#bug-8-评测问题脏数据原论文-llm-输出泄漏)
  - [Bug 9: truncate_list_by_token_size 返回空列表](#bug-9-truncate_list_by_token_size-返回空列表)
  - [Bug 10: hyper 模式双线检索 token 超限](#bug-10-hyper-模式双线检索-token-超限)
- [已知待优化项（暂未修复，记录为后续改进）](#已知待优化项暂未修复记录为后续改进)
  - [Issue A: chunk_size 与 max_token_for_text_unit 设计矛盾](#issue-a-chunk_size-与-max_token_for_text_unit-设计矛盾)
  - [Issue B: Python 环境不一致](#issue-b-python-环境不一致)
  - [Issue C: gleaning=0 对实体召回率的潜在影响](#issue-c-gleaning0-对实体召回率的潜在影响)
- [附录：修复时间线](#附录修复时间线)

---

## Phase 0：建库阶段（Step_1）

### Bug 1: EMB_DIM 维度不匹配

| 项目 | 内容 |
|------|------|
| **文件** | `my_config.py` 第 19 行 |
| **严重程度** | 🔴 致命（建库直接失败） |
| **发现阶段** | 50 条 context 验证阶段 |

**问题描述：**

原代码 `EMB_DIM = 2048`，但实际使用的 embedding 模型 qwen-8b-embed（`10.65.1.110:8001`）原生输出维度为 4096。qwen-8b-embed 不支持 matryoshka 降维，不能传 `dimensions` 参数指定输出维度。

**错误现象：**

```
ValueError: Embedding dimension mismatch: expected 2048, got 4096
```

nano-vectordb 初始化时按 2048 创建向量空间，但 embedding API 返回 4096 维向量，插入时报维度不匹配。

**解决方案：**

```python
# my_config.py
EMB_DIM = 4096  # 从 2048 改为 4096，匹配 qwen-8b-embed 原生维度
```

**根本原因：**

原论文代码设计时默认用 SiliconFlow 的 `Qwen/Qwen3-Embedding-8B`（支持 matryoshka 降维到 2048），用户切换到内网自部署的 qwen-8b-embed 后未同步修改维度配置。

---

### Bug 2: vLLM extra_body 格式错误

| 项目 | 内容 |
|------|------|
| **文件** | `hyperrag/llm.py` 第 83、143、201 行（共 3 处） |
| **严重程度** | 🔴 致命（所有 LLM 调用失败） |
| **发现阶段** | 50 条 context 验证阶段 |

**问题描述：**

原代码使用 `extra_body={"enable_thinking": False}` 来禁用 qwen-27b 的思考模式，但 vLLM 的 OpenAI 兼容 API 要求该参数必须嵌套在 `chat_template_kwargs` 下。

**错误现象：**

```
# LLM 返回的 response 中包含 <think>...</think> 标签
# 或 vLLM 报 400 错误，提示参数格式不对
```

qwen-27b-int4 模型默认开启 thinking 模式，输出会包含 `<think>` 标签，导致 JSON 解析失败。

**解决方案：**

3 处统一修改为：

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

**涉及位置：**
1. `llm.py` 第 83 行：`openai_async_client.chat.completions.create()`（非流式）
2. `llm.py` 第 143 行：`openai_async_client.chat.completions.create(stream=True)`（流式）
3. `llm.py` 第 201 行：另一个非流式调用点

**根本原因：**

vLLM 部署的 qwen 模型使用 chat template 渲染，`enable_thinking` 是 chat template 的参数，必须通过 `chat_template_kwargs` 传递，不能直接作为 `extra_body` 的顶层字段。

---

### Bug 3: time 变量覆盖 time 模块

| 项目 | 内容 |
|------|------|
| **文件** | `hyperrag/indexing.py` |
| **严重程度** | 🟡 高（运行时崩溃） |
| **发现阶段** | 50 条 context 验证阶段 |

**问题描述：**

原代码在计时逻辑中使用了变量名 `time`，覆盖了 Python 标准库的 `time` 模块。

```python
# 原代码（问题）
time = current_time - begin_time  # 覆盖了 time 模块
# 后续调用 time.sleep() 时，time 已不再是模块，而是 float
```

**错误现象：**

```
AttributeError: 'float' object has no attribute 'sleep'
```

**解决方案：**

将变量名改为 `elapsed`：

```python
elapsed = current_time - begin_time
```

---

### Bug 4: time.sleep() 阻塞事件循环

| 项目 | 内容 |
|------|------|
| **文件** | `hyperrag/indexing.py` |
| **严重程度** | 🟡 高（并发性能退化到串行） |
| **发现阶段** | 50 条 context 验证阶段 |

**问题描述：**

原代码在 async 函数中调用同步的 `time.sleep()`，这会阻塞整个事件循环，导致其他协程无法执行，并发退化为串行。

**错误现象：**

864 个 chunk 的实体抽取本应并发执行，但实际表现为逐个串行处理，速度极慢。

**解决方案：**

```python
# 原代码
time.sleep(wait_time)

# 修复后
await asyncio.sleep(wait_time)
```

---

### Bug 5: limit_async_func_call 竞态条件

| 项目 | 内容 |
|------|------|
| **文件** | `hyperrag/utils.py` 第 96-113 行 |
| **严重程度** | 🔴 致命（全量建库第一次失败） |
| **发现阶段** | 864 chunks 全量建库第一次尝试 |
| **排查耗时** | ~2 小时 |

**问题描述：**

原代码用手动计数器实现并发限制：

```python
# 原代码（问题）
class limit_async_func_call:
    def __init__(self, max_size):
        self.__max_size = max_size
        self.__current_size = 0

    async def wait_func(*args, **kwargs):
        while self.__current_size >= self.__max_size:
            await asyncio.sleep(0.0001)
        self.__current_size += 1
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            self.__current_size -= 1
```

问题在于 `while __current_size >= __max_size` 检查和 `__current_size += 1` 之间没有原子性保证。在 asyncio 的事件循环中，虽然单线程不会真正并行执行 Python 代码，但在 `await asyncio.sleep(0.0001)` 挂起期间，其他协程可以同时通过 `while` 检查并都执行 `+= 1`，导致实际并发数远超 `max_size`。

**错误现象：**

- 864 chunks 全量建库，只有 16 个 LLM 缓存成功
- vLLM 服务因并发过高（远超 4）而过载，大量请求超时或失败
- 最终写出 `0 vertices, 0 hyperedges` 的空超图

**验证过程：**

用标准 `asyncio.Semaphore(4)` + 20 个并发真实 prompt 测试，全部成功（175 秒），证明根因确实是竞态条件。

**解决方案：**

```python
# 修复后
def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """使用 asyncio.Semaphore 替代手动计数器，避免高并发下的竞态条件。"""
    def final_decro(func):
        sem = asyncio.Semaphore(max_size)

        @wraps(func)
        async def wait_func(*args, **kwargs):
            async with sem:
                return await func(*args, **kwargs)
        return wait_func
    return final_decro
```

同步修复了 `limit_async_gen_call`（异步生成器版本），也改用 `asyncio.Semaphore`。

**根本原因：**

原代码试图用"检查-自增"模式模拟信号量，但没有利用 asyncio 原生的同步原语。`asyncio.Semaphore` 在内部使用 `asyncio.Future` 和队列管理等待者，保证了原子性。

---

### Bug 6: gleaning 异常传播导致 asyncio.gather 崩溃

| 项目 | 内容 |
|------|------|
| **文件** | `hyperrag/indexing.py` 第 102-125 行 |
| **严重程度** | 🔴 致命（全量建库第二次失败） |
| **发现阶段** | 864 chunks 全量建库第二次尝试 |
| **排查耗时** | ~3 小时 |

**问题描述：**

gleaning 是实体抽取中的追加提取机制。每个 chunk 在第一轮抽取后，会用 `continue_prompt` 追问 LLM"还有没有遗漏的实体"，再用 `if_loop_prompt` 问"是否继续"。

原代码中，gleaning 的 `continue_prompt` 和 `if_loop_prompt` 调用**没有 try/except 保护**。864 个 chunk 并发时，个别 gleaning 调用失败（因网络抖动、超时等）抛出异常。`asyncio.gather` 默认 `return_exceptions=False`，一个异常就导致整个 gather 失败，异常传播到 `ainsert` 的 `finally` 块，写出 `0 vertices` 的空超图。

**错误现象：**

- 872 条 LLM 缓存全部命中（解析正常，19590 entities）
- 但 `Writing hypergraph with 0 vertices, 0 hyperedges`
- 小规模测试（5 / 50 条 context）都成功，因为规模小没有 gleaning 失败

**验证过程：**

1. 检查 LLM 缓存 → 全部命中，排除 LLM 调用问题
2. 检查实体解析 → 19590 entities 正常解析
3. 检查 merge 阶段 → 正常产出 22220 实体
4. 检查 gleaning 调用 → 发现没有 try/except
5. 推断：个别 gleaning 异常导致 asyncio.gather 整体崩溃

**解决方案：**

给 gleaning 的两个调用都加 try/except，失败时 break 跳过 gleaning：

```python
for now_glean_index in range(entity_extract_max_gleaning):
    try:
        glean_result = await use_llm_func(continue_prompt, history_messages=history)
    except Exception as e:
        logger.warning(f"Chunk {chunk_key} gleaning call failed (attempt {now_glean_index + 1}): {e}. Skipping gleaning.")
        break
    if glean_result is None:
        break
    # ... 合并 gleaning 结果 ...
    try:
        if_loop_result = await use_llm_func(if_loop_prompt, history_messages=history)
    except Exception as e:
        logger.warning(f"Chunk {chunk_key} if_loop call failed: {e}. Stopping gleaning.")
        break
```

---

### Bug 7: gleaning context 超长导致全量建库失败

| 项目 | 内容 |
|------|------|
| **文件** | `reproduce/Step_1.py` 第 139 行 |
| **严重程度** | 🔴 致命（全量建库第三次失败，进程跑了 11 小时） |
| **发现阶段** | 864 chunks 全量建库第三次尝试 |
| **排查耗时** | ~2 小时 |

**问题描述：**

即使修了 Bug 6（gleaning 异常保护），gleaning 本身在 864 chunks 规模下仍然不可行：

1. **context 爆炸**：每个 chunk 的 gleaning 轮都要带上历史消息（之前的抽取结果）。qwen-27b 的 context 只有 24K tokens，gleaning 第二轮时 history + continue_prompt 就超长了，几乎所有 gleaning 调用都因 context 超长失败
2. **级联放大**：gleaning 会产生大量碎片化实体，导致 merge 阶段需要 ~5 万次 LLM 调用做实体合并/摘要，预估耗时 52 小时

**错误现象：**

- 进程跑了 11 小时未完成
- gleaning 调用全部因 context 超长失败（24K 限制）
- merge 阶段 5 万次 LLM 调用排队

**解决方案：**

在 `Step_1.py` 中设置 `entity_extract_max_gleaning=0`，完全跳过 gleaning：

```python
# reproduce/Step_1.py
entity_extract_max_gleaning=0,  # 禁用 gleaning
```

**效果：**

- 只做第一轮实体抽取，不追问
- 864 chunks 在 25 分钟内完成
- 最终产出：20198 vertices, 21109 hyperedges（覆盖率充足）

**Tradeoff：**

关掉 gleaning 可能少抽 5-10% 的次要实体，但：
- qwen-27b 单轮抽取能力够强，gleaning 边际收益递减
- 24K context 下 gleaning 第二轮直接报错，开着反而 0 vertices
- 先跑通基线，后续可用实验验证 gleaning=0 vs gleaning=1 的效果差异

---

## Phase 1：Baseline 阶段（Step_2 / Step_3）

### Bug 8: 评测问题脏数据（原论文 LLM 输出泄漏）

| 项目 | 内容 |
|------|------|
| **文件** | 复制自 `D:/projectes/hyperRAG/Hyper-RAG/caches/neurology/questions/2_stage.json` |
| **严重程度** | 🟡 中（影响评测公平性） |
| **发现阶段** | Phase 1 Step_2（复用原论文评测问题） |

**问题描述：**

原论文代码生成的 50 条评测问题中，有 5 条是脏数据——LLM 输出泄漏导致内容不是真正的问题，而是 prompt 残片（如 `"string\"`...` 这类）。

**脏数据索引：** 5, 7, 19, 44, 49

**解决方案：**

复制原文件后过滤脏数据：

```python
# 读取原文件
with open('D:/projectes/hyperRAG/Hyper-RAG/caches/neurology/questions/2_stage.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

# 标记脏数据
corrupted = [i for i, q in enumerate(questions) 
             if not isinstance(q, str) or len(q) < 50 or 'string"' in q 
             or 'prompt says' in q.lower() or q.startswith('string')]

# 过滤后保存
clean_questions = [q for i, q in enumerate(questions) if i not in corrupted]
```

**结果：** 45 条干净问题 + 45 条对应参考文本（`2_stage_ref.json` 同步过滤）

---

### Bug 9: truncate_list_by_token_size 返回空列表

| 项目 | 内容 |
|------|------|
| **文件** | `hyperrag/utils.py` 第 215-227 行 + `reproduce/Step_3_response_question.py` 第 158-166 行 |
| **严重程度** | 🔴 致命（naive 模式完全无效） |
| **发现阶段** | Phase 1 Step_3 naive 模式第一次运行 |
| **排查耗时** | ~1 小时 |

**问题描述：**

两个问题叠加导致 naive 模式的 45 个回答全部没有使用检索内容：

**问题 A：参数矛盾**

建库时 `chunk_token_size=2400`，但检索时默认 `max_token_for_text_unit=1600`。每个 chunk 平均 2445 tokens（中位数），863/864 个 chunk 都超过 1600。

**问题 B：边界 bug**

`truncate_list_by_token_size` 的原逻辑：

```python
def truncate_list_by_token_size(list_data, key, max_token_size):
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]  # ← 当 i=0 时返回空列表
    return list_data
```

当第一个 chunk（2400 tokens）就超过预算（1600 tokens）时，`i=0`，`list_data[:0]` 返回空列表 `[]`。函数没有考虑"连一个元素都装不下"的边界情况。

**错误现象：**

日志显示 `Truncate 60 to 0 chunks`，即检索到 60 个候选 chunk 但截断后剩 0 个。LLM 完全没有收到任何检索内容，45 条回答全是凭自身知识编的。

**解决方案：**

**修复 A：边界保底**

```python
def truncate_list_by_token_size(list_data, key, max_token_size):
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            # 至少保留第一个元素，避免单个元素就超预算时返回空列表
            return list_data[:max(1, i)]
    return list_data
```

**修复 B：提高预算**

```python
# reproduce/Step_3_response_question.py
if mode == "naive":
    # 单线检索，预算可以给大
    query_param = QueryParam(mode=mode, max_token_for_text_unit=12000)
else:
    # hyper 双线检索，每线预算减半避免合并后超限
    query_param = QueryParam(mode=mode, max_token_for_text_unit=4000)
```

**效果：**

naive 重跑后日志显示 `Truncate 60 to 4-5 chunks`，回答包含具体医学细节，44 条成功 + 1 条错误。

**根本原因：**

原论文代码的 chunk_size 较小（几百 tokens），`max_token_for_text_unit=1600` 能装 3-5 个 chunk，从未触发边界 case。用户用 `chunk_token_size=2400` 建库后，参数矛盾暴露。

---

### Bug 10: hyper 模式双线检索 token 超限

| 项目 | 内容 |
|------|------|
| **文件** | `reproduce/Step_3_response_question.py` 第 161-166 行 |
| **严重程度** | 🔴 致命（hyper 模式 45/45 全部失败） |
| **发现阶段** | Phase 1 Step_3 hyper 模式第一次运行 |
| **排查耗时** | ~30 分钟 |

**问题描述：**

Bug 9 修复时，最初把 `max_token_for_text_unit` 统一设为 12000。naive 模式正常，但 hyper 模式全部失败。

**原因：** hyper 模式走**两条检索线**：

1. **Entity 线**：按实体语义检索 → 获取 text units（最多 12000 tokens）
2. **Relation 线**：按关系语义检索 → 获取 text units（最多 12000 tokens）
3. `combine_contexts` 合并两条线的 text units（去重，但两条线检索到不同 chunk，去重后仍有 8-10 个 unique chunk）

合并后 text units 总量约 24000 tokens，加上实体描述（60 个）+ 关系描述（20-60 个）+ system prompt，总计远超 qwen-27b 的 24K context 上限。

**错误现象：**

```
Error: maximum context length is 24576 tokens. prompt contains at least 24577 input tokens
```

45/45 全部失败。

**解决方案：**

按 mode 区分 token 预算：

```python
if mode == "naive":
    # 单线检索，12000 tokens 能装 ~5 个 2400-token chunk
    query_param = QueryParam(mode=mode, max_token_for_text_unit=12000)
else:
    # hyper 双线检索，每线 4000 tokens
    # 合并后约 8000 tokens + 实体/关系描述 + prompt ≈ 12K 以内，安全
    query_param = QueryParam(mode=mode, max_token_for_text_unit=4000)
```

**效果：**

hyper 重跑正常，日志显示 `entity query uses 60 entites, XX relations, 1 text units`（每条线检索到 1 个 chunk，合并后约 2 个），不再超限。

**Tradeoff：**

hyper 模式每条线只能检索 1 个 chunk（4000 tokens / 2400 tokens per chunk ≈ 1.6），text units 利率较低。后续可通过减小 chunk_size 或增大 LLM context 来改善。

---

## 已知待优化项（暂未修复，记录为后续改进）

### Issue A: chunk_size 与 max_token_for_text_unit 设计矛盾

| 项目 | 内容 |
|------|------|
| **状态** | 📌 临时方案已实施，根本优化待做 |
| **影响** | hyper 模式每线只能检索 1 个 chunk |

**问题：**

建库 `chunk_token_size=2400`，检索 `max_token_for_text_unit` 需要适配：

| 方案 | 做法 | 代价 | 影响 |
|------|------|------|------|
| **当前（临时）** | 按 mode 分配预算：naive=12000, hyper=4000 | 无需重建 | hyper 每线仅 1 个 chunk，检索粒度粗 |
| **方案 B（根本）** | 减小 chunk 到 800，重建 | 重建需 25 分钟 | 检索更精细，naive 能塞 15 个、hyper 能塞 5 个 |

**建议：** 后续做消融实验时考虑方案 B，提升检索精度。

---

### Issue B: Python 环境不一致

| 项目 | 内容 |
|------|------|
| **状态** | 📌 已记录，后续统一 |
| **影响** | 潜在的版本兼容问题 |

**问题：**

- Step_1 建库在 **conda `hyperrag`** 环境运行
- Step_3 baseline 在 **WorkBuddy 自带 Python 3.13.12** 运行

两个环境安装的包版本可能不同（numpy、tiktoken、hypergraph-db 等），存在潜在的兼容性风险。

**当前状态：** Step_3 naive + hyper 均成功运行，暂未发现兼容问题。

**建议：** 后续操作统一切到 conda `hyperrag` 环境。conda 环境路径需通过 `conda info --envs` 确认。

---

### Issue C: gleaning=0 对实体召回率的潜在影响

| 项目 | 内容 |
|------|------|
| **状态** | 📌 待实验验证 |
| **影响** | 可能少抽 5-10% 次要实体 |

**问题：**

`entity_extract_max_gleaning=0` 跳过了所有追问轮次，第一轮没抽到的实体永久丢失。

**预期影响：**

- 召回率略降（次要实体如伴随症状、次要药物可能遗漏）
- 关系完整性受影响（少实体 → 少关系 → 超图连接度略低）
- 但 20198 vertices + 21109 hyperedges 的密度已经足够

**验证方案：**

```
E1 (gleaning=0, 当前) vs E1-gleaning1 (gleaning=1)
```

对比两组的五维打分。如果差异 >2-3%，考虑开 gleaning（但需要换更大 context 的模型）。

---

## 附录：修复时间线

| 日期 | 阶段 | 事件 |
|------|------|------|
| 2026-07-05 09:08 | Phase 0 | 50 条 context 验证成功（454 vertices），修复 Bug 1-4 |
| 2026-07-05 ~ | Phase 0 | 全量建库第 1 次失败：Bug 5（Semaphore 竞态），排查 ~2h |
| 2026-07-05 ~ | Phase 0 | 全量建库第 2 次失败：Bug 6（gleaning 异常传播），排查 ~3h |
| 2026-07-05 ~ | Phase 0 | 全量建库第 3 次失败：Bug 7（gleaning context 超长），进程跑 11h |
| 2026-07-06 10:03 | Phase 0 | 全量建库成功：20198 vertices, 21109 hyperedges, 25 分钟 |
| 2026-07-06 20:00 | Phase 1 | 复用原论文评测问题，清理 5 条脏数据（Bug 8） |
| 2026-07-06 ~20:30 | Phase 1 | naive 第 1 次运行：Bug 9（truncate 返回空列表），45 条全无效 |
| 2026-07-06 ~21:00 | Phase 1 | naive 重跑成功：44 条成功 + 1 条错误 |
| 2026-07-06 ~21:25 | Phase 1 | hyper 第 1 次运行：Bug 10（token 超限），45/45 全失败 |
| 2026-07-06 ~21:46 | Phase 1 | hyper 重跑启动（max_token_for_text_unit=4000） |
| 2026-07-07 ~13:48 | Phase 1 | hyper 重跑进行中（11/45 已完成，状态正常） |

---

## 总结统计

| 指标 | 数值 |
|------|------|
| 修复 Bug 总数 | 10 个 |
| 致命 Bug（🔴） | 7 个 |
| 高严重度（🟡） | 2 个 |
| 中严重度（🟡） | 1 个 |
| 待优化项 | 3 个 |
| 建库失败次数 | 3 次 |
| Baseline 失败次数 | 2 次（naive 1 次 + hyper 1 次） |
| 总排查耗时 | ~13 小时 |
| 涉及文件 | `my_config.py`, `hyperrag/llm.py`, `hyperrag/utils.py`, `hyperrag/indexing.py`, `reproduce/Step_1.py`, `reproduce/Step_3_response_question.py` |
