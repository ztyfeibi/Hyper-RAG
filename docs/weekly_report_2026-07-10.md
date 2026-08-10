# Hyper-RAG 项目周工作汇报

## 一、上周情况回顾

上周项目主要处于“原始代码接入与方法规划”阶段。基础代码来自 Hyper-RAG 论文实验代码，项目目标是在原始 Hyper-RAG 基础上扩展为 **Query-Adaptive Domain-Aware Hyper-RAG**，即让系统先理解问题类型，再结合领域实体类型和超图结构，动态调整检索范围、扩散深度和超边权重。

上周已完成的基础工作包括：

1. 完成项目结构梳理与模块拆分
   原始 `operate.py` 已逐步拆分为更清晰的模块，包括：
   - `indexing.py`：建库与实体/关系抽取
   - `query_modes.py`：不同查询模式
   - `query_context.py`：检索上下文构建
   - `query_keywords.py`：查询关键词解析
   - `graph_upsert.py`：实体与超边合并写入
   - `extraction.py`：LLM 抽取结果解析

2. 完成医学领域 schema 初步设计
   已整理 neurology 领域实体类型、关系类型和高阶关系类型，为后续“领域类型增强索引”和“类型感知检索”打基础。

3. 明确实验总路线
   项目实验路线被规划为：
   - Phase 0：基础设施修复
   - Phase 1：Baseline 复现
   - Phase 2：创新模块实现
   - Phase 3：消融实验
   - Phase 4：结果分析与论文写作

当时的主要问题是：原始代码还不能稳定完成 neurology 数据集建库、查询和评测，实验流水线没有完全跑通。

## 二、本周主要工作与进展

本周工作重点从“方法设计”推进到“实验流水线打通与 baseline 复现”。整体上，本周完成了 Phase 0 的大量工程修复，并推进 Phase 1 baseline 实验。

### 1. 修复建库阶段关键问题

本周集中修复了 Step_1 建库阶段的一系列致命问题，使系统从“无法稳定建库”推进到“能够完成完整知识库构建”。

主要修复包括：

- 修复 embedding 维度不匹配
  原配置为 2048，但当前 qwen-8b-embed 实际输出为 4096，已统一调整为 4096。

- 修复 vLLM Qwen thinking 参数问题
  将错误的：
  ```python
  extra_body={"enable_thinking": False}
  ```
  修正为：
  ```python
  extra_body={"chat_template_kwargs": {"enable_thinking": False}}
  ```
  避免 `<think>` 内容污染 JSON 输出。

- 修复异步并发控制问题
  原 `limit_async_func_call` 使用手动计数器，存在并发竞态，导致实际并发超过配置，进而压垮 LLM 服务。已改为 `asyncio.Semaphore`。

- 修复 gleaning 异常传播问题
  原 gleaning 阶段任一 chunk 出错会导致 `asyncio.gather` 整体崩溃，已增加异常保护。

- 明确关闭大规模 gleaning 的必要性
  在 chunk=2400 的场景下，gleaning 会导致上下文过长和大量额外 LLM 调用，最终决定 baseline 建库先使用 `gleaning=0`。

### 2. 完成 chunk=2400 工程 baseline

本周成功完成 `caches/neurology` 的 chunk=2400 建库：

- 864 chunks
- 20198 vertices
- 21109 hyperedges
- 建库耗时约 25 分钟
- 已完成完整备份：`backups/chunk2400_baseline_20260706/`

该版本作为工程基线可用，证明代码链路已经跑通。

### 3. 跑通 Phase 1 初版 baseline 评测

基于 chunk=2400 知识库，完成了 naive 和 hyper 的回答生成与五维打分评测。

当前结果如下：

| 指标 | Naive | Hyper |
|---|---:|---:|
| Comprehensiveness | 88.2 | 75.4 |
| Diversity | 78.8 | 83.3 |
| Empowerment | 76.2 | 64.2 |
| Logical | 92.5 | 90.4 |
| Readability | 92.6 | 93.3 |
| Overall | 85.6 | 81.3 |

这个结果说明当前 chunk=2400 baseline 下 naive 反而高于 hyper。但进一步分析发现，这不是方法本身一定失败，而是实验设置不公平：

- naive 是单线 chunk 检索，可以使用较大的 text budget；
- hyper 是 entity 线和 relation 线双线检索，合并后容易超过 24K context；
- chunk=2400 太大，hyper 每条线实际只能容纳很少 chunk；
- 因此 hyper 的 source evidence 被严重压缩，影响 Comprehensiveness 和 Empowerment。

### 4. 修复查询阶段 token budget 问题

本周还修复了 Step_3 查询阶段的问题：

- 修复 `truncate_list_by_token_size` 第一个 chunk 超预算时返回空列表的问题；
- 为 naive / hyper 设置不同默认文本预算；
- 增加 `--max-token-for-text-unit` 参数，方便后续显式控制实验条件。

这使 naive 和 hyper 都能够稳定生成回答，不再出现全部查询失败或空上下文的问题。

### 5. 推进 chunk=1000 公平 baseline 重建

为了解决 chunk=2400 对 hyper 不公平的问题，本周设计并启动了 `neurology_chunk1000` 知识库重建。

目标配置：

```text
chunk_token_size = 1000
chunk_overlap_token_size = 150
gleaning = 0
batch_size = 3000 contexts
```

本周完成了以下工作：

- 新增 Step_1 参数：
  - `--source-data-name`
  - `--chunk-token-size`
  - `--chunk-overlap-token-size`
  - `--gleaning`
  - `--batch-size`

- 增加分批建库能力
  避免一次性处理 2317 个 chunk 导致内存或任务过重。当前按 3000 contexts 一批，大约 5 批。

- 修复 batch 失败后伪成功问题
  如果某个 batch 重试 3 次仍失败，现在会直接报错退出，避免生成残缺知识库却显示成功。

- 编写 `run_rebuild.bat`
  支持：
  - `run_rebuild.bat fresh`：清空并重新建库
  - `run_rebuild.bat`：同参数下断点续跑

截至目前（7 月 10 日 14:00），`neurology_chunk1000` 重建已运行约 2 天并持续正常推进：

- **Batch 1/5** ✅ 已完成落盘
- **Batch 2/5** ✅ 已完成落盘
- **Batch 3/5** ⏳ 进行中（约 23%，实体抽取阶段，无异常）
- 当前缓存大小：~948MB（vdb_entities 395M、vdb_relationships 495M、hgdb 13M、vdb_chunks 23M）
- stderr 中无任何错误或超时日志，所有 LLM/Embedding 请求均返回 200 OK

这说明新的分批建库机制有效，已不再是"跑完才落盘"的高风险模式。当前 Batch 3 预计还需 1-2 天完成。

## 三、本周相比上周的变化

相比上周，本周项目从“方法方案和代码结构整理”推进到了“可运行、可评测、可定位问题”的实验状态。

主要变化包括：

1. 从不能稳定建库，到完成 chunk=2400 全量建库。
2. 从没有 baseline 结果，到得到 naive / hyper 的初版五维评测结果。
3. 从单一大 chunk 实验，推进到 chunk=1000 的公平 baseline 重建。
4. 从一次性全量建库，升级为分批建库、分批落盘、可断点续跑。
5. 从问题定位依赖手工排查，转为有 bugfix log、handoff 文档、重建计划和脚本支持。

本周的核心成果不是最终分数提升，而是把实验基础设施打稳，并发现了原始 Hyper-RAG 在当前数据和模型约束下的关键实验偏差：**chunk 粒度和双线检索预算会显著影响 hyper 的表现**。

## 四、当前问题与风险

当前仍存在几个需要关注的问题：

1. chunk=1000 正式 baseline 尚未完全完成
   当前已成功落盘 2/5 批，第 3/5 批正在推进（实体会抽取阶段，约 23%）。无错误或超时日志，稳定性良好。
   仍需等待全 5 批完成后检查最终 `vdb_*`、`kv_store_*` 和 `.hgdb` 是否完整。

2. chunk=2400 的 naive > hyper 结果不能作为最终方法结论
   该结果更多反映实验设置不公平，不能直接用于论文主结论。

3. 分批建库改变了 full_doc 粒度
   每批会作为一个 doc 插入，主要影响 `full_docs` 粒度，对 chunk/entity/relation 检索影响较小，但后续实验记录中需要说明。

4. 评测问题数量仍偏小
   当前主要基于 2-stage 问题，后续应扩展 1-stage 和 3-stage，以验证“复杂问题上 adaptive 更有效”的论文假设。

## 五、下一步计划

下一步分为三条线推进。

### 1. 完成 chunk=1000 公平 baseline

优先完成 `neurology_chunk1000` 的剩余 batch 建库，完成后检查：

- `kv_store_text_chunks.json`
- `vdb_chunks.json`
- `vdb_entities.json`
- `vdb_relationships.json`
- `hypergraph_chunk_entity_relation.hgdb`
- 超图 vertices / hyperedges 数量
- `OTHER` 类型比例
- 是否存在 traceback 或 context overflow

随后复制 2-stage 问题到新 cache，重新跑：

```text
Step_3 naive
Step_3 hyper
evaluate_by_scoring naive
evaluate_by_scoring hyper
evaluate_by_selection naive vs hyper
```

目标是得到一个更公平的 baseline，用于后续 adaptive 方法对比。

### 2. 开始 Phase 2 创新模块开发

在 chunk=1000 baseline 稳定后，进入创新实现：

- 新增 `adaptive` mode
- 实现 Query Router
- 根据 query complexity 动态调整 top-k、扩散深度和上下文预算
- 实现类型感知检索 soft weighting
- 实现质量感知超边排序
- 实现结构化 evidence context

第一阶段建议先实现轻量版：

```text
Query Router + 类型感知检索 + adaptive top-k
```

先跑 E3/E4 消融，不急着一次性实现完整 E7。

### 3. 完善实验记录与论文材料

后续需要补充：

- chunk=1000 实验 metadata
- naive/hyper/adaptive 每题检索时间
- context token 数
- entity 命中数
- high-order edge 比例
- source chunk 命中数
- simple / complex 问题分层分析

这些中间指标会支撑论文中的“为什么有效”，不只看最终回答分数。

## 六、本周总结

本周主要完成了 Hyper-RAG 实验代码从"不可稳定复现"到"可建库、可查询、可评测"的工程打通工作。我们修复了 10 个关键问题（含 7 个致命级），完成 chunk=2400 baseline 建库和评测并备份，发现该 baseline 对 hyper 不公平后立即启动 chunk=1000 公平 baseline 重建。建库流程已升级为分批落盘、可断点续跑的形式，彻底解决了全量建库内存溢出导致数小时白跑的问题。

截至报告时（7 月 10 日），chunk=1000 重建已完成 2/5 批落盘、第 3/5 批正常推进中，无任何异常。后续需等待全量重建完成后跑通 naive/hyper 公平评测，即可进入创新模块开发阶段。
