# Agent 可控后端全局切片检索设计

## 目标

`POST /agent/retrieval/search_slices` 应当像正常知识库检索一样，从用户指定的多个知识库中查找全局最相关的证据切片，同时返回 `section`、`table` 和 `original-text` handles，供 agent 继续深挖。

每次请求显式选择一种检索后端：Libing RAG（本地 Elasticsearch）或 IPD RAG。同一次请求不会混合两个系统的候选结果。可选 rerank 对两种后端采用一致规则。

本功能不得改变现有 `get_related_docs` 检索路径的行为。

## 接口契约

为 `SearchSlicesRequest` 增加：

```python
retrieval_backend: Literal["libing", "ipd"] = "libing"
enable_rerank: bool = False
```

默认值保证旧调用方兼容：不传新增字段时，继续使用 Libing 检索且不启用 rerank。

为 `SearchSlicesResponse` 增加：

```python
request_id: str
```

调用方可以通过该标识将响应切片与 `RequestResponseLog` 中的检索日志关联起来。

现有 `query`、`kb_sn`/`kb_sn_list`、`top_k`、`include_refs` 和 `include_artifact_handles` 继续保留。最终结果数量继续限制在 `1..50`。

## 总体流程

每次 `search_slices` 请求按以下流程执行：

1. 获取或生成 `request_id`，识别用户，并对所有请求的 KB 执行现有读权限校验。
2. 计算 `final_top_k = clamp(request.top_k, 1, 50)`。
3. 若 `enable_rerank=true`，令 `candidate_top_k = 100`；否则令其等于 `final_top_k`。
4. 仅分发到本次请求选择的 `retrieval_backend`。
5. 在最终选择前，按切片文本精确去重。
6. 若启用 rerank，仅执行一次 rerank，再截取最多 `final_top_k`；否则按后端原始得分降序选择最多 `final_top_k`。
7. 使用现有 agent slice 映射生成可用的 artifact handles。
8. 记录检索日志，并将 `request_id` 与 slices 一并返回。

## Libing RAG 后端

当只有一个已授权 KB 时，继续使用该 KB 的本地 vector-store 配置和 indexes，将 `candidate_top_k` 作为检索候选规模。

当存在多个已授权 KB 时：

1. 使用已有的多 KB 配置辅助函数生成一份统一检索配置。
2. 针对完整 KB 选择集合，按照兼容的 analyzer 和 embedding model 获取 vector-store 分组。
3. 对每个兼容分组调用一次 Elasticsearch 检索 manager，使同一组中的 indexes 在一个搜索空间内竞争。
4. 合并不同兼容组返回的结果，按文本去重；未启用 rerank 时按得分排序。

这会替换 `search_slices` 当前逐 KB 独立检索后再比较得分的行为。改动仅位于 agent 端点内部，不修改共享的 `get_related_docs` 编排逻辑。

## IPD RAG 后端

当 `retrieval_backend="ipd"` 时：

1. 对每个请求的 KB 仍通过相同权限入口校验。
2. 仅保留具备 `ipd_rag_kb_id` 映射的已选 KB。
3. 使用这些映射 ID 与 `candidate_top_k` 调用 IPD RAG。
4. 对于没有 IPD 映射的已授权 KB，跳过并在诊断日志中记录；不自动回退 Libing，也不令整个请求失败。
5. 如果所有已选 KB 均无映射，返回空 slices，并记录该结果。

## Rerank 行为

`enable_rerank` 是 `search_slices` 是否重排的唯一开关。

- 为 `false` 时，两种后端都不 rerank，按后端原始得分产生结果。
- 为 `true` 时，两种后端都先检索最多 100 个候选，调用现有 rerank 服务一次，再返回 `final_top_k` 条最高排名结果。
- 本端点复用现有默认多 KB rerank 模型选择，不额外暴露 `rerank_model` 接口参数。
- 若 rerank 调用失败而现有降级行为返回后端候选，响应仍可使用，同时诊断日志必须记录发生了降级。

## Agent Handles

现有 `_search_slice()` 映射继续负责切片文本、得分、文档标识、位置、actions 及 handles。

- 含结构化 artifact 引用的 Libing 候选继续生成文档、章节和表格 handles。
- IPD 候选只有在返回 metadata 包含相应 artifact 引用时，才能生成相同 handles。
- 缺少 artifact metadata 的 IPD 切片仍作为证据文本返回；不可用的 handles 为 null，对应 actions 为 false。
- 本次不为 IPD 增加额外的 metadata 回填调用。

## 日志

只为 `search_slices` 完整记录检索日志。`get_document_outline`、`get_section`、`get_table` 与 `get_original_text` 的日志行为保持不变。

每次调用写入或更新一条 `RequestResponseLog`，至少包括：

- `request_id`、`user_id`、`method_name="agent.retrieval.search_slices"`、问题和已选 KB；
- 在现有日志辅助能力允许时记录请求及检索的起止时间；
- 在 `retrieve_result` 中以 JSON 保存最终返回的 `SearchSlice` 内容；
- 检索失败时保存 `error_reason`；
- 在 `extra_info` 中保存诊断信息。

诊断字段至少包括：

```json
{
  "retrieval_backend": "libing",
  "enable_rerank": true,
  "requested_top_k": 20,
  "final_top_k": 20,
  "candidate_top_k": 100,
  "returned_slice_count": 20,
  "kb_sn_list": ["kb-a", "kb-b"],
  "candidate_count_before_dedup": 130,
  "candidate_count_after_dedup": 100,
  "rerank_requested": true,
  "rerank_degraded": false,
  "ipd_mapped_kb_sn_list": [],
  "ipd_skipped_unmapped_kb_sn_list": []
}
```

Libing 多 KB 检索还要记录查询的 analyzer 分组数量。本次不要求新增 `RequestDocumentHit` 记录。

## 错误处理

- 缺少用户身份、缺少数据库 session 以及 KB 权限失败，沿用现有失败行为。
- 非法 backend 值由请求校验拒绝。
- IPD 映射缺失按“跳过并记录”处理。
- 后端失败继续暴露给调用方；在日志基础设施可用时同步记录失败原因。

## 测试要求

聚焦测试必须验证：

1. 旧请求默认使用 Libing 且不 rerank，并在响应中返回 `request_id`。
2. Libing 单 KB 沿用原路径，多 KB 使用 grouped global retrieval。
3. Libing 不同分组返回的重复切片在最终选取前去重。
4. 选择 IPD 时不调用 Elasticsearch，仅传递有映射的 ID，跳过并记录未映射 KB；全无映射时返回空结果。
5. 未启用 rerank 时，候选规模使用边界处理后的 `final_top_k`。
6. 启用 rerank 时，两种后端都检索固定 100 个候选，且只 rerank 一次，最终最多返回 50 条。
7. rerank 失败后的降级结果可用并在诊断信息中可观察。
8. 包含结构化 metadata 的候选可生成 handles；缺少 metadata 的 IPD 切片只返回文本。
9. `RequestResponseLog.retrieve_result` 记录最终 slices，`extra_info` 记录后端、rerank、候选规模和 IPD 映射诊断。
10. 生成的 agent skill 契约可以发送新增请求参数。
11. 现有 `get_related_docs` 测试或聚焦的无干扰回归测试继续通过。

## 预计修改文件

- `rag_service/agent_retrieval/models.py`：请求 backend/rerank 字段及响应 `request_id`。
- `rag_service/agent_retrieval/router.py`：request id 与日志上下文接入。
- `rag_service/agent_retrieval/service.py`：后端分发、Libing 全局检索、IPD 分发、候选规模、去重、rerank 与诊断信息。
- `rag_service/agent_retrieval/skill_generator.py`：生成的请求契约及脚本参数。
- 已有日志辅助模块；若本 checkout 缺少被引用的应用日志辅助模块，则增加严格限定于 agent 的日志辅助逻辑。
- `tests/test_agent_retrieval.py` 及必要的聚焦日志/契约回归测试。

正常 `get_related_docs` 端点中的代码不属于本次改动范围。
