# 文档解析、切片与表格块统一设计

## 背景

当前向量化流程中，`load_original_documents` 调用 `load_file` 后直接得到最终切片。`load_file` 内部同时负责选择 loader、执行文档解析、调用 splitter、改写 metadata 和处理异常。这个边界会带来三个问题：

1. Dagster UI 只能看到“加载文档”节点，看不到“解析文档”和“切片文档”两个阶段。
2. 解析后的文档内容无法在切片前单独存储，后续难以做文档存储、解析结果审计或复用。
3. 表格解析逻辑分散在 Excel、HTML、Docx loader 中，复杂表格能力难以复用和扩展。

本设计目标是将文档解析和文档切片拆成两个明确阶段，并把跨格式表格统一为可扩展的 `TableBlock` 协议。

## 目标

- 在 Dagster 流程中显式展示 `parse_original_documents`、`save_parsed_documents`、`split_parsed_documents`。
- 让解析阶段输出稳定的中间态，后续可存储、审计、复用。
- 让 Excel、HTML、Docx、Markdown 等文档中的表格统一进入表格块协议。
- 让向量库文本和前端展示内容解耦：`page_content` 面向检索和 LLM，`metadata.display` 面向前端渲染。
- 保持现有 `Document` 作为最终向量切片格式，降低对 embedding 和 vectorstore 的影响。

## 非目标

- 不在第一阶段引入完整文档 AST。
- 不要求所有 loader 一次性迁移到新协议。
- 不把 HTML 表格内容直接作为向量化主文本。
- 不改检测流程的业务语义，检测节点继续消费最终切片。

## 推荐架构

新增一个轻量中间层：`ParsedBlock`。所有 loader 的解析结果先进入 block，再由统一 splitter 转成 LangChain `Document`。

```python
ParsedBlock(
    block_type="text" | "table" | "image" | "code",
    text="给检索和 LLM 使用的语义文本",
    metadata={
        "source": "/data/example.xlsx",
        "headers": ["销售统计"],
        "display": {
            "type": "table",
            "format": "html",
            "content": "<table><tr><th>区域</th></tr><tr><td>华东</td></tr></table>",
        },
        "table": {
            "title": "区域销售统计",
            "source_type": "excel|html|docx|markdown",
            "sheet_name": "销售",
            "cell_range": "A5:E20",
            "header_rows": [["区域", "产品", "销量"], ["", "", "Q1"]],
            "flatten_headers": ["区域", "产品", "销量-Q1"],
        },
        "split_policy": "text" | "table_rows" | "no_split",
    },
)
```

`ParsedBlock.text` 用于切片、embedding 和 LLM 加工；`metadata.display` 是可选展示协议，只有需要特殊渲染的块提供。

## Dagster 流程

当前流程：

```python
documents = save_document_metadata(fetch_raw_document(change_vectorization_job_status_to_started())).map(
    load_original_documents
)
documents = documents.map(sensitive_words_detect)
embeddings = documents.map(embedding_documents)
```

目标流程：

```python
document_batches = save_document_metadata(fetch_raw_document(change_vectorization_job_status_to_started()))
parsed_documents = document_batches.map(parse_original_documents)
parsed_documents = parsed_documents.map(save_parsed_documents)
document_chunks = parsed_documents.map(split_parsed_documents)

document_chunks = document_chunks.map(sensitive_words_detect)
embeddings = document_chunks.map(embedding_documents)
```

节点职责：

- `parse_original_documents`：文件格式解析，输出 `List[Tuple[OriginalDocument, List[ParsedBlock]]]`。
- `save_parsed_documents`：保存解析中间态，第一阶段可以先 no-op 或写入本地/对象存储。
- `split_parsed_documents`：把 `ParsedBlock` 转成最终 `List[Document]` 切片。
- `embedding_documents`：保持现状，只消费最终 `Document` 切片。

对于 `AssetType.types_skip_loader_and_split()` 的资产，仍然可以直接读取预制切片，并适配成最终 `Document`，不强制经过解析中间态。

## Loader 层拆分

`loader.py` 从单个 `load_file` 拆成三个函数：

```python
def parse_file(original_document, vectorization_config, asset_type) -> List[ParsedBlock]:
    return parse_with_selected_loader(original_document, vectorization_config, asset_type)

def split_parsed_file(parsed_blocks, original_document, vectorization_config, asset_type) -> List[Document]:
    return split_blocks_to_documents(parsed_blocks, original_document, vectorization_config, asset_type)

def load_file(original_document, vectorization_config, asset_type) -> List[Document]:
    parsed_blocks = parse_file(original_document, vectorization_config, asset_type)
    return split_parsed_file(parsed_blocks, original_document, vectorization_config, asset_type)
```

`load_file` 保留为兼容入口，旧调用方可以继续使用。Dagster 新流程优先调用 `parse_file` 和 `split_parsed_file`。

metadata 归一化从 `load_file` 中提取为通用函数：

```python
def normalize_chunk_metadata(doc, original_document, asset_type) -> Optional[Document]:
    return attach_original_document_metadata(doc, original_document, asset_type)
```

这样 Excel、HTML、Docx loader 放入的 `display`、`table`、`headers` 等信息可以原样进入 `extended_metadata`。

## 表格协议

新增 `table` 子模块，沉淀跨格式表格能力：

```text
document_loaders/
  parsed_blocks.py
  table/
    models.py
    renderers.py
    splitters.py
    html_parser.py
    docx_parser.py
    excel_parser.py
```

核心职责：

- `models.py`：定义 `TableBlock`、`TableCell`、`TableContext`。
- `renderers.py`：生成 `search_text`、`markdown_table`、`html_table`。
- `splitters.py`：按表格行数、token 长度或表格区域切分。
- `html_parser.py`：接管 HTML 表格的 rowspan、colspan、caption、上下文抽取。
- `docx_parser.py`：从 python-docx 的 `Table` 转成 `TableBlock`。
- `excel_parser.py`：处理 sheet 分区、多表识别、合并单元格、多层表头和说明文本。

最终向量切片仍然是 LangChain `Document`：

```python
Document(
    page_content="文件: xx.xlsx\nSheet: 销售\n表格: 区域销售统计\n字段: 区域 / 产品 / Q1 / Q2 / 合计\n区域=华东, 产品=A, Q1=100, Q2=120, 合计=220",
    metadata={
        "source": "/data/xx.xlsx",
        "block_type": "table",
        "display": {
            "type": "table",
            "format": "html",
            "content": "<table><tr><th>区域</th><th>产品</th></tr><tr><td>华东</td><td>A</td></tr></table>",
        },
        "table": {
            "source_type": "excel",
            "sheet_name": "销售",
            "cell_range": "A5:E20",
            "title": "区域销售统计",
        },
    },
)
```

## Excel 复杂解析接入

Excel loader 不再假设“一个 sheet 一张表、第一行是表头”。解析阶段改为：

1. 读取 sheet 的原始网格和合并单元格信息。
2. 根据非空区域、空行、空列、边框样式和内容密度识别候选 block。
3. 将说明性文本识别为 `text` block 或 table context。
4. 将每个候选表格区域转成 `TableBlock`。
5. 识别多层表头，生成 `header_rows` 和 `flatten_headers`。
6. 生成 `search_text` 用于检索，生成 `html_table` 用于前端展示。

LLM 可以作为可选增强：只把压缩后的候选表格区域、坐标、样例行、合并单元格和说明文本给 LLM，用于判定表头层级、表名和说明文本归属。第一阶段先保留规则解析，后续再接 LLM 辅助。

## HTML、Docx、Markdown 迁移

HTML 当前存在两条表格路径：正文解析中的 `_process_table_tag` 和额外的 `process_html_table`。迁移后应避免重复：

- 正文解析遇到 table 时，只保留简短占位或摘要。
- 完整表格由 `html_parser.py` 生成 `TableBlock`。

Docx 当前 `_handle_table` 直接返回 Markdown 并拼进章节内容。迁移后：

- 段落和标题继续生成 `text` block。
- 表格使用 `docx_parser.py` 生成 `TableBlock`。
- 表格所在标题路径写入 `headers` 和 `table.title`。

Markdown 先转 HTML 的路径可以复用 HTML parser。Markdown 表格在转换为 HTML 后进入统一 `TableBlock`。

## 切片策略

`split_parsed_documents` 根据 `split_policy` 选择策略：

- `text`：使用现有 text splitter。
- `table_rows`：按表头加若干数据行切片，每片都带表名、标题路径、字段信息。
- `no_split`：适合短代码块、短说明、单图描述。

表格切片的 `page_content` 应包含：

- 文件名、sheet/page/section 信息。
- 表名或上级标题。
- 字段列表。
- 当前行数据，采用压缩 Markdown。

`metadata.display` 保留 HTML 表格展示内容。对于大表，可以只放当前切片对应的 HTML 子表，也可以放 `display_ref` 指向解析文档存储中的完整表格。

## 前端和检索兼容

检索结果返回协议建议新增可选顶层字段：

```python
{
    "text": doc.page_content,
    "metadata": doc.metadata,
    "display": doc.metadata.get("extended_metadata", {}).get("display"),
    "score": score,
    "es_doc_id": es_doc_id,
    "es_index": es_index,
}
```

前端逻辑：

```typescript
if (chunk.display?.type === "table" && chunk.display?.format === "html") {
  renderHtmlTable(chunk.display.content)
} else {
  renderText(chunk.text)
}
```

非表格文档没有 `display`，继续走 `text` 渲染。

## 错误处理

- 解析失败：沿用现有 `FileTooLarge`、`UnicodeDecodeError`、`BadZipFile`、`ValueError` 和通用异常处理，更新文档加载状态。
- 单个表格解析失败：记录日志，降级为普通文本 block，不影响整个文档。
- 展示 HTML 生成失败：保留 `text`，不写 `display`。
- 文档存储失败：第一阶段建议让 `save_parsed_documents` 失败并重试，避免向量库和解析存储状态不一致。

## 测试策略

- `parse_file` 和 `split_parsed_file` 分别增加单元测试。
- `load_file` 增加兼容性测试，确保旧入口输出仍为 `List[Document]`。
- Excel 增加覆盖样例：说明行、多层表头、单 sheet 多表、合并单元格、空行空列。
- HTML 增加覆盖样例：thead/tbody、rowspan、colspan、caption、无表头表格。
- Docx 增加覆盖样例：标题下表格、普通段落与表格混排。
- Dagster op 增加轻量测试，验证 parse、save、split 节点输入输出类型。

## 迁移步骤

1. 新增 `ParsedBlock`、`DisplayPayload`、`TableBlock` 数据结构。
2. 拆分 `loader.py`：新增 `parse_file`、`split_parsed_file`，保留 `load_file` 兼容。
3. 新增 Dagster op：`parse_original_documents`、`save_parsed_documents`、`split_parsed_documents`。
4. 先将现有 loader 输出适配成 `text` block，不改变业务结果。
5. 抽取 HTML 表格逻辑到统一 table parser，消除重复表格切片。
6. 抽取 Docx 表格逻辑到统一 table parser。
7. 重构 Excel loader，接入复杂表格识别并输出 `TableBlock`。
8. 检索返回层提升 `extended_metadata.display` 为顶层 `display`。

## 验收标准

- Dagster UI 中能看到解析、解析结果保存、切片三个节点。
- 原有 docx/pdf/txt/html/markdown/excel 基础向量化流程保持可用。
- 表格切片的 `page_content` 不包含大量 HTML 标签。
- 表格切片 metadata 中可带 `display`，前端可按类型渲染。
- Excel 复杂表格可以被拆成多个表格块，并保留 sheet、range、表头和展示 HTML。
- `load_file` 兼容旧调用方式。
