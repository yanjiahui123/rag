---
name: kb-retrieval
description: 当回答需要企业知识库、内部文档、故障处理记录、SOP、手册、历史案例、文档章节、表格、目录或相邻原文作为证据的问题时，使用此 skill。
---

# KB 检索

使用此 skill 搜索已配置的知识库，并查看来源证据。服务只返回检索证据；是否需要继续探索、以及最终答案如何组织，由你决定。

## 执行规则

优先使用 request 文件执行器。这样可以避免把中文、引号或动态参数直接放进 shell 参数里。

先编辑 `.opencode/skills/kb-retrieval/request.json`，然后从工作区根目录运行这个固定的无参数命令：

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

如果当前目录已经在 `kb-retrieval` skill 目录内，使用：

```bash
python scripts/kb_retrieval_request.py
```

正常使用时，不要把搜索问题放在 shell 命令行参数里。动态值都写到 `request.json`。

在 bash 中不要在命令前拼接 Windows 盘符路径形式的 `cd`。如果工具支持工作目录参数，使用工具的 working-directory 选项。如果必须使用 Windows 绝对路径，Python 脚本路径请使用正斜杠，例如：`python D:/project/knowledge_base_answer/.opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py`。

默认情况下，request 执行器会输出原始 UTF-8，保证中文可读。如果终端或 agent 日志里出现乱码，在 `request.json` 中设置 `"utf8_output": false`，切回 ASCII-safe JSON 输出。

## 配置

编辑此 skill 目录下的 `config.json`：

```json
{
  "base_url": "http://localhost:8001",
  "uid": "your-employee-id",
  "kb_sn_list": ["your-kb-sn"],
  "headers": {
    "X-HW-ID": "your-hw-id",
    "X-HW-APPKEY": "your-hw-appkey"
  },
  "timeout_seconds": 300
}
```

需要时，环境变量可以覆盖文件配置：

```bash
export KB_RETRIEVAL_BASE_URL="http://localhost:8001"
export KB_RETRIEVAL_UID="your-employee-id"
export KB_SN_LIST="your-kb-sn"
```

请在 `headers` 中填写 `X-HW-ID` 和 `X-HW-APPKEY`；客户端会在每次请求时把它们作为 HTTP headers 发送。

## 工作流

1. 每次知识库问题都从编辑 `request.json` 开始：

```json
{
  "command": "search",
  "pretty": true,
  "utf8_output": true,
  "query": "USER_QUERY_HERE",
  "top_k": 20,
  "retrieval_backend": "libing",
  "enable_rerank": false
}
```

然后运行：

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

阅读返回的标题、片段、分数、短 OBS-key 文档 handle、章节 handle、表格 handle 和 block 标识。

2. 把返回的切片视为候选证据。如果某个切片需要上下文，把返回的 handle 写入 `request.json`：

```json
{
  "command": "section",
  "pretty": true,
  "section_handle": "...",
  "max_chars": 12000
}
```

然后运行：

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

3. 如果切片来自表格，用 `llm_text` 或 `summary` 模式获取表格：

```json
{
  "command": "table",
  "pretty": true,
  "table_handle": "...",
  "mode": "llm_text",
  "max_chars": 40000
}
```

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

4. 对于宽泛问题或结构性问题，查看文档目录：

```json
{
  "command": "outline",
  "pretty": true,
  "document_handle": "..."
}
```

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

5. 如果切片范围太窄，获取相邻原文：

```json
{
  "command": "original-text",
  "pretty": true,
  "document_handle": "...",
  "center_block_id": "block_001",
  "before": 2,
  "after": 2,
  "max_chars": 12000
}
```

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

6. 仅当所选知识库已经映射到 IPD RAG 时，才使用 `"retrieval_backend": "ipd"`。如果需要更好的排序质量，可以设置 `"enable_rerank": true`，在返回请求的 `top_k` 前对检索候选进行 rerank。

7. 如果结果较弱，在本地改写问题后再次搜索。

最终答案只能基于返回的切片、章节、表格或原文。可用时，请引用文档标题和章节/表格名称。

## 故障处理

如果 `python` 不可用，使用 `python3` 重试同一命令。

如果 request 执行器路径不存在，运行 `pwd` 并列出 `.opencode/skills/kb-retrieval/scripts/`，确认 skill 的实际安装位置。然后用真实的 `kb_retrieval_request.py` 路径重试。

如果 bash 报错找不到类似 `D:project...` 的路径，说明命令在 bash 下使用了 Windows 反斜杠路径。请去掉 `cd`，改用工具的工作目录参数，或使用正斜杠形式的 Python 脚本绝对路径。

如果输出乱码，在 `request.json` 中设置 `"utf8_output": false` 后重试。ASCII-safe JSON 会使用 `\uXXXX` 转义，适用于不可靠的编码路径。

仅在手动终端调试时使用 `scripts/kb_retrieval.sh`；它接受普通 CLI 参数。
