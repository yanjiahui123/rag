---
name: kb-retrieval
description: Use when answering questions that require evidence from configured enterprise knowledge bases, internal documents, troubleshooting records, SOPs, manuals, historical cases, document sections, tables, outlines, or nearby original text.
---

# KB Retrieval

Use this skill to search configured knowledge bases and inspect source evidence. The service returns retrieval evidence only; you decide whether more exploration is needed and write the final answer.

## Execution Rule

Prefer the request-file runner. It avoids passing Chinese text, quotes, or dynamic arguments through shell argv.

First edit `.opencode/skills/kb-retrieval/request.json`, then run this fixed no-argument command from the workspace root:

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

If already running from inside the `kb-retrieval` skill directory, use:

```bash
python scripts/kb_retrieval_request.py
```

Do not pass the search query on the shell command line for normal use. Put dynamic values in `request.json`.

Do not prepend a Windows drive-path `cd` before the command when the shell is bash. Use the tool's working-directory option when available. If an absolute Windows path is necessary, use forward slashes in the Python script path, for example `python D:/project/knowledge_base_answer/.opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py`.

By default, the request runner prints raw UTF-8 so Chinese text remains readable. If the terminal or agent log path displays mojibake, set `"utf8_output": false` in `request.json` to switch back to ASCII-safe JSON.

## Configuration

Edit `config.json` in this skill folder:

```json
{
  "base_url": "http://localhost:8001",
  "uid": "your-employee-id",
  "kb_sn_list": ["your-kb-sn"],
  "timeout_seconds": 30
}
```

Environment variables can override the file when needed:

```bash
export KB_RETRIEVAL_BASE_URL="http://localhost:8001"
export KB_RETRIEVAL_UID="your-employee-id"
export KB_SN_LIST="your-kb-sn"
```

## Workflow

1. Start every knowledge-base question by editing `request.json`:

```json
{
  "command": "search",
  "pretty": true,
  "utf8_output": true,
  "query": "USER_QUERY_HERE",
  "top_k": 20
}
```

Then run:

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

Read the returned titles, snippets, scores, short OBS-key document handles, section handles, table handles, and block identifiers.

2. Treat returned slices as candidate evidence. If a slice needs surrounding context, edit `request.json` with the returned handle:

```json
{
  "command": "section",
  "pretty": true,
  "section_handle": "...",
  "max_chars": 12000
}
```

Then run:

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

3. If the slice is from a table, fetch the table in `llm_text` or `summary` mode:

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

4. For broad or structural questions, inspect the document outline:

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

5. If the slice is too narrow, fetch neighboring original text:

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

6. If results are weak, rewrite the query locally and search again.

Only answer with facts grounded in returned slices, sections, tables, or original text. Cite document titles and section/table names when available.

## Failure Handling

If `python` is unavailable, retry the same command with `python3`.

If the request runner path does not exist, run `pwd` and list `.opencode/skills/kb-retrieval/scripts/` to confirm where the skill was installed. Retry with the actual `kb_retrieval_request.py` path.

If bash reports that `cd` cannot find a path like `D:project...`, the command used a Windows backslash path under bash. Retry without `cd`, set the tool working directory instead, or use a forward-slash absolute Python script path.

If output is garbled, set `"utf8_output": false` in `request.json` and retry. ASCII-safe JSON uses `\uXXXX` escapes and is intended for unreliable encoding paths.

For manual terminal debugging only, `scripts/kb_retrieval.sh` accepts normal CLI arguments.
