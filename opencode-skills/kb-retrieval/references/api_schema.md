# Agent Retrieval API

Base path: `/agent/retrieval`

Configure `X-HW-ID` and `X-HW-APPKEY` in `config.json`; the client sends them as HTTP headers for every API call.

## `POST /search_slices`

Body:

```json
{"uid": "employee-id", "query": "question", "kb_sn_list": ["kb-1"], "top_k": 20}
```

Returns ranked slices with document metadata, location metadata, action flags, and short OBS-key handles.

## `POST /get_document_outline`

Body:

```json
{"uid": "employee-id", "document_handle": "..."}
```

Returns sections and tables discovered from the document manifest. `document_handle` is a `.../manifest.json` OBS key.

## `POST /get_section`

Body:

```json
{"uid": "employee-id", "section_handle": "...", "max_chars": 12000}
```

Returns section markdown text. `section_handle` is a `.../sections/<section>.md` OBS key.

## `POST /get_table`

Body:

```json
{"uid": "employee-id", "table_handle": "...", "mode": "llm_text"}
```

Modes: `llm_text`, `json`, `html`, `summary`. `table_handle` is a `.../tables/<table>.llm.md` OBS key; sibling `.json` and `.html` artifacts are derived from it when those modes are requested.

## `POST /get_original_text`

Body:

```json
{"uid": "employee-id", "document_handle": "...", "center_block_id": "block_001", "before": 2, "after": 2}
```

Returns bounded neighboring block text when available.
