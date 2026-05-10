# OpenCode Agent Retrieval Skill Design

## Goal

Expose knowledge base retrieval as a standalone agent capability for personal AI tools such as OpenCode.

The service should provide fast, low-cost retrieval primitives. The personal AI performs query planning, evidence sufficiency judgment, follow-up exploration, and final answer generation. The service does not call an LLM for rewrite, rerank, or answer synthesis in the default path.

## Product Boundary

This is a new capability, separate from the existing knowledge base QA API surface.

Do not place the new endpoints in `knowledge_base_api.py` or implement the flow inside the large `knowledge_base_service.py`. Add a dedicated module:

```text
rag_service/
  agent_retrieval/
    __init__.py
    router.py
    service.py
    models.py
    skill_generator.py
    templates/
      opencode_skill.md
      api_schema.md
      config.example.json
      kb_retrieval.py
```

The module may reuse lower-level retrieval, permission, vector store, artifact, and evidence package helpers, but it owns the external agent contract.

## First Version Delivery Shape

Ship an OpenCode-compatible skill package that contains instructions and a Python client script.

Generated package:

```text
kb-retrieval/
  SKILL.md
  config.example.json
  references/
    api_schema.md
  scripts/
    kb_retrieval.py
```

The user installs the folder into one of OpenCode's skill locations, for example:

```text
.opencode/skills/kb-retrieval/
~/.config/opencode/skills/kb-retrieval/
```

The user configures:

```text
KB_RETRIEVAL_BASE_URL
KB_RETRIEVAL_TOKEN
KB_SN_LIST
```

or copies `config.example.json` to a local config file and fills in the same values.

OpenCode loads `SKILL.md` on demand. The skill instructs the agent to call the Python script through the shell when it needs to search or inspect knowledge base content.

## Why Script-Based First

Use Python scripts instead of remote MCP in the first version.

Benefits:

- Lower integration cost for users.
- No MCP server lifecycle or protocol compatibility work.
- Easy to inspect and debug because users can run the same script manually.
- Keeps the server as plain REST APIs.
- Allows a future OpenCode custom tool wrapper to call the same script without changing the server contract.

Future versions may add `.opencode/tools/*.ts` wrappers or a remote MCP endpoint, but they should be adapters over the same REST API.

## API Prefix

Use a dedicated prefix:

```text
/agent/retrieval
```

Initial endpoints:

```text
POST /agent/retrieval/search_slices
POST /agent/retrieval/get_document_outline
POST /agent/retrieval/get_section
POST /agent/retrieval/get_table
POST /agent/retrieval/get_original_text
GET  /agent/retrieval/opencode/skill-package
```

The API accepts `kb_sn_list` from the request body or a configured header. The Python script should default to its configured `KB_SN_LIST`, so OpenCode users do not need to repeat knowledge base serial numbers in every prompt.

## Retrieval Behavior

`search_slices` is the default first tool call.

It performs fast candidate retrieval across the configured knowledge bases and returns slice-level metadata. Default behavior:

- No question rewrite.
- No LLM answer generation.
- No LLM rerank.
- Use existing vector/full-text retrieval according to the knowledge base config, unless the request explicitly overrides allowed retrieval options.
- Return enough metadata for the personal AI to choose follow-up calls.
- Keep server-side top-k bounded.

Recommended defaults:

```text
top_k: 20
max_top_k: 50
include_refs: true
include_artifact_handles: true
```

Response shape:

```json
{
  "query": "original user query",
  "kb_sn_list": ["kb-1"],
  "slices": [
    {
      "slice_id": "opaque-slice-id",
      "rank": 1,
      "text": "retrieved slice text",
      "score": 0.82,
      "doc": {
        "doc_id": "doc-1",
        "title": "Document title",
        "source": "document.docx",
        "asset_name": "asset-1",
        "kb_sn": "kb-1"
      },
      "location": {
        "block_id": "block_001",
        "block_index": 3,
        "block_type": "text",
        "section_id": "section_001",
        "section_title": "Risk",
        "section_path": "Report > Risk",
        "table_id": null,
        "row_range": null
      },
      "actions": {
        "can_get_section": true,
        "can_get_table": false,
        "can_get_original_text": true
      },
      "handles": {
        "section_handle": "opaque-section-handle",
        "table_handle": null,
        "document_handle": "opaque-document-handle"
      }
    }
  ]
}
```

Use opaque handles for external consumers. The agent API should avoid leaking raw object storage keys by default.

## Exploration Endpoints

### `get_document_outline`

Purpose: help the personal AI understand document structure before deeper exploration.

Input:

```json
{
  "doc_id": "doc-1",
  "document_handle": "opaque-document-handle"
}
```

Output:

```json
{
  "doc_id": "doc-1",
  "title": "Document title",
  "sections": [
    {
      "section_id": "section_001",
      "title": "Risk",
      "headers": ["Report", "Risk"],
      "block_count": 4,
      "section_handle": "opaque-section-handle"
    }
  ],
  "tables": [
    {
      "table_id": "table_001",
      "title": "Sales detail",
      "row_count": 120,
      "table_handle": "opaque-table-handle"
    }
  ]
}
```

### `get_section`

Purpose: fetch the full section or a bounded merged section around a retrieved slice.

Input:

```json
{
  "section_handle": "opaque-section-handle",
  "max_chars": 12000
}
```

Output:

```json
{
  "section_id": "section_001",
  "title": "Risk",
  "headers": ["Report", "Risk"],
  "mode": "full_section",
  "text": "section markdown text"
}
```

### `get_table`

Purpose: fetch a table in the form most useful to the personal AI.

Input:

```json
{
  "table_handle": "opaque-table-handle",
  "mode": "llm_text"
}
```

Allowed modes:

```text
llm_text
json
html
summary
```

Output:

```json
{
  "table_id": "table_001",
  "title": "Sales detail",
  "mode": "llm_text",
  "row_count": 120,
  "content": "table content"
}
```

### `get_original_text`

Purpose: fetch nearby original text when a slice is too narrow or ambiguous.

Input:

```json
{
  "document_handle": "opaque-document-handle",
  "center_block_id": "block_001",
  "before": 2,
  "after": 2,
  "max_chars": 12000
}
```

Output:

```json
{
  "doc_id": "doc-1",
  "mode": "neighbor_blocks",
  "blocks": [
    {
      "block_id": "block_000",
      "block_index": 2,
      "text": "previous block"
    }
  ]
}
```

## Skill Instructions

The generated `SKILL.md` should be concise and procedural.

Core behavior:

1. Start with `search_slices` for any knowledge-base question.
2. Treat search results as evidence candidates, not final truth.
3. If top slices are enough, answer from them and cite document titles and section/table names.
4. If a slice is in a section and the answer depends on surrounding context, call `get_section`.
5. If a slice is in a table, call `get_table` with `llm_text` or `summary`; use `json` only when exact row/column structure matters.
6. If the question is broad, ambiguous, or asks about document structure, call `get_document_outline`.
7. If a retrieved slice is truncated or lacks context, call `get_original_text`.
8. If results are weak, rewrite the search query locally and call `search_slices` again.
9. Do not claim facts that are not grounded in returned slices, sections, tables, or original text.

The skill should show exact script commands rather than requiring the model to infer the CLI.

Example commands:

```bash
python scripts/kb_retrieval.py search --query "..." --top-k 20
python scripts/kb_retrieval.py section --section-handle "..."
python scripts/kb_retrieval.py table --table-handle "..." --mode llm_text
python scripts/kb_retrieval.py outline --document-handle "..."
python scripts/kb_retrieval.py original-text --document-handle "..." --center-block-id "block_001" --before 2 --after 2
```

## Python Script Responsibilities

`scripts/kb_retrieval.py` should:

- Read config from env vars first, then optional config file.
- Provide subcommands matching the REST endpoints.
- Print compact JSON by default.
- Support `--pretty` for human-readable debugging.
- Fail with clear non-zero exit codes.
- Avoid storing tokens in generated logs or outputs.
- Include a short timeout and useful error messages for auth, permission, network, and server errors.

It should not implement retrieval logic locally. It is a thin HTTP client.

## Server Responsibilities

The `agent_retrieval` service should:

- Authenticate the user or token.
- Validate access to all requested `kb_sn` values.
- Enforce top-k and content-size limits.
- Convert raw metadata and artifact refs into opaque handles.
- Resolve opaque handles back to authorized artifacts or document locations.
- Reuse existing artifact readers for sections, tables, and document markdown.
- Log request id, user id, endpoint, kb count, result count, and latency.
- Avoid LLM calls in the default path.

## Handle Design

Opaque handles should encode or reference:

```text
uid or token subject
kb_sn
doc_id
artifact type
artifact key or logical location
expiry timestamp
signature
```

The first version may use signed, URL-safe serialized payloads if there is already an internal signing utility. If not, introduce a small signer helper with a server secret. Handles should expire to limit leakage risk.

## Error Handling

- Unauthorized token: return 401.
- User lacks access to a knowledge base: return 403 without exposing whether hidden docs exist.
- Invalid or expired handle: return 403.
- Missing artifact: return 404 only after permission passes.
- Oversized section/table/original text: return a bounded response with `truncated: true`.
- Retrieval timeout: return partial results if available and include a warning.

## Compatibility With Existing Evidence API

The new module does not replace:

```text
/kb/get_evidence_packages
/kb/get_answer_evidence
/kb/get_related_docs
```

Existing API mode and agent/debug mode remain available for web QA and internal workflows.

The new agent retrieval API can reuse:

- structured metadata in Elasticsearch
- `EvidenceItem` conversion logic
- table and section artifact loading
- document grouping helpers
- existing permission helpers

but should expose a smaller, stable contract specialized for personal AI exploration.

## Testing

Add focused tests for:

- Router registration under `/agent/retrieval`.
- `search_slices` does not call question rewrite, rerank, or answer generation.
- `search_slices` returns slice metadata and action handles.
- Access checks reject unauthorized `kb_sn`.
- Handles cannot be used by a different user.
- `get_section` resolves a valid section handle and enforces `max_chars`.
- `get_table` supports `llm_text`, `json`, `html`, and `summary` modes where artifacts exist.
- `get_original_text` returns bounded neighboring blocks.
- Skill package generation includes `SKILL.md`, `scripts/kb_retrieval.py`, `config.example.json`, and `references/api_schema.md`.
- The generated Python script builds the expected HTTP requests.

## Future Extensions

After the script-based version is stable, add optional OpenCode custom tool wrappers:

```text
.opencode/tools/kb-retrieval.ts
```

These wrappers can call the bundled Python script and expose named OpenCode tools such as:

```text
kb_search_slices
kb_get_section
kb_get_table
kb_get_document_outline
kb_get_original_text
```

Remote MCP support can be added later if customers prefer centralized tool configuration over script-based skills.
