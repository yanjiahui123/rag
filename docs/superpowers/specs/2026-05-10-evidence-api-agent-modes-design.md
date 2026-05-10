# Evidence API And Agent Modes Design

## Goal

Split evidence retrieval output into two stable consumer-facing modes while keeping one internal retrieval and package-building pipeline.

The API mode is for web question answering. It should return ready-to-use LLM context and ready-to-render page display sources. The agent mode is for tools and agents. It should return full slice-level retrieval details, artifact references, and source metadata so the caller can plan follow-up actions.

## Current Problem

The current evidence package response exposes an internal rich model directly. It mixes three responsibilities:

- LLM context text for answer generation.
- Display artifacts for the web page, especially table display HTML.
- Retrieval internals for agents and debugging, including block ids, row ranges, refs, Elasticsearch ids, and package artifacts.

This makes the web and QA callers depend on internal evidence assembly details. It also makes future changes to section merging, table expansion, artifact storage, or slice selection harder because those internals become part of the external contract.

## Design Summary

Keep a single internal evidence package builder:

```text
retrieve candidates
  -> build internal rich evidence model
  -> API presenter
  -> agent presenter
```

The internal model can stay rich and implementation-oriented. Public endpoint responses should be projections of that model.

## Endpoints

### API mode

Add:

```text
POST /kb/get_answer_evidence
```

Purpose: serve web QA flows.

The endpoint returns:

- `llm_context`: ordered evidence content that the QA platform can put into the prompt.
- `display_sources`: document-level display data that the frontend can render as answer sources.

The frontend should not need to know object storage keys, table JSON refs, LLM table refs, manifests, block ids, or Elasticsearch ids in this mode.

### Agent mode

Keep:

```text
POST /kb/get_evidence_packages
```

Purpose: serve agent, tool, and debug flows.

This endpoint should return complete package and slice details. It may continue to expose artifacts, refs, block ids, section ids, table ids, row ranges, and Elasticsearch ids.

If a clearer public name is needed later, add `/kb/get_agent_evidence_packages` as an alias and preserve `/kb/get_evidence_packages` for compatibility.

### Artifact access

Add an artifact proxy:

```text
GET /kb/evidence_artifacts/{artifact_id}
```

`artifact_id` is an opaque server-generated id, not a raw object storage key. The proxy validates user access, fetches the object storage artifact, and returns the correct content type. API-mode display responses should expose frontend-safe `display_url` values that point to this proxy instead of raw object storage keys.

## API Mode Response

The response shape should be intentionally small:

```json
{
  "query": "original question",
  "rewrite_query": "rewritten question",
  "llm_context": [
    {
      "context_id": "C1",
      "source_id": "S1",
      "title": "Sales Report",
      "content": "Evidence text prepared for the LLM.",
      "content_type": "text",
      "score": 0.91
    }
  ],
  "display_sources": [
    {
      "source_id": "S1",
      "title": "Sales Report",
      "source": "sales.xlsx",
      "score": 0.91,
      "items": [
        {
          "type": "text",
          "context_id": "C1",
          "title": "Risk",
          "content": "Text shown in the answer source panel."
        },
        {
          "type": "table",
          "context_id": "C2",
          "title": "Sales detail",
          "display_url": "/kb/evidence_artifacts/abc",
          "row_count": 120,
          "hit_row_ranges": [[0, 84]]
        }
      ]
    }
  ]
}
```

Rules:

- `llm_context[].content` is the only field the QA platform needs for prompt construction.
- `display_sources[].items[]` is the only field the frontend needs for rendering answer sources.
- `context_id` connects displayed items to LLM context items.
- Table display uses `display_url`; the page does not receive raw `display_ref` values.
- Text display items contain renderable text directly.
- Table LLM context may use matched chunk text, expanded `llm_table` text, or a bounded table summary depending on current table expansion rules.

## Agent Mode Response

The agent response should expose the retrieval scene, not a simplified display model:

```json
{
  "query": "original question",
  "rewrite_query": "rewritten question",
  "packages": [
    {
      "doc_id": "doc-1",
      "title": "Sales Report",
      "source": "sales.xlsx",
      "score": 0.91,
      "artifacts": {
        "document": {},
        "sections": [],
        "tables": []
      },
      "slices": [
        {
          "slice_id": "doc-1:block-1",
          "rank": 1,
          "text": "Retrieved slice text.",
          "score": 0.91,
          "block_id": "block_1",
          "block_index": 3,
          "block_type": "table",
          "section_id": "section_1",
          "table_id": "table_1",
          "row_range": [0, 20],
          "refs": {
            "display": "doc-1/structured_excel/tables/table_1.html",
            "table_json": "doc-1/structured_excel/tables/table_1.json",
            "llm_table": "doc-1/structured_excel/tables/table_1.llm.md"
          },
          "es_index": "index-name",
          "es_doc_id": "es-id"
        }
      ]
    }
  ]
}
```

Rules:

- `slices` must represent original retrieved candidate slices, not only final promoted or merged evidence items.
- Agent mode may include raw artifact refs because agents need actionable references for follow-up calls.
- If signed URLs are used later, agent mode can either keep raw refs or expose both raw refs and resolved URLs under explicit field names.

## Internal Model Changes

The current final `evidence[]` list is assembled after deduplication, table promotion, section merging, and evidence limiting. It is useful for answer construction, but it is not enough to reconstruct every original retrieved slice for agent mode.

The internal package accumulator should preserve candidate slice records before promotion and limiting. A possible internal shape is:

```text
DocumentEvidencePackage
  artifacts
  evidence          # promoted/merged answer evidence
  candidate_slices  # original deduplicated retrieved slices
```

Only presenter functions should decide which fields leave the service:

- `to_answer_evidence_response()`
- `to_agent_evidence_response()`

## Error Handling

- If artifact display HTML is unavailable, API mode should still return text evidence and omit `display_url` for that table item.
- If an artifact proxy request fails permission checks, return an authorization error rather than leaking whether the object key exists.
- If table expansion cannot inline `llm_table`, the LLM context should fall back to matched table chunk text.
- If no evidence is found, API mode should return empty `llm_context` and `display_sources` arrays.

## Compatibility

- Keep `/kb/get_related_docs` unchanged.
- Keep `/kb/get_answer` and `/kb/get_stream_answer` unchanged unless they later opt into `get_answer_evidence`.
- Preserve `/kb/get_evidence_packages` as the full agent/debug endpoint.
- Add `/kb/get_answer_evidence` for the new simplified API response.

## Testing

Add focused tests for:

- API presenter returns only `llm_context` and `display_sources`.
- API presenter hides raw refs, manifests, block ids, and Elasticsearch ids.
- Table API display items expose `display_url` and hit row ranges.
- LLM context receives inline full-table text when expansion succeeds.
- LLM context falls back to matched chunk text when table artifacts cannot be inlined.
- Agent presenter exposes original candidate slices with refs and retrieval metadata.
- Existing evidence package aggregation tests continue to pass.

## Approved Direction

Use split endpoints:

- `/kb/get_answer_evidence` for API mode.
- `/kb/get_evidence_packages` for agent mode.

Use one internal builder and two presenter functions so retrieval behavior remains consistent across modes.
