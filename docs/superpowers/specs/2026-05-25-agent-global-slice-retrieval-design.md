# Agent Selectable Global Slice Retrieval Design

## Goal

`POST /agent/retrieval/search_slices` should retrieve globally relevant evidence slices from multiple user-selected knowledge bases while returning `section`, `table`, and `original-text` handles for agent follow-up exploration.

Each request explicitly selects one retrieval backend: Libing RAG (local Elasticsearch) or IPD RAG. Results from the two systems are never mixed in one request. Optional reranking applies consistently to either backend.

This feature must not change the behavior of the existing `get_related_docs` retrieval path.

## API Contract

Extend `SearchSlicesRequest` with:

```python
retrieval_backend: Literal["libing", "ipd"] = "libing"
enable_rerank: bool = False
```

Defaults preserve existing callers: requests that omit both fields use Libing retrieval without reranking.

Extend `SearchSlicesResponse` with:

```python
request_id: str
```

The identifier allows the caller to correlate returned slices with `RequestResponseLog`.

Existing `query`, `kb_sn`/`kb_sn_list`, `top_k`, `include_refs`, and `include_artifact_handles` fields remain accepted. The final result limit remains bounded to `1..50`.

## Retrieval Flow

For every `search_slices` request:

1. Obtain or create a `request_id`, identify the caller, and apply the existing read-permission validation to all requested KBs.
2. Set `final_top_k = clamp(request.top_k, 1, 50)`.
3. Set `candidate_top_k = final_top_k` for either backend. Libing retains its existing Elasticsearch candidate amplification; the agent endpoint does not inflate the requested IPD size.
4. Dispatch only to the selected `retrieval_backend`.
5. Remove exact duplicate slice texts before final selection for IPD, multi-KB Libing retrieval, and reranked single-KB Libing retrieval where multiple embedding models can return the same slice.
6. If reranking is enabled, rerank the candidate list once and return at most `final_top_k`; otherwise sort by backend score descending and return at most `final_top_k`.
7. Convert candidates through the existing agent slice projection so artifact handles are returned where metadata supports them.
8. Persist a retrieval log record and return `request_id` with the slices.

## Libing RAG Backend

For a single authorized KB, continue using its local vector-store configuration and indexes, with `final_top_k` passed as the manager retrieval size. Elasticsearch retains its existing internal candidate amplification. When reranking is enabled, exact duplicate texts returned across embedding models are removed before reranking.

For multiple authorized KBs:

1. Derive one configuration using the established multi-KB retrieval configuration helper.
2. Obtain vector stores grouped by compatible analyzer and embedding model for the complete KB selection.
3. Call the Elasticsearch retrieval manager once for each compatible group so indexes in that group compete within one search.
4. Merge results from separate compatibility groups, deduplicate exact text, and sort by score unless reranking is enabled.

This replaces the current `search_slices` behavior of performing independent per-KB calls and comparing their scores afterward. It is intentionally local to the agent endpoint and does not modify shared `get_related_docs` orchestration.

## IPD RAG Backend

For `retrieval_backend="ipd"`:

1. Validate permissions for every requested KB through the same permission gate.
2. Retain only selected KBs with an `ipd_rag_kb_id` mapping.
3. Query IPD RAG using those mapped IDs and `candidate_top_k`.
4. Skip authorized KBs without an IPD mapping, recording the skipped list in diagnostics; do not silently fall back to Libing and do not fail the complete request.
5. If no selected KB is mapped, return an empty slice list and log that outcome.

## Reranking

`enable_rerank` is the only switch that controls reranking for `search_slices`.

- When false, neither backend is reranked; raw backend scores determine output order.
- When true for Libing, the endpoint passes `final_top_k` to the existing retrieval manager, consumes its Elasticsearch-expanded candidates, removes duplicate texts, applies rerank once, and returns the highest ranked `final_top_k` candidates.
- When true for IPD, the endpoint requests `final_top_k` candidates, applies rerank once, and returns at most `final_top_k` candidates.
- The endpoint reuses established retrieval configuration selection rather than exposing a model parameter: a single KB uses its KB configuration, and multiple KBs use the shared multi-KB configuration.
- If rerank invocation fails and the existing fallback returns backend candidates, the response remains usable and the diagnostic log records rerank degradation.

## Agent Handles

The existing `_search_slice()` mapping remains responsible for text, score, document identity, location, actions, and handles.

- Libing candidates with structured artifact references retain their document, section, and table handles.
- IPD candidates produce the same handles only when their returned metadata contains the corresponding artifact references.
- An IPD slice without artifact metadata is still returned as evidence text; its unavailable handles are null and the related actions are false.
- This change does not add a second metadata enrichment call for IPD results.

## Logging

Logging is added for `search_slices` only. The document-exploration operations (`get_document_outline`, `get_section`, `get_table`, and `get_original_text`) are unchanged.

Each call writes or updates a `RequestResponseLog` record with:

- `request_id`, `user_id`, `method_name="agent.retrieval.search_slices"`, question, and selected KBs;
- request/retrieval start and end times where supported by the existing log helper;
- `retrieve_result` containing the final returned `SearchSlice` objects as JSON;
- `error_reason` when retrieval fails;
- `extra_info` diagnostics.

Diagnostics include at least:

```json
{
  "retrieval_backend": "libing",
  "enable_rerank": true,
  "requested_top_k": 20,
  "final_top_k": 20,
  "candidate_top_k": 20,
  "returned_slice_count": 20,
  "kb_sn_list": ["kb-a", "kb-b"],
  "candidate_count_before_dedup": 130,
  "candidate_count_after_dedup": 100,
  "libing_manager_top_k": 20,
  "rerank_requested": true,
  "rerank_degraded": false,
  "ipd_mapped_kb_sn_list": [],
  "ipd_skipped_unmapped_kb_sn_list": []
}
```

For Libing retrieval, diagnostics additionally record the manager `top_k` and embedding/analyzer groups queried. This feature does not require new `RequestDocumentHit` rows.

## Error Handling

- Missing user identity, missing database session, and KB permission failures retain existing failure behavior.
- Invalid backend values are rejected by request validation.
- IPD mapping gaps follow the skip-and-log rule above.
- Backend failures remain visible to the caller and are recorded in the request/response log when logging infrastructure is available.

## Testing

Focused tests must prove:

1. Legacy requests default to Libing without reranking and now return a `request_id`.
2. Libing retrieval uses the existing single-KB selection behavior for one KB and grouped global retrieval plus text deduplication for multiple KBs.
3. Duplicate slices across Libing groups are removed before selecting final results.
4. Selecting IPD avoids Elasticsearch retrieval, passes only mapped IDs, skips/logs unmapped KBs, and returns empty output when none map.
5. With reranking disabled, candidate retrieval uses bounded `final_top_k`.
6. With reranking enabled, either backend receives bounded `final_top_k`; Libing reranks its backend-expanded candidates and IPD reranks the candidates it returns, with both producing at most 50 slices.
7. Rerank fallback remains usable and is observable in diagnostics.
8. Structured metadata creates agent handles, while missing IPD metadata produces text-only slices.
9. `RequestResponseLog.retrieve_result` records final slices and `extra_info` records backend, rerank, candidate-size, and IPD mapping diagnostics.
10. The generated agent skill contract can send the new request parameters.
11. Existing `get_related_docs` tests or a focused non-interaction regression test remain unchanged and pass.

## Expected Files

- `rag_service/agent_retrieval/models.py`: request backend/rerank fields and response `request_id`.
- `rag_service/agent_retrieval/router.py`: request-id and logging integration.
- `rag_service/agent_retrieval/service.py`: backend dispatch, global Libing behavior, IPD dispatch, candidate sizing, deduplication, reranking, and diagnostics.
- `rag_service/agent_retrieval/skill_generator.py`: generated contract/script parameters.
- Existing logging helper module, or a narrowly scoped agent logging helper if the imported application helper is unavailable in this checkout.
- `tests/test_agent_retrieval.py` and focused log/contract regression tests.

No code in the normal `get_related_docs` endpoint is part of this change.
