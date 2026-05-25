# Agent Global Slice Retrieval Design

## Goal

Align `POST /agent/retrieval/search_slices` with the existing global multi-knowledge-base Elasticsearch retrieval semantics while retaining its agent-facing evidence exploration output.

The endpoint should find globally relevant candidate slices across the user's selected local Elasticsearch-backed knowledge bases and return `section_handle`, `table_handle`, and `document_handle` values for subsequent agent exploration.

## Current Problem

`rag_service/agent_retrieval/service.py::_default_retrieve_documents()` currently loops over authorized knowledge bases and invokes `manager.retrieve()` once for each knowledge base. It then merges independently scored results and sorts them in Python.

This differs from the established multi-KB retrieval path in `rag_service/rag_app/service/knowledge_base_service.py`, which:

- derives one multi-KB retrieval configuration;
- groups compatible Elasticsearch indexes across knowledge bases;
- searches each analyzer/embedding-compatible group as a combined candidate space;
- deduplicates merged candidate text before selecting final results.

The current agent behavior creates three concrete problems:

- independently produced ES scores are merged as though they came from one comparable retrieval space;
- repeated content returned by different knowledge-base calls is not removed globally;
- query embeddings and ES searches are repeated per knowledge base even when indexes are compatible.

## Scope

This change is limited to the default candidate retrieval implementation used by `search_slices`.

In scope:

- preserve user and KB read-permission validation;
- preserve the existing request and `SearchSlicesResponse` contract;
- for multiple selected local ES-backed KBs, use the established grouped-index retrieval shape;
- use a shared multi-KB retrieval configuration for the grouped search;
- merge results from incompatible analyzer groups, remove duplicate text, sort by score, and return global `top_k`;
- continue converting selected candidates into agent slices and artifact handles.

Out of scope:

- question rewrite;
- conversational-history enrichment;
- online QA lookup;
- answer generation;
- LLM reranking;
- combining IPD RAG results with Elasticsearch candidates;
- changes to outline, section, table, or original-text endpoints.

## Retrieval Behavior

### Single Knowledge Base

For one authorized KB, preserve the existing behavior:

1. Derive retrieval configuration from that KB plus request `top_k`.
2. Load that KB's embedding-model-to-vector-store mapping.
3. Invoke `manager.retrieve()` using its analyzer, query strategy, and score threshold.
4. Sort and limit candidates to the bounded request `top_k`.

### Multiple Knowledge Bases

For multiple authorized KBs:

1. Derive one multi-KB configuration through `get_multi_kb_retrieve_param(knowledge_bases, query_request)`.
2. Obtain grouped local vector stores through:

   ```python
   get_grouped_vector_stores_by_knowledge_base_and_asset(
       session,
       {knowledge_base.sn: [] for knowledge_base in knowledge_bases},
   )
   ```

3. For each returned `analyzer -> embedding_model_to_vector_stores` group, call `manager.retrieve()` once with the shared multi-KB configuration.
4. Combine results from the groups.
5. Remove duplicate slices using the existing `filter_same_document()` text-based behavior.
6. Sort remaining candidates by `score` descending.
7. Limit to the bounded request `top_k`.

This creates global competition inside each compatible ES index group while respecting the existing analyzer grouping required by the vector-store layer.

## IPD RAG Boundary

This design does not add IPD RAG retrieval to `search_slices`.

The endpoint remains an atomic Elasticsearch slice retrieval surface. Mixing IPD candidates into this path would require an explicit quality strategy for comparing or reranking scores from different retrieval systems. That is a separate design decision.

Consequently, this change only aligns selected local Elasticsearch-backed KBs. Any future requirement for agent exploration over IPD-only KBs should be specified separately.

## Output Behavior

No API response fields change. After candidate retrieval, the existing `_search_slice()` mapping remains responsible for:

- slice text, score, and rank;
- document identity and title;
- block, section, table, and row positioning;
- `document_handle`, `section_handle`, and `table_handle`;
- `can_get_section`, `can_get_table`, and `can_get_original_text`.

The options `include_refs` and `include_artifact_handles` remain unchanged by this work; their current behavior is not expanded here.

## Error Handling

Existing behavior remains:

- requests without a user identity fail at the router;
- requests without a database session fail in the default retriever;
- permission errors from selected KBs continue to be surfaced through the existing knowledge-base permission path;
- an empty set of retrievable ES candidates returns an empty `slices` list.

No fallback from ES retrieval to IPD RAG is introduced.

## Testing

Add focused tests for `_default_retrieve_documents()` and retain existing slice-projection coverage:

1. Multiple KBs use `get_multi_kb_retrieve_param()` instead of per-KB retrieval configuration.
2. Multiple KBs request grouped vector stores once for the complete KB selection.
3. A compatible analyzer group results in one `manager.retrieve()` invocation across indexes.
4. Multiple analyzer groups are merged and returned by global score order.
5. Duplicate text across groups is returned only once.
6. Requested `top_k` still limits final candidate count.
7. A single KB continues using its existing per-KB vector-store path.
8. Existing tests continue to prove returned candidates are mapped to agent artifact handles.

## Files Expected To Change

- `rag_service/agent_retrieval/service.py`: update only default candidate retrieval orchestration.
- `tests/test_agent_retrieval.py` or the existing focused retrieval coverage file: add regression tests for multi-KB grouping and single-KB compatibility.

No endpoint models or API documentation changes are required because the external contract is unchanged.
