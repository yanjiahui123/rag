# Agent Selectable Global Slice Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `search_slices` with request-selected Libing/IPD retrieval, optional 100-candidate reranking, correlatable retrieval logging, and unchanged `get_related_docs` behavior.

**Architecture:** Keep all new orchestration in `rag_service/agent_retrieval`: the service validates/dispatches a request to one backend, normalizes candidates, performs optional agent-only rerank, projects handles, and logs the final response. It invokes existing `knowledge_base_service` primitives dynamically but does not edit that module or its normal `/kb/get_related_docs` endpoint.

**Tech Stack:** Python, Pydantic request/response models, FastAPI router, SQLModel session/model already used by the application, `unittest` with injected/stub dependencies.

---

### Task 1: API Model And Search Execution Envelope

**Files:**
- Modify: `rag_service/agent_retrieval/models.py`
- Modify: `rag_service/agent_retrieval/service.py`
- Test: `tests/test_agent_retrieval.py`

- [ ] **Step 1: Write failing contract and execution-envelope tests**

Add assertions that the old request defaults remain compatible, a caller may select IPD/rerank, and a direct service call returns a stable request identifier and emits final-slice log data through an injected writer:

```python
def test_search_request_defaults_and_response_request_id(self):
    from rag_service.agent_retrieval.models import SearchSlicesRequest
    from rag_service.agent_retrieval.service import AgentRetrievalService

    req = SearchSlicesRequest(query="risk", kb_sn_list=["kb-1"])
    self.assertEqual(req.retrieval_backend, "libing")
    self.assertFalse(req.enable_rerank)
    response = AgentRetrievalService(
        retrieve_documents=lambda request, uid, session=None: [],
        request_id_factory=lambda: "req-1",
    ).search_slices(req, uid="user-1")
    self.assertEqual(response.request_id, "req-1")

def test_search_slices_logs_returned_slice_payload(self):
    records = []
    service = AgentRetrievalService(
        retrieve_documents=lambda request, uid, session=None: [FakeRetrievedDocument("hit", "a.md", 0.8)],
        search_log_writer=lambda record, session: records.append(record),
        request_id_factory=lambda: "req-log",
    )
    response = service.search_slices(SearchSlicesRequest(query="q", kb_sn="kb"), uid="u", session=object())
    self.assertEqual(records[0]["request_id"], response.request_id)
    self.assertEqual(json.loads(records[0]["retrieve_result"])[0]["text"], "hit")
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python -m unittest tests.test_agent_retrieval.AgentRetrievalTests.test_search_request_defaults_and_response_request_id tests.test_agent_retrieval.AgentRetrievalTests.test_search_slices_logs_returned_slice_payload -v`

Expected: FAIL because the new request fields, response `request_id`, and injectable log/request-id hooks do not exist.

- [ ] **Step 3: Implement the request/response fields and service envelope**

Add model fields:

```python
class SearchSlicesRequest(AgentRetrievalRequest):
    query: str
    kb_sn: Optional[str] = None
    kb_sn_list: List[str] = Field(default_factory=list)
    top_k: int = 20
    include_refs: bool = True
    include_artifact_handles: bool = True
    retrieval_backend: Literal["libing", "ipd"] = "libing"
    enable_rerank: bool = False

class SearchSlicesResponse(BaseModel):
    request_id: str
    query: str
    kb_sn_list: List[str]
    slices: List[SearchSlice] = Field(default_factory=list)
```

Extend `AgentRetrievalService.__init__` with optional `search_log_writer` and `request_id_factory`, generate a request id with `uuid.uuid4().hex` by default, build response slices inside `try/except`, and emit a record containing `retrieve_result` JSON and `extra_info` on both success and failure. Leave injected `retrieve_documents` support intact so current handle-projection tests remain valid.

- [ ] **Step 4: Run the focused tests and existing handle projection test**

Run: `python -m unittest tests.test_agent_retrieval.AgentRetrievalTests.test_search_request_defaults_and_response_request_id tests.test_agent_retrieval.AgentRetrievalTests.test_search_slices_logs_returned_slice_payload tests.test_agent_retrieval.AgentRetrievalTests.test_search_slices_projects_documents_with_action_handles -v`

Expected: PASS.

### Task 2: Libing Backend Global Candidate Retrieval

**Files:**
- Modify: `rag_service/agent_retrieval/service.py`
- Test: `tests/test_agent_retrieval.py`
- Test: `tests/test_requested_line_coverage.py`

- [ ] **Step 1: Write failing Libing backend tests**

Use a stub `rag_service.rag_app.service.knowledge_base_service` module, as existing requested-line tests already do, to assert:

```python
def test_default_libing_multi_kb_searches_grouped_indexes_and_deduplicates(self):
    grouped_requests = []
    manager_calls = []
    kb_a = SimpleNamespace(sn="kb-a", analyzer="ik")
    kb_b = SimpleNamespace(sn="kb-b", analyzer="ik")
    duplicate_low = FakeRetrievedDocument("duplicate", "low.md", 0.2)
    duplicate_high = FakeRetrievedDocument("duplicate", "high.md", 0.7)
    global_high = FakeRetrievedDocument("global-high", "top.md", 0.9)
    manager = SimpleNamespace(
        retrieve=lambda *args, **kwargs: manager_calls.append((args, kwargs))
        or ([duplicate_low, global_high] if len(manager_calls) == 1 else [duplicate_high])
    )
    fake_service = SimpleNamespace(
        permission_judge=lambda session, kb_sns, uid: [kb_a, kb_b],
        get_multi_kb_retrieve_param=lambda kbs, req: SimpleNamespace(
            top_k=req.top_k, document_score_threshold=0.0, query_strategy="hybrid", rerank_model="rerank"
        ),
        get_grouped_vector_stores_by_knowledge_base_and_asset=lambda session, kb_map: (
            grouped_requests.append(kb_map) or {"ik": "stores-a", "standard": "stores-b"}
        ),
        get_vector_store_manager=lambda: manager,
    )
    service_pkg = ModuleType("rag_service.rag_app.service")
    service_pkg.knowledge_base_service = fake_service
    with patch.dict(
        sys.modules,
        {
            "rag_service.rag_app.service": service_pkg,
            "rag_service.rag_app.service.knowledge_base_service": fake_service,
        },
    ):
        outcome = svc._default_retrieve_outcome(
            SearchSlicesRequest(query="q", kb_sn_list=["kb-a", "kb-b"]),
            "uid",
            session=object(),
        )
    self.assertEqual(grouped_requests, [{"kb-a": [], "kb-b": []}])
    self.assertEqual(len(manager_calls), 2)
    self.assertEqual([doc.text for doc in outcome.documents], ["global-high", "duplicate"])
    self.assertEqual(outcome.diagnostics["libing_analyzer_group_count"], 2)
```

Retain and update the current single-KB stub test to verify a legacy Libing request calls `get_embedding_model_and_vector_stores()` and does not require grouped lookup.

- [ ] **Step 2: Run Libing tests to verify failure**

Run: `python -m unittest tests.test_agent_retrieval.AgentRetrievalTests.test_default_libing_multi_kb_searches_grouped_indexes_and_deduplicates tests.test_requested_line_coverage.RequestedAgentRetrievalLineCoverageTests.test_default_retrieval_uses_injected_service_module -v`

Expected: FAIL because `_default_retrieve_documents` currently performs one manager call per KB and exposes no diagnostics.

- [ ] **Step 3: Implement agent-only Libing orchestration**

Introduce an internal outcome type and helper functions in `service.py`:

```python
@dataclass
class RetrievalOutcome:
    documents: List[Any]
    diagnostics: Dict[str, Any]

def _candidate_top_k(request: SearchSlicesRequest) -> int:
    return 100 if request.enable_rerank else _bounded_top_k(request.top_k)

def _deduplicate_documents(documents: Iterable[Any]) -> List[Any]:
    seen = set()
    unique = []
    for document in documents:
        text = getattr(document, "text", "")
        if text not in seen:
            unique.append(document)
            seen.add(text)
    return unique
```

Implement `_default_retrieve_outcome()` to call `permission_judge()`, keep the existing single-KB manager call, and for multiple Libing KBs call `get_multi_kb_retrieve_param()` plus `get_grouped_vector_stores_by_knowledge_base_and_asset(session, {kb.sn: [] for kb in knowledge_bases})`. Query each analyzer group once, deduplicate after merging, sort raw results, and retain `_default_retrieve_documents()` as a list-returning compatibility wrapper.

- [ ] **Step 4: Run focused Libing tests**

Run: `python -m unittest tests.test_agent_retrieval tests.test_requested_line_coverage.RequestedAgentRetrievalLineCoverageTests.test_default_retrieval_uses_injected_service_module -v`

Expected: PASS for the Libing cases and existing agent handle tests.

### Task 3: IPD Backend And Explicit Reranking

**Files:**
- Modify: `rag_service/agent_retrieval/service.py`
- Test: `tests/test_agent_retrieval.py`

- [ ] **Step 1: Write failing IPD and rerank tests**

Add stub-driven tests that require exclusive IPD dispatch and fixed rerank candidate expansion:

```python
def test_ipd_backend_skips_unmapped_kbs_without_libing_fallback(self):
    request = SearchSlicesRequest(query="q", kb_sn_list=["mapped", "missing"], retrieval_backend="ipd")
    outcome = svc._default_retrieve_outcome(request, "uid", session=object())
    self.assertEqual(ipd_calls[0]["knowledge_bases"], [mapped_kb])
    self.assertEqual(outcome.diagnostics["ipd_skipped_unmapped_kb_sn_list"], ["missing"])
    self.assertEqual(manager_calls, [])

def test_rerank_fetches_fixed_hundred_candidates_and_limits_final_results(self):
    request = SearchSlicesRequest(query="q", kb_sn="kb", top_k=3, enable_rerank=True)
    response = service.search_slices(request, uid="uid", session=object())
    self.assertEqual(retrieved_top_ks, [100])
    self.assertEqual(rerank_top_ks, [3])
    self.assertEqual(len(response.slices), 3)
```

Also assert that a rerank exception returns backend-ranked candidates and records `rerank_degraded=True`.

- [ ] **Step 2: Run IPD/rerank tests to verify failure**

Run: `python -m unittest tests.test_agent_retrieval.AgentRetrievalTests.test_ipd_backend_skips_unmapped_kbs_without_libing_fallback tests.test_agent_retrieval.AgentRetrievalTests.test_rerank_fetches_fixed_hundred_candidates_and_limits_final_results tests.test_agent_retrieval.AgentRetrievalTests.test_rerank_failure_is_logged_as_degraded -v`

Expected: FAIL because backend selection and explicit agent reranking are not implemented.

- [ ] **Step 3: Implement IPD dispatch and isolated rerank adapter**

In `_default_retrieve_outcome()`, use `request.retrieval_backend` to select exactly one path. For IPD, form:

```python
mapped = [kb for kb in knowledge_bases if getattr(kb, "ipd_rag_kb_id", None)]
documents = knowledge_base_service.retrieve_documents_from_ipd_rag(
    query_request.question,
    candidate_top_k,
    retrieve_config.query_strategy,
    mapped,
) if mapped else []
```

Record mapped/skipped KB SN lists. Implement an agent-local rerank adapter that calls the knowledge-base module's `rerank_embedding` and `get_rerank_format`, catches its own failure to set `rerank_degraded`, and never changes `knowledge_base_service.rerank_retrieved_documents()` used by `get_related_docs`.

- [ ] **Step 4: Run backend and rerank tests**

Run: `python -m unittest tests.test_agent_retrieval -v`

Expected: PASS.

### Task 4: Request ID Routing And Database Log Persistence

**Files:**
- Modify: `rag_service/agent_retrieval/router.py`
- Modify: `rag_service/agent_retrieval/service.py`
- Test: `tests/test_agent_retrieval.py`

- [ ] **Step 1: Write failing router and default log-writer tests**

Add tests for forwarding an incoming header id and writing the final slice JSON into a fake session:

```python
def test_search_router_forwards_request_id_header(self):
    request = SimpleNamespace(state=SimpleNamespace(uid="user"), headers={"X-Request-ID": "trace-1"})
    with patch.object(router_module, "_service", return_value=fake_service):
        router_module.search_slices(request, SearchSlicesRequest(query="q"), session=object())
    self.assertEqual(fake_service.request_id, "trace-1")

def test_default_log_writer_merges_request_response_log(self):
    session = FakeSession()
    record = {
        "request_id": "trace-1",
        "user_id": "user",
        "method_name": "agent.retrieval.search_slices",
        "kb_sn": "kb",
        "question": "q",
        "request_start_time": None,
        "request_end_time": None,
        "request_time_use": 0.0,
        "retrieve_start_time": None,
        "retrieve_end_time": None,
        "retrieve_time_use": 0.0,
        "retrieve_result": '[{"text": "hit"}]',
        "error_reason": None,
        "extra_info": {"retrieval_backend": "libing"},
    }
    svc._default_search_log_writer(record, session)
    self.assertEqual(session.merged.retrieve_result, '[{"text": "hit"}]')
    self.assertTrue(session.committed)
```

- [ ] **Step 2: Run routing/log persistence tests to verify failure**

Run: `python -m unittest tests.test_agent_retrieval.AgentRetrievalTests.test_search_router_forwards_request_id_header tests.test_agent_retrieval.AgentRetrievalTests.test_default_log_writer_merges_request_response_log -v`

Expected: FAIL because router request-id forwarding and DB log writer are absent.

- [ ] **Step 3: Implement request-id forwarding and narrow log writer**

In `router.py`, add `_request_id(request)` that reads `X-Request-ID` case-insensitively or returns `None`, then pass `request_id=_request_id(request)` into `service.search_slices()`.

In `service.py`, implement `_default_search_log_writer(record, session)` by importing `RequestResponseLog` only when a session is present, constructing the row fields defined by the model, and using `session.merge(row)` followed by `session.commit()`. Catch log persistence failures in `search_slices` with module logging so logging cannot replace a successful retrieval response with an unrelated database exception.

- [ ] **Step 4: Run agent tests**

Run: `python -m unittest tests.test_agent_retrieval -v`

Expected: PASS.

### Task 5: Generated Agent Skill Request Parameters

**Files:**
- Modify: `rag_service/agent_retrieval/skill_generator.py`
- Test: `tests/test_agent_retrieval.py`

- [ ] **Step 1: Write failing generated-client tests**

Extend package and request-runner tests:

```python
self.assertEqual(json.loads(package.files["request.json"])["retrieval_backend"], "libing")
self.assertFalse(json.loads(package.files["request.json"])["enable_rerank"])
self.assertEqual(captured["payload"]["retrieval_backend"], "ipd")
self.assertTrue(captured["payload"]["enable_rerank"])
```

- [ ] **Step 2: Run generated skill tests to verify failure**

Run: `python -m unittest tests.test_agent_retrieval.AgentRetrievalTests.test_skill_package_has_opencode_files_and_no_token_requirement tests.test_agent_retrieval.AgentRetrievalTests.test_generated_request_runner_dispatches_unicode_query_without_shell_args -v`

Expected: FAIL because generated request and client payload do not expose backend/rerank fields.

- [ ] **Step 3: Extend generated request/schema/client**

Add defaults to `request.json`; document the fields in `SKILL_TEMPLATE` and `API_SCHEMA`; add `--retrieval-backend` and `--enable-rerank` handling in the generated client and request runner so the search payload includes:

```python
{
    "uid": config.get("uid"),
    "query": args.query,
    "kb_sn_list": config.get("kb_sn_list", []),
    "top_k": args.top_k,
    "retrieval_backend": args.retrieval_backend,
    "enable_rerank": args.enable_rerank,
}
```

- [ ] **Step 4: Run generated skill tests**

Run: `python -m unittest tests.test_agent_retrieval -v`

Expected: PASS.

### Task 6: Regression Verification And Change Isolation

**Files:**
- Verify only: `rag_service/rag_app/service/knowledge_base_service.py`
- Verify only: `rag_service/rag_app/router/knowledge_base_api.py`
- Test: `tests/test_requested_line_coverage.py`

- [ ] **Step 1: Confirm normal retrieval source files remain unmodified**

Run: `git diff --name-only -- rag_service/rag_app/service/knowledge_base_service.py rag_service/rag_app/router/knowledge_base_api.py`

Expected: no output. This is the direct source-isolation guarantee for `get_related_docs`.

- [ ] **Step 2: Run focused retrieval regression tests**

Run: `python -m unittest tests.test_agent_retrieval tests.test_requested_line_coverage -v`

Expected: PASS.

- [ ] **Step 3: Run available full test discovery**

Run: `python -m unittest discover -s tests -v`

Expected: PASS, or record pre-existing/import-environment failures separately from agent retrieval failures.

- [ ] **Step 4: Inspect the final diff and commit only feature files**

Run: `git diff --stat -- rag_service/agent_retrieval tests/test_agent_retrieval.py tests/test_requested_line_coverage.py docs/superpowers/plans/2026-05-25-agent-selectable-global-slice-retrieval-implementation.md`

Expected: modifications are confined to the agent retrieval feature, tests, and this plan; normal `get_related_docs` code is absent.
