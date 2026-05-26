import json
import sys
import unittest
from datetime import timezone
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()

from rag_service.agent_retrieval.models import SearchSlicesRequest
from rag_service.agent_retrieval import service as svc


class FakeDocument:
    def __init__(self, text, score, metadata=None):
        self.text = text
        self.score = score
        self.metadata = SimpleNamespace(
            source=f"{text}.md",
            extended_metadata=metadata or {},
            retrieve_metadata=SimpleNamespace(title=text, file_address=f"{text}.md"),
        )
        self.es_index = None
        self.es_doc_id = None


def knowledge_base_service_modules(fake_service):
    service_package = ModuleType("rag_service.rag_app.service")
    service_package.knowledge_base_service = fake_service
    return {
        "rag_service.rag_app.service": service_package,
        "rag_service.rag_app.service.knowledge_base_service": fake_service,
    }


class MergeSession:
    def __init__(self):
        self.row = None
        self.committed = False

    def merge(self, row):
        self.row = row

    def commit(self):
        self.committed = True


class AddSession:
    def __init__(self):
        self.row = None
        self.committed = False

    def add(self, row):
        self.row = row

    def commit(self):
        self.committed = True


def fake_request_response_log(**values):
    return SimpleNamespace(**values)


class AgentServiceCoverageTests(unittest.TestCase):
    def test_search_slices_records_success_failure_and_log_writer_failure(self):
        records = []
        low = FakeDocument("low", 0.1)
        high = FakeDocument("high", 0.9)
        service = svc.AgentRetrievalService(
            retrieve_documents=lambda request, uid, session=None: [low, high],
            search_log_writer=lambda record, session: records.append(record),
            request_id_factory=lambda: "request-success",
        )

        response = service.search_slices(SearchSlicesRequest(query="q", top_k=1), uid="uid")

        self.assertEqual(response.request_id, "request-success")
        self.assertEqual([item.text for item in response.slices], ["high"])
        self.assertEqual(json.loads(records[0]["retrieve_result"])[0]["text"], "high")
        self.assertEqual(records[0]["extra_info"]["candidate_count_before_dedup"], 2)

        failure_records = []

        def fail_retrieval(request, uid, session=None):
            raise RuntimeError("backend unavailable")

        failure_service = svc.AgentRetrievalService(
            retrieve_documents=fail_retrieval,
            search_log_writer=lambda record, session: failure_records.append(record),
            request_id_factory=lambda: "request-failure",
        )
        with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
            failure_service.search_slices(SearchSlicesRequest(query="q"), uid="uid")
        self.assertEqual(failure_records[0]["error_reason"], "backend unavailable")
        self.assertEqual(json.loads(failure_records[0]["retrieve_result"]), [])

        def fail_log_write(record, session):
            raise RuntimeError("logging unavailable")

        with self.assertLogs("rag_service.agent_retrieval.service", level="WARNING") as output:
            logged_response = svc.AgentRetrievalService(
                retrieve_documents=lambda request, uid, session=None: [],
                search_log_writer=fail_log_write,
            ).search_slices(SearchSlicesRequest(query="q"), uid="uid")
        self.assertEqual(logged_response.slices, [])
        self.assertIn("log persistence failed", "\n".join(output.output))

    def test_default_service_path_uses_outcome_and_default_callbacks(self):
        default_service = svc.AgentRetrievalService()
        self.assertTrue(default_service.uses_default_retriever)
        self.assertEqual(len(default_service.request_id_factory()), 32)

        expected = svc.RetrievalOutcome(
            documents=[FakeDocument("default", 0.8)],
            diagnostics={"backend_marker": "default"},
        )
        with patch.object(svc, "_default_retrieve_outcome", return_value=expected):
            response = svc.AgentRetrievalService(
                search_log_writer=lambda record, session: None,
                request_id_factory=lambda: "default-request",
            ).search_slices(SearchSlicesRequest(query="q"), uid="uid", session=object())

        self.assertEqual(response.slices[0].text, "default")

    def test_default_retrieval_requires_session_and_handles_no_authorized_kbs(self):
        with self.assertRaisesRegex(RuntimeError, "database session"):
            svc._default_retrieve_documents(SearchSlicesRequest(query="q"), "uid")

        empty_service = SimpleNamespace(permission_judge=lambda session, kb_sns, uid: [])
        with patch.dict(sys.modules, knowledge_base_service_modules(empty_service)):
            empty = svc._default_retrieve_outcome(
                SearchSlicesRequest(query="empty", kb_sn="kb-empty"),
                "uid",
                session=object(),
            )
        self.assertEqual(empty.documents, [])
        self.assertEqual(empty.diagnostics["candidate_count_after_dedup"], 0)

    def test_default_libing_single_kb_returns_highest_score(self):
        kb = SimpleNamespace(sn="kb-one", analyzer="ik")
        single_calls = []
        single_service = SimpleNamespace(
            permission_judge=lambda session, kb_sns, uid: [kb],
            get_retrieve_param_by_kb_config_and_request=lambda knowledge_base, query: SimpleNamespace(
                top_k=query.top_k,
                document_score_threshold=0.0,
                query_strategy="hybrid",
            ),
            get_embedding_model_and_vector_stores=lambda session, sn: "single-store",
            get_vector_store_manager=lambda: SimpleNamespace(
                retrieve=lambda *args, **kwargs: single_calls.append((args, kwargs))
                or [FakeDocument("low", 0.1), FakeDocument("high", 0.9)]
            ),
        )
        with patch.dict(sys.modules, knowledge_base_service_modules(single_service)):
            documents = svc._default_retrieve_documents(
                SearchSlicesRequest(query="q", kb_sn="kb-one", top_k=1),
                "uid",
                session=object(),
            )
        self.assertEqual([document.text for document in documents], ["high"])
        self.assertEqual(single_calls[0][1]["analyzer"], "ik")

    def test_default_libing_grouped_kbs_deduplicate_candidates(self):
        kb_a = SimpleNamespace(sn="kb-a", analyzer="ik")
        kb_b = SimpleNamespace(sn="kb-b", analyzer="standard")
        grouped_service = SimpleNamespace(
            permission_judge=lambda session, kb_sns, uid: [kb_a, kb_b],
            get_multi_kb_retrieve_param=lambda knowledge_bases, query: SimpleNamespace(
                top_k=query.top_k,
                document_score_threshold=0.0,
                query_strategy="hybrid",
            ),
            get_grouped_vector_stores_by_knowledge_base_and_asset=lambda session, kb_map: {
                "ik": "a",
                "standard": "b",
            },
            get_vector_store_manager=lambda: SimpleNamespace(
                retrieve=lambda question, top_k, stores, threshold, **kwargs: (
                    [FakeDocument("same", 0.2), FakeDocument("unique", 0.95)]
                    if stores == "a"
                    else [FakeDocument("same", 0.8)]
                )
            ),
        )
        with patch.dict(sys.modules, knowledge_base_service_modules(grouped_service)):
            grouped = svc._default_retrieve_outcome(
                SearchSlicesRequest(query="q", kb_sn_list=["kb-a", "kb-b"], top_k=2),
                "uid",
                session=object(),
            )
        self.assertEqual([document.text for document in grouped.documents], ["unique", "same"])
        self.assertEqual(grouped.diagnostics["libing_analyzer_group_count"], 2)
        self.assertEqual(grouped.diagnostics["candidate_count_before_dedup"], 3)
        self.assertEqual(grouped.diagnostics["candidate_count_after_dedup"], 2)

    def test_ipd_rerank_filters_mappings_and_updates_score(self):
        mapped = SimpleNamespace(sn="mapped", ipd_rag_kb_id="ipd-1")
        missing = SimpleNamespace(sn="missing", ipd_rag_kb_id=None)
        query_top_k = []
        ipd_service = SimpleNamespace(
            permission_judge=lambda session, kb_sns, uid: [mapped, missing],
            get_multi_kb_retrieve_param=lambda knowledge_bases, query: SimpleNamespace(
                top_k=query.top_k,
                document_score_threshold=0.5,
                query_strategy="hybrid",
                rerank_model="model",
            ),
            retrieve_documents_from_ipd_rag=lambda question, top_k, strategy, knowledge_bases: (
                query_top_k.append(top_k)
                or [FakeDocument("reject", 0.8), FakeDocument("winner", 0.2)]
            ),
            get_rerank_format=lambda document: document.text,
            rerank_embedding=lambda pairs, model: [0.1, 0.95],
        )
        with patch.dict(sys.modules, knowledge_base_service_modules(ipd_service)):
            reranked = svc._default_retrieve_outcome(
                SearchSlicesRequest(
                    query="q",
                    kb_sn_list=["mapped", "missing"],
                    top_k=1,
                    retrieval_backend="ipd",
                    enable_rerank=True,
                ),
                "uid",
                session=object(),
            )
        self.assertEqual(query_top_k, [100])
        self.assertEqual([document.text for document in reranked.documents], ["winner"])
        self.assertEqual(reranked.documents[0].score, 0.95)
        self.assertEqual(reranked.diagnostics["ipd_mapped_kb_sn_list"], ["mapped"])
        self.assertEqual(reranked.diagnostics["ipd_skipped_unmapped_kb_sn_list"], ["missing"])
        self.assertFalse(reranked.diagnostics["rerank_degraded"])

    def test_ipd_without_mapping_returns_no_documents(self):
        missing = SimpleNamespace(sn="missing", ipd_rag_kb_id=None)
        no_mapping_service = SimpleNamespace(
            permission_judge=lambda session, kb_sns, uid: [missing],
            get_retrieve_param_by_kb_config_and_request=lambda knowledge_base, query: SimpleNamespace(
                top_k=query.top_k,
                document_score_threshold=0.0,
                query_strategy="hybrid",
            ),
            retrieve_documents_from_ipd_rag=lambda *args: self.fail("unmapped knowledge base was queried"),
        )
        with patch.dict(sys.modules, knowledge_base_service_modules(no_mapping_service)):
            no_mapping = svc._default_retrieve_outcome(
                SearchSlicesRequest(query="q", kb_sn="missing", retrieval_backend="ipd"),
                "uid",
                session=object(),
            )
        self.assertEqual(no_mapping.documents, [])

    def test_rerank_failure_returns_raw_ranked_document(self):
        degrading_service = SimpleNamespace(
            get_rerank_format=lambda document: document.text,
            rerank_embedding=lambda pairs, model: (_ for _ in ()).throw(RuntimeError("rerank failed")),
        )
        with self.assertLogs("rag_service.agent_retrieval.service", level="WARNING"):
            fallback, degraded = svc._rerank_documents(
                degrading_service,
                "q",
                [FakeDocument("raw-low", 0.1), FakeDocument("raw-high", 0.9)],
                1,
                SimpleNamespace(rerank_model="model", document_score_threshold=0.0),
            )
        self.assertTrue(degraded)
        self.assertEqual([document.text for document in fallback], ["raw-high"])

    def test_search_log_record_and_dump_helpers(self):
        start = svc._utcnow()
        request = SearchSlicesRequest(query="q", kb_sn="kb")
        record = svc._search_log_record(
            "request-id",
            "uid",
            request,
            None,
            {"candidate_count": 0},
            start,
            start,
            svc._utcnow(),
            "failure",
        )
        self.assertEqual(record["error_reason"], "failure")
        svc._default_search_log_writer(record, None)
        self.assertEqual(svc._dump_model(SimpleNamespace(model_dump=lambda: {"new": True})), {"new": True})
        self.assertEqual(svc._dump_model(SimpleNamespace(dict=lambda: {"old": True})), {"old": True})
        self.assertIs(start.tzinfo, timezone.utc)

    def test_default_search_log_writer_supports_merge_and_add_sessions(self):
        record = {"error_reason": "failure"}
        fake_models = ModuleType("rag_service.models.database.models")
        fake_models.RequestResponseLog = fake_request_response_log
        merge_session = MergeSession()
        add_session = AddSession()
        with patch.dict(sys.modules, {"rag_service.models.database.models": fake_models}):
            svc._default_search_log_writer(record, merge_session)
            svc._default_search_log_writer(record, add_session)
        self.assertTrue(merge_session.committed)
        self.assertTrue(add_session.committed)
        self.assertIsNone(merge_session.row.question_id)
        self.assertEqual(add_session.row.error_reason, "failure")

    def test_obs_payload_parsers_cover_valid_scopes_and_invalid_shapes(self):
        self.assertIsNone(svc._structured_artifact_key_parts("doc/not_structured/manifest.json"))
        self.assertIsNone(svc._document_obs_payload("doc/not_structured/manifest.json", ["doc", "not_structured", "manifest.json"]))
        self.assertIsNone(svc._document_obs_payload("doc/structured_docx/bad.json", ["doc", "structured_docx", "bad.json"]))
        document = svc._document_obs_payload(
            "doc/structured_docx/document.md",
            ["doc", "structured_docx", "document.md"],
        )
        self.assertEqual(document["manifest_key"], "doc/structured_docx/manifest.json")

        self.assertIsNone(svc._section_obs_payload("doc/nope/sections/s.md", ["doc", "nope", "sections", "s.md"]))
        self.assertIsNone(svc._section_obs_payload("doc/structured_docx/sections/s.txt", ["doc", "structured_docx", "sections", "s.txt"]))
        section = svc._section_obs_payload(
            "asset/doc/structured_html/sections/s.md",
            ["asset", "doc", "structured_html", "sections", "s.md"],
        )
        self.assertEqual(section["knowledge_base_asset_id"], "asset")

        self.assertIsNone(svc._table_obs_payload("doc/nope/tables/t.html", ["doc", "nope", "tables", "t.html"]))
        self.assertIsNone(svc._table_obs_payload("doc/structured_excel/other/t.html", ["doc", "structured_excel", "other", "t.html"]))
        self.assertIsNone(svc._table_obs_payload("doc/structured_excel/tables/t.txt", ["doc", "structured_excel", "tables", "t.txt"]))
        table = svc._table_obs_payload(
            "asset/doc/artifacts/structured_excel/tables/t.llm.md",
            ["asset", "doc", "artifacts", "structured_excel", "tables", "t.llm.md"],
        )
        self.assertEqual(table["doc_id"], "doc")
        self.assertEqual(table["knowledge_base_asset_id"], "asset")
        self.assertEqual(table["display_ref"], "asset/doc/artifacts/structured_excel/tables/t.html")

        self.assertEqual(svc._artifact_type_index(["doc", "structured_docx", "manifest.json"]), 1)
        self.assertEqual(svc._artifact_type_index(["asset", "doc", "structured_docx", "manifest.json"]), 2)
        self.assertEqual(svc._artifact_type_index(["asset", "doc", "artifacts", "structured_docx", "manifest.json"]), 3)
        self.assertIsNone(svc._obs_payload("doc/structured_docx/manifest.json", "unknown"))


if __name__ == "__main__":
    unittest.main()
