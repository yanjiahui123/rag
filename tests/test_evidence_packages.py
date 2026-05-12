import unittest
import threading
import ast
import json
import posixpath
from pathlib import Path

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()


class FakeRetrieveMetadata:
    def __init__(self, title="", file_address=""):
        self.title = title
        self.file_address = file_address


class FakeMetadata:
    def __init__(self, source, extended_metadata=None, retrieve_metadata=None, mtime="2026-05-08"):
        self.source = source
        self.extended_metadata = extended_metadata or {}
        self.retrieve_metadata = retrieve_metadata
        self.mtime = mtime


class FakeRetrievedDocument:
    def __init__(self, text, source, score, extended_metadata=None, es_index=None, es_doc_id=None):
        self.text = text
        self.score = score
        self.metadata = FakeMetadata(
            source,
            extended_metadata,
            FakeRetrieveMetadata(
                title=(extended_metadata or {}).get("title", ""),
                file_address=source,
            ),
        )
        self.es_index = es_index
        self.es_doc_id = es_doc_id


class EvidencePackageTests(unittest.TestCase):
    def test_evidence_package_models_use_pydantic_base_model(self):
        from pydantic import BaseModel
        from rag_service.retrieval import evidence_packages

        model_types = [
            evidence_packages.EvidencePackageOptions,
            evidence_packages.DocumentArtifact,
            evidence_packages.EvidenceItem,
            evidence_packages.DocumentSectionArtifact,
            evidence_packages.TableArtifact,
            evidence_packages.PackageArtifacts,
            evidence_packages.DocumentEvidencePackage,
            evidence_packages.EvidencePackageResponse,
        ]

        for model_type in model_types:
            self.assertTrue(issubclass(model_type, BaseModel), model_type.__name__)

    def test_evidence_packages_module_does_not_use_dataclasses(self):
        tree = ast.parse(Path("rag_service/retrieval/evidence_packages.py").read_text(encoding="utf-8"))
        imported_modules = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        dataclass_decorators = [
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == "dataclass"
        ]

        self.assertNotIn("dataclasses", imported_modules)
        self.assertEqual(dataclass_decorators, [])

    def test_evidence_api_keeps_request_parameters_minimal(self):
        router_tree = ast.parse(Path("rag_service/rag_app/router/knowledge_base_api.py").read_text(encoding="utf-8"))
        service_source = Path("rag_service/rag_app/service/knowledge_base_service.py").read_text(encoding="utf-8")
        request_fields = _class_annotation_names(router_tree, "EvidencePackageRequest")

        redundant_fields = {
            "package_top_k",
            "max_evidence_per_package",
            "candidate_multiplier",
            "include_online_qa",
            "enable_table_expansion",
            "table_expand_ratio_threshold",
            "max_full_table_rows",
            "max_inline_table_chars",
        }
        self.assertEqual(set(request_fields), set())
        self.assertTrue(redundant_fields.isdisjoint(request_fields))
        for field_name in redundant_fields:
            self.assertNotIn(f'getattr(req, "{field_name}"', service_source)

    def test_answer_evidence_endpoint_uses_api_presenter(self):
        router_source = Path("rag_service/rag_app/router/knowledge_base_api.py").read_text(encoding="utf-8")
        service_source = Path("rag_service/rag_app/service/knowledge_base_service.py").read_text(encoding="utf-8")

        self.assertIn('"/get_answer_evidence"', router_source)
        self.assertIn("def get_answer_evidence(", router_source)
        self.assertIn("knowledge_base_service.get_answer_evidence", router_source)
        self.assertIn("def get_answer_evidence(", service_source)
        self.assertIn("get_evidence_packages(req, background_tasks, session)", service_source)
        self.assertIn("to_answer_evidence_response", service_source)
        self.assertIn('"/evidence_artifacts/{artifact_id}"', router_source)
        self.assertIn("knowledge_base_service.get_evidence_artifact", router_source)
        self.assertIn("def get_evidence_artifact(", service_source)
        self.assertIn("download_file_as_bytes", service_source)
        self.assertIn("artifact_url_builder=lambda ref: _evidence_artifact_url(ref, req.uid)", service_source)
        self.assertIn('"object_key"', service_source)
        self.assertIn('"uid"', service_source)
        self.assertIn("OperationNotPermittedException", service_source)
        self.assertIn("def get_evidence_artifact(request: Request, artifact_id: str)", router_source)
        self.assertIn("knowledge_base_service.get_evidence_artifact(artifact_id, request.state.uid", router_source)

    def test_groups_candidates_by_document_and_resolves_artifact_refs(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        response = build_evidence_packages(
            query="q",
            rewrite_query="rewritten q",
            documents=_grouped_artifact_candidates(),
            options=EvidencePackageOptions(
                package_top_k=2,
                max_evidence_per_package=2,
                artifact_mode="signed_url",
            ),
            artifact_url_resolver=_signed_artifact_url,
        )

        _assert_grouped_artifact_response(self, response)

    def test_returns_key_refs_without_downloading_by_default_and_bounds_packages(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        candidates = [
            FakeRetrievedDocument(
                "one",
                "a.md",
                0.8,
                {
                    "doc_id": "doc-a",
                    "block_id": "block_001",
                    "table_id": "table_001",
                    "display_ref": "doc-a/structured_markdown/tables/table.html",
                },
            ),
            FakeRetrievedDocument("two", "b.md", 0.7, {"doc_id": "doc-b", "block_id": "block_001"}),
        ]

        response = build_evidence_packages(
            query="q",
            rewrite_query="q",
            documents=candidates,
            options=EvidencePackageOptions(package_top_k=1),
            artifact_url_resolver=lambda key: self.fail("resolver should not be called in key mode"),
        )

        self.assertEqual(len(response.packages), 1)
        self.assertEqual(response.packages[0].doc_id, "doc-a")
        self.assertEqual(
            response.packages[0].artifacts.tables[0].refs["display"],
            "doc-a/structured_markdown/tables/table.html",
        )

    def test_answer_evidence_response_splits_llm_context_from_display_sources(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
            to_answer_evidence_response,
        )

        response = build_evidence_packages(
            query=_ANSWER_QUERY,
            rewrite_query=_ANSWER_REWRITE_QUERY,
            documents=_answer_evidence_documents(),
            options=EvidencePackageOptions(package_top_k=1, max_evidence_per_package=3),
        )

        api_response = to_answer_evidence_response(
            response,
            artifact_url_builder=_evidence_artifact_test_url,
        )

        _assert_answer_evidence_response(self, api_response)

    def test_agent_evidence_response_exposes_original_candidate_slices(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
            to_agent_evidence_response,
        )

        response = build_evidence_packages(
            query="q",
            rewrite_query="rewritten q",
            documents=_agent_evidence_documents(),
            options=EvidencePackageOptions(package_top_k=1, max_evidence_per_package=1),
        )

        agent_response = to_agent_evidence_response(response)

        _assert_agent_evidence_response(self, agent_response)

    def test_preserves_distinct_table_chunks_from_same_table(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        candidates = [
            FakeRetrievedDocument(
                "row one",
                "sales.xlsx",
                0.93,
                {"doc_id": "doc-a", "block_id": "table_001_row_001", "table_id": "table_001"},
            ),
            FakeRetrievedDocument(
                "row two",
                "sales.xlsx",
                0.91,
                {"doc_id": "doc-a", "block_id": "table_001_row_002", "table_id": "table_001"},
            ),
            FakeRetrievedDocument(
                "row three",
                "sales.xlsx",
                0.89,
                {"doc_id": "doc-a", "block_id": "table_001_row_003", "table_id": "table_001"},
            ),
        ]

        response = build_evidence_packages(
            query="table name",
            rewrite_query="table name",
            documents=candidates,
            options=EvidencePackageOptions(package_top_k=1, max_evidence_per_package=3),
        )

        evidence = response.packages[0].evidence
        self.assertEqual([item.text for item in evidence], ["row one", "row two", "row three"])
        self.assertEqual([item.table_id for item in evidence], ["table_001", "table_001", "table_001"])

    def test_merges_related_document_blocks_from_same_section(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        response = build_evidence_packages(
            query="sales",
            rewrite_query="sales",
            documents=_related_section_candidates(),
            options=EvidencePackageOptions(package_top_k=1, max_evidence_per_package=2),
        )

        _assert_related_section_package(self, response.packages[0])

    def test_merged_document_section_uses_section_artifact_text_when_available(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        candidates = [
            FakeRetrievedDocument(
                "partial intro",
                "report.md",
                0.91,
                {
                    "doc_id": "doc-a",
                    "block_id": "block_001",
                    "block_type": "text",
                    "section_id": "section_001",
                    "section_ref": "doc-a/structured_markdown/sections/section_001.md",
                    "headers": ["Report", "Sales"],
                    "title": "Sales",
                },
            ),
            FakeRetrievedDocument(
                "partial detail",
                "report.md",
                0.9,
                {
                    "doc_id": "doc-a",
                    "block_id": "block_002",
                    "block_type": "text",
                    "section_id": "section_001",
                    "section_ref": "doc-a/structured_markdown/sections/section_001.md",
                    "headers": ["Report", "Sales"],
                    "title": "Sales",
                },
            ),
        ]

        response = build_evidence_packages(
            query="sales",
            rewrite_query="sales",
            documents=candidates,
            options=EvidencePackageOptions(package_top_k=1, max_evidence_per_package=1),
            artifact_text_resolver=lambda key: "# Sales\n\ncomplete section from obs",
        )

        package = response.packages[0]
        self.assertEqual(
            package.artifacts.sections[0].refs["section"],
            "doc-a/structured_markdown/sections/section_001.md",
        )
        self.assertEqual(package.evidence[0].text, "# Sales\n\ncomplete section from obs")

    def test_manifest_coverage_controls_full_section_expansion_and_logs_trace(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        structured_markdown = {
            "artifact_prefix": "doc-a/structured_markdown/",
            "document_markdown_key": "doc-a/structured_markdown/document.md",
            "manifest_key": "doc-a/structured_markdown/manifest.json",
            "block_count": 4,
        }
        manifest = {
            "sections": [
                {
                    "section_id": "section_001",
                    "title": "Sales",
                    "headers": ["Report", "Sales"],
                    "block_ids": ["block_001", "block_002", "block_003", "block_004"],
                    "section_ref": "doc-a/structured_markdown/sections/section_001.md",
                }
            ]
        }
        candidates = [
            FakeRetrievedDocument(
                f"partial {index}",
                "report.md",
                0.91 - index * 0.01,
                {
                    "doc_id": "doc-a",
                    "block_id": f"block_{index + 1:03d}",
                    "block_type": "text",
                    "section_id": "section_001",
                    "headers": ["Report", "Sales"],
                    "title": "Sales",
                    "structured_markdown": structured_markdown,
                },
            )
            for index in range(3)
        ]

        def resolve_text(key):
            if key.endswith("manifest.json"):
                return json.dumps(manifest)
            if key.endswith("section_001.md"):
                return "# Sales\n\nfull section from manifest"
            return ""

        with self.assertLogs("rag_service.retrieval.evidence_packages", level="DEBUG") as logs:
            response = build_evidence_packages(
                query="sales",
                rewrite_query="sales",
                documents=candidates,
                options=EvidencePackageOptions(package_top_k=1, max_evidence_per_package=1),
                artifact_text_resolver=resolve_text,
            )

        section = response.packages[0].artifacts.sections[0]
        self.assertEqual(section.mode, "full_section")
        self.assertEqual(section.refs["section"], "doc-a/structured_markdown/sections/section_001.md")
        self.assertEqual(response.packages[0].evidence[0].text, "# Sales\n\nfull section from manifest")
        self.assertIn("coverage=0.750000", "\n".join(logs.output))
        self.assertIn("mode=full_section", "\n".join(logs.output))

    def test_manifest_low_coverage_keeps_merged_section_chunks(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        structured_markdown = {
            "artifact_prefix": "doc-a/structured_markdown/",
            "manifest_key": "doc-a/structured_markdown/manifest.json",
        }
        manifest = {
            "sections": [
                {
                    "section_id": "section_001",
                    "block_ids": ["block_001", "block_002", "block_003", "block_004"],
                    "section_ref": "doc-a/structured_markdown/sections/section_001.md",
                }
            ]
        }
        candidates = [
            FakeRetrievedDocument(
                text,
                "report.md",
                score,
                {
                    "doc_id": "doc-a",
                    "block_id": block_id,
                    "block_type": "text",
                    "section_id": "section_001",
                    "structured_markdown": structured_markdown,
                },
            )
            for text, score, block_id in [("partial one", 0.91, "block_001"), ("partial two", 0.9, "block_002")]
        ]
        fetched_keys = []

        def resolve_text(key):
            fetched_keys.append(key)
            if key.endswith("manifest.json"):
                return json.dumps(manifest)
            return "should not fetch full section"

        response = build_evidence_packages(
            query="sales",
            rewrite_query="sales",
            documents=candidates,
            options=EvidencePackageOptions(package_top_k=1, max_evidence_per_package=1),
            artifact_text_resolver=resolve_text,
        )

        section = response.packages[0].artifacts.sections[0]
        self.assertEqual(section.mode, "merged_section")
        self.assertEqual(response.packages[0].evidence[0].text, "partial one\n\npartial two")
        self.assertEqual(fetched_keys, ["doc-a/structured_markdown/manifest.json"])

    def test_keeps_distinct_text_chunks_with_same_block_id_for_section_merge(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        candidates = [
            FakeRetrievedDocument(
                "first split",
                "report.md",
                0.91,
                {
                    "doc_id": "doc-a",
                    "block_id": "block_001",
                    "block_index": 1,
                    "block_type": "text",
                    "headers": ["Report"],
                    "title": "Report",
                },
            ),
            FakeRetrievedDocument(
                "second split",
                "report.md",
                0.9,
                {
                    "doc_id": "doc-a",
                    "block_id": "block_001",
                    "block_index": 1,
                    "block_type": "text",
                    "headers": ["Report"],
                    "title": "Report",
                },
            ),
        ]

        response = build_evidence_packages(
            query="report",
            rewrite_query="report",
            documents=candidates,
            options=EvidencePackageOptions(package_top_k=1, max_evidence_per_package=1),
        )

        evidence = response.packages[0].evidence[0]
        self.assertEqual(evidence.evidence_type, "section")
        self.assertEqual(evidence.text, "first split\n\nsecond split")
        self.assertEqual(evidence.hit_blocks, 2)
        self.assertEqual(evidence.merged_block_ids, ["block_001"])

    def test_promotes_covered_table_chunks_to_single_inline_table_evidence(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        response = build_evidence_packages(
            query="sales detail",
            rewrite_query="sales detail",
            documents=_covered_table_chunk_candidates(),
            options=EvidencePackageOptions(
                package_top_k=1,
                max_evidence_per_package=2,
                table_expand_ratio_threshold=0.4,
                max_inline_table_chars=100,
            ),
            artifact_text_resolver=_covered_table_text,
        )

        package = response.packages[0]
        _assert_inline_table_evidence(self, package.evidence)
        _assert_inline_table_artifact(self, package.artifacts.tables[0])

    def test_does_not_expand_table_when_hit_ratio_is_below_threshold(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        table_summary = {"table_id": "table_001", "row_count": 10}
        candidates = [
            FakeRetrievedDocument(
                f"row chunk {index}",
                "sales.xlsx",
                0.9,
                {
                    "doc_id": "doc-a",
                    "block_id": f"table_001_chunk_{index + 1:03d}",
                    "table_id": "table_001",
                    "row_range": (index, index + 1),
                    "table": table_summary,
                },
            )
            for index in range(2)
        ]

        response = build_evidence_packages(
            query="sales detail",
            rewrite_query="sales detail",
            documents=candidates,
            options=EvidencePackageOptions(
                package_top_k=1,
                max_evidence_per_package=2,
                table_expand_ratio_threshold=0.5,
            ),
        )

        self.assertEqual(len(response.packages[0].evidence), 2)
        self.assertEqual(len(response.packages[0].artifacts.tables), 1)
        self.assertFalse(response.packages[0].artifacts.tables[0].expanded)
        self.assertEqual(response.packages[0].artifacts.tables[0].hit_row_ranges, [[0, 2]])
        self.assertEqual(response.packages[0].artifacts.tables[0].hit_ratio, 0.2)

    def test_chunked_table_without_row_ranges_does_not_expand_as_full_table(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        table_summary = {"table_id": "table_001", "row_count": 10}
        candidates = [
            FakeRetrievedDocument(
                "chunk one",
                "sales.xlsx",
                0.9,
                {
                    "doc_id": "doc-a",
                    "block_id": "table_001_chunk_001",
                    "block_type": "table",
                    "table_id": "table_001",
                    "table": table_summary,
                },
            ),
            FakeRetrievedDocument(
                "chunk two",
                "sales.xlsx",
                0.8,
                {
                    "doc_id": "doc-a",
                    "block_id": "table_001_chunk_002",
                    "block_type": "table",
                    "table_id": "table_001",
                    "table": table_summary,
                },
            ),
        ]

        response = build_evidence_packages(
            query="sales detail",
            rewrite_query="sales detail",
            documents=candidates,
            options=EvidencePackageOptions(package_top_k=1, table_expand_ratio_threshold=0.7),
        )

        table_artifact = response.packages[0].artifacts.tables[0]
        self.assertFalse(table_artifact.expanded)
        self.assertEqual(table_artifact.hit_row_ranges, [])
        self.assertEqual(table_artifact.hit_rows, 0)
        self.assertEqual(table_artifact.hit_ratio, 0)

    def test_table_expansion_uses_merged_row_ranges_and_can_degrade_for_row_cap(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        table_summary = {"table_id": "table_001", "row_count": 10}
        candidates = [
            FakeRetrievedDocument(
                "rows 1 to 4",
                "sales.xlsx",
                0.9,
                {
                    "doc_id": "doc-a",
                    "block_id": "table_001_chunk_001",
                    "table_id": "table_001",
                    "row_range": (0, 4),
                    "table": table_summary,
                    "display_ref": "doc-a/structured_excel/tables/table_001.html",
                },
            ),
            FakeRetrievedDocument(
                "rows 3 to 6",
                "sales.xlsx",
                0.8,
                {
                    "doc_id": "doc-a",
                    "block_id": "table_001_chunk_002",
                    "table_id": "table_001",
                    "row_range": (2, 6),
                    "table": table_summary,
                    "display_ref": "doc-a/structured_excel/tables/table_001.html",
                },
            ),
        ]

        response = build_evidence_packages(
            query="sales detail",
            rewrite_query="sales detail",
            documents=candidates,
            options=EvidencePackageOptions(
                package_top_k=1,
                max_evidence_per_package=2,
                table_expand_ratio_threshold=0.5,
                max_full_table_rows=3,
            ),
        )

        table_artifact = response.packages[0].artifacts.tables[0]
        self.assertEqual(table_artifact.hit_row_ranges, [[0, 6]])
        self.assertEqual(table_artifact.hit_rows, 6)
        self.assertEqual(table_artifact.hit_ratio, 0.6)
        self.assertEqual(table_artifact.mode, "table_reference")
        self.assertEqual(response.packages[0].evidence[0].mode, "table_reference")
        self.assertEqual(response.packages[0].evidence[0].hit_row_ranges, [[0, 6]])
        self.assertIn("rows 1 to 4", response.packages[0].evidence[0].text)

    def test_table_artifact_reports_disjoint_hit_row_ranges(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        table_summary = {"table_id": "table_001", "row_count": 10}
        candidates = [
            FakeRetrievedDocument(
                "first data row",
                "sales.xlsx",
                0.9,
                {
                    "doc_id": "doc-a",
                    "block_id": "table_001_chunk_001",
                    "table_id": "table_001",
                    "row_range": (0, 1),
                    "table": table_summary,
                },
            ),
            FakeRetrievedDocument(
                "sixth data row",
                "sales.xlsx",
                0.8,
                {
                    "doc_id": "doc-a",
                    "block_id": "table_001_chunk_006",
                    "table_id": "table_001",
                    "row_range": (5, 6),
                    "table": table_summary,
                },
            ),
        ]

        response = build_evidence_packages(
            query="sales detail",
            rewrite_query="sales detail",
            documents=candidates,
            options=EvidencePackageOptions(package_top_k=1, table_expand_ratio_threshold=0.7),
        )

        table_artifact = response.packages[0].artifacts.tables[0]
        self.assertEqual(table_artifact.hit_row_ranges, [[0, 1], [5, 6]])
        self.assertEqual(table_artifact.hit_rows, 2)

    def test_covered_table_degrades_when_inline_text_is_too_large(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        table_summary = {"table_id": "table_001", "row_count": 2}
        candidates = [
            FakeRetrievedDocument(
                "row one",
                "sales.xlsx",
                0.9,
                {
                    "doc_id": "doc-a",
                    "block_id": "table_001_chunk_001",
                    "table_id": "table_001",
                    "row_range": (0, 1),
                    "table": table_summary,
                    "llm_table_ref": "doc-a/structured_excel/tables/table_001.llm.md",
                },
            ),
            FakeRetrievedDocument(
                "row two",
                "sales.xlsx",
                0.8,
                {
                    "doc_id": "doc-a",
                    "block_id": "table_001_chunk_002",
                    "table_id": "table_001",
                    "row_range": (1, 2),
                    "table": table_summary,
                    "llm_table_ref": "doc-a/structured_excel/tables/table_001.llm.md",
                },
            ),
        ]

        response = build_evidence_packages(
            query="sales detail",
            rewrite_query="sales detail",
            documents=candidates,
            options=EvidencePackageOptions(
                package_top_k=1,
                table_expand_ratio_threshold=1,
                max_inline_table_chars=5,
            ),
            artifact_text_resolver=lambda key: "FULL TABLE TEXT",
        )

        table_evidence = response.packages[0].evidence[0]
        self.assertEqual(table_evidence.mode, "table_reference")
        self.assertNotEqual(table_evidence.text, "FULL TABLE TEXT")
        self.assertIn("row one", table_evidence.text)

    def test_expands_candidate_top_k_with_upper_bound(self):
        from rag_service.retrieval.evidence_packages import expanded_candidate_top_k

        self.assertEqual(expanded_candidate_top_k(5, 6, 3, max_candidate_k=100), 90)
        self.assertEqual(expanded_candidate_top_k(20, 10, 5, max_candidate_k=100), 100)
        self.assertEqual(expanded_candidate_top_k(0, 10, 5, max_candidate_k=100), 1)

    def test_reuses_resolved_artifact_urls_within_one_response(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        calls = []

        def resolver(key):
            calls.append(key)
            return "signed://" + key

        candidates = [
            FakeRetrievedDocument(
                "one",
                "a.xlsx",
                0.8,
                {
                    "doc_id": "doc-a",
                    "block_id": "table_001_chunk_001",
                    "table_id": "table_001",
                    "display_ref": "doc-a/structured_excel/tables/table_001.html",
                },
            ),
            FakeRetrievedDocument(
                "two",
                "a.xlsx",
                0.7,
                {
                    "doc_id": "doc-a",
                    "block_id": "table_001_chunk_002",
                    "table_id": "table_001",
                    "display_ref": "doc-a/structured_excel/tables/table_001.html",
                },
            ),
        ]

        build_evidence_packages(
            query="q",
            rewrite_query="q",
            documents=candidates,
            options=EvidencePackageOptions(artifact_mode="signed_url"),
            artifact_url_resolver=resolver,
        )

        self.assertEqual(calls, ["doc-a/structured_excel/tables/table_001.html"])

    def test_runs_independent_retrievers_in_parallel(self):
        from rag_service.retrieval.evidence_packages import run_parallel_retrievers

        barrier = threading.Barrier(2)

        def first():
            barrier.wait(timeout=1)
            return ["local"]

        def second():
            barrier.wait(timeout=1)
            return ["remote"]

        self.assertEqual(
            sorted(run_parallel_retrievers([first, second])),
            ["local", "remote"],
        )

    def test_parallel_retrievers_skip_failed_branch(self):
        from rag_service.retrieval.evidence_packages import run_parallel_retrievers

        def failed():
            raise RuntimeError("boom")

        def ok():
            return ["local"]

        self.assertEqual(run_parallel_retrievers([failed, ok]), ["local"])

    def test_parallel_retrievers_log_failed_branch(self):
        from rag_service.retrieval.evidence_packages import run_parallel_retrievers

        def failed():
            raise RuntimeError("boom")

        def ok():
            return ["local"]

        failed.source = "ipd"
        failed.name = "ipd-primary"

        with self.assertLogs("rag_service.retrieval.evidence_packages", level="WARNING") as logs:
            self.assertEqual(run_parallel_retrievers([failed, ok]), ["local"])

        log_text = "\n".join(logs.output)
        self.assertIn("evidence.retriever_failed", log_text)
        self.assertIn("source=ipd", log_text)
        self.assertIn("name=ipd-primary", log_text)

    def test_parallel_retrievers_can_raise_when_all_branches_fail(self):
        from rag_service.retrieval.evidence_packages import run_parallel_retrievers

        def failed():
            raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "all retrievers failed"):
            run_parallel_retrievers([failed], raise_on_all_failed=True)

    def test_prefetches_unique_signed_urls_in_parallel(self):
        from rag_service.retrieval.evidence_packages import (
            EvidencePackageOptions,
            build_evidence_packages,
        )

        barrier = threading.Barrier(2)

        def resolver(key):
            barrier.wait(timeout=1)
            return "signed://" + key

        candidates = [
            FakeRetrievedDocument(
                "one",
                "a.xlsx",
                0.8,
                {
                    "doc_id": "doc-a",
                    "block_id": "block_001",
                    "table_id": "table_001",
                    "display_ref": "doc-a/structured_excel/tables/table_001.html",
                },
            ),
            FakeRetrievedDocument(
                "two",
                "b.xlsx",
                0.7,
                {
                    "doc_id": "doc-b",
                    "block_id": "block_001",
                    "table_id": "table_001",
                    "display_ref": "doc-b/structured_excel/tables/table_001.html",
                },
            ),
        ]

        response = build_evidence_packages(
            query="q",
            rewrite_query="q",
            documents=candidates,
            options=EvidencePackageOptions(artifact_mode="signed_url"),
            artifact_url_resolver=resolver,
        )

        refs = [package.artifacts.tables[0].refs["display"] for package in response.packages]
        self.assertEqual(
            sorted(refs),
            [
                "signed://doc-a/structured_excel/tables/table_001.html",
                "signed://doc-b/structured_excel/tables/table_001.html",
            ],
        )

    def test_caps_collected_candidates_globally_after_merging_branches(self):
        from rag_service.retrieval.evidence_packages import collect_evidence_candidate_documents

        batches = [
            ("libing", [FakeRetrievedDocument(f"local-{i}", "a.md", 0.1 + i) for i in range(100)]),
            ("ipd", [FakeRetrievedDocument(f"ipd-{i}", "b.md", 100 + i) for i in range(100)]),
            ("libing", [FakeRetrievedDocument(f"other-{i}", "c.md", 200 + i) for i in range(100)]),
        ]

        documents, has_ipd_documents = collect_evidence_candidate_documents(batches, max_documents=100)

        self.assertTrue(has_ipd_documents)
        self.assertEqual(len(documents), 100)
        self.assertEqual(documents[0].text, "other-99")
        self.assertEqual(documents[-1].text, "other-0")

    def test_collected_candidates_dedupe_by_text_keeps_highest_score(self):
        from rag_service.retrieval.evidence_packages import collect_evidence_candidate_documents

        low = FakeRetrievedDocument("same text", "a.md", 0.1)
        high = FakeRetrievedDocument("same text", "b.md", 0.9)

        documents, _ = collect_evidence_candidate_documents(
            [("libing", [low]), ("ipd", [high])],
            max_documents=10,
        )

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].score, 0.9)


_ANSWER_QUERY = "华东区域 Q1 收入为什么下降？"
_ANSWER_REWRITE_QUERY = "华东区域 Q1 收入下降的原因是什么？"


def _signed_artifact_url(key):
    return "signed://" + key


def _evidence_artifact_test_url(ref):
    return posixpath.join("/kb/evidence_artifacts", ref.replace("/", "_"))


def _structured_excel_summary():
    return {
        "artifact_prefix": "doc-1/structured_excel/",
        "document_markdown_key": "doc-1/structured_excel/document.md",
        "manifest_key": "doc-1/structured_excel/manifest.json",
        "table_count": 1,
        "block_count": 3,
    }


def _grouped_artifact_candidates():
    structured_excel = _structured_excel_summary()
    return [
        _grouped_table_candidate(structured_excel),
        FakeRetrievedDocument(
            "duplicate lower table evidence",
            "sales.xlsx",
            0.63,
            _grouped_table_metadata(structured_excel, "table_001_chunk_002"),
        ),
        FakeRetrievedDocument("nearby text evidence", "sales.xlsx", 0.72, _grouped_text_metadata(structured_excel)),
        FakeRetrievedDocument(
            "other doc evidence",
            "ops.md",
            0.5,
            {"kb_sn": "kb-1", "asset_name": "asset-ops", "doc_id": "doc-2", "block_id": "block_001"},
        ),
    ]


def _grouped_table_candidate(structured_excel):
    return FakeRetrievedDocument(
        "table evidence high",
        "sales.xlsx",
        0.91,
        _grouped_table_metadata(structured_excel, "table_001_chunk_001", include_refs=True),
        es_index="idx-a",
        es_doc_id="es-1",
    )


def _grouped_table_metadata(structured_excel, block_id, include_refs=False):
    metadata = {
        "kb_sn": "kb-1",
        "asset_name": "asset-sales",
        "doc_id": "doc-1",
        "block_type": "table",
        "block_id": block_id,
        "table_id": "table_001",
        "structured_excel": structured_excel,
    }
    if include_refs:
        metadata.update(
            {
                "title": "Sales Report",
                "headers": ["Sales", "Q1"],
                "display_ref": "doc-1/structured_excel/tables/table_001.html",
                "table_json_ref": "doc-1/structured_excel/tables/table_001.json",
                "llm_table_ref": "doc-1/structured_excel/tables/table_001.llm.md",
            }
        )
    return metadata


def _grouped_text_metadata(structured_excel):
    return {
        "kb_sn": "kb-1",
        "asset_name": "asset-sales",
        "doc_id": "doc-1",
        "block_type": "text",
        "block_id": "block_002",
        "headers": ["Sales"],
        "structured_excel": structured_excel,
    }


def _assert_grouped_artifact_response(test_case, response):
    test_case.assertEqual(response.query, "q")
    test_case.assertEqual(response.rewrite_query, "rewritten q")
    test_case.assertEqual(len(response.packages), 2)
    package = response.packages[0]
    test_case.assertEqual(package.doc_id, "doc-1")
    test_case.assertEqual(package.title, "Sales Report")
    test_case.assertEqual(package.artifacts.document.artifact_prefix, "doc-1/structured_excel/")
    test_case.assertEqual(package.artifacts.document.document_markdown, "signed://doc-1/structured_excel/document.md")
    test_case.assertEqual(package.artifacts.document.manifest, "signed://doc-1/structured_excel/manifest.json")
    test_case.assertEqual([item.table_id for item in package.evidence], ["table_001", None])
    test_case.assertEqual(package.evidence[0].text, "table evidence high")
    test_case.assertEqual(package.artifacts.tables[0].refs["display"], "signed://doc-1/structured_excel/tables/table_001.html")
    test_case.assertEqual(package.evidence[0].es_index, "idx-a")
    test_case.assertEqual(package.evidence[0].es_doc_id, "es-1")
    _assert_agent_response_hides_internal_refs(test_case, response.to_dict())


def _assert_agent_response_hides_internal_refs(test_case, response_dict):
    test_case.assertNotIn("artifact_summary", response_dict["packages"][0])
    test_case.assertNotIn("table_contexts", response_dict["packages"][0])
    test_case.assertNotIn("expansions", response_dict["packages"][0])
    test_case.assertNotIn("refs", response_dict["packages"][0]["evidence"][0])


def _answer_evidence_documents():
    return [
        FakeRetrievedDocument(
            "华东区域 Q1 收入同比下降 12%，主要原因是渠道库存消化周期拉长。",
            "2025经营分析报告.docx",
            0.92,
            {
                "doc_id": "doc-1",
                "title": "2025经营分析报告.docx",
                "block_type": "text",
                "block_id": "block_001",
                "section_id": "section_3_2",
                "headers": ["3.2 华东区域经营情况"],
            },
            es_index="idx-a",
            es_doc_id="es-text",
        ),
        _answer_table_document(),
    ]


def _answer_table_document():
    return FakeRetrievedDocument(
        "季度: Q1\n收入: 8800万\n同比: -12%\n主要原因: 渠道库存消化、新客户延迟、折扣提升",
        "2025经营分析报告.docx",
        0.9,
        {
            "doc_id": "doc-1",
            "title": "华东区域季度收入",
            "block_type": "table",
            "block_id": "table_001_chunk_001",
            "table_id": "table_001",
            "row_range": [0, 1],
            "table": {"title": "华东区域季度收入", "row_count": 2},
            "display_ref": "doc-1/structured_docx/tables/table_001.html",
            "table_json_ref": "doc-1/structured_docx/tables/table_001.json",
            "llm_table_ref": "doc-1/structured_docx/tables/table_001.llm.md",
        },
        es_index="idx-a",
        es_doc_id="es-table",
    )


def _assert_answer_evidence_response(test_case, api_response):
    test_case.assertEqual(api_response["query"], _ANSWER_QUERY)
    test_case.assertEqual(api_response["rewrite_query"], _ANSWER_REWRITE_QUERY)
    test_case.assertEqual(
        [(item["context_id"], item["source_id"], item["content_type"]) for item in api_response["llm_context"]],
        [("C1", "S1", "text"), ("C2", "S1", "table")],
    )
    test_case.assertIn("渠道库存消化周期拉长", api_response["llm_context"][0]["content"])
    test_case.assertIn("同比: -12%", api_response["llm_context"][1]["content"])
    _assert_answer_display_source(test_case, api_response["display_sources"][0])
    serialized = json.dumps(api_response, ensure_ascii=False)
    for hidden_value in ("display_ref", "table_json_ref", "llm_table_ref", "block_001", "idx-a"):
        test_case.assertNotIn(hidden_value, serialized)


def _assert_answer_display_source(test_case, source):
    test_case.assertEqual(source["source_id"], "S1")
    test_case.assertEqual(source["title"], "2025经营分析报告.docx")
    test_case.assertEqual([item["type"] for item in source["items"]], ["text", "table"])
    test_case.assertEqual(source["items"][0]["context_id"], "C1")
    test_case.assertEqual(source["items"][1]["context_id"], "C2")
    test_case.assertEqual(
        source["items"][1]["display"],
        {"mode": "url", "url": "/kb/evidence_artifacts/doc-1_structured_docx_tables_table_001.html"},
    )
    test_case.assertEqual(source["items"][1]["hit_row_ranges"], [[0, 1]])


def _agent_evidence_documents():
    return [
        FakeRetrievedDocument(
            "first matched table row",
            "sales.xlsx",
            0.91,
            _agent_table_metadata(),
            es_index="idx-sales",
            es_doc_id="es-table-1",
        ),
        FakeRetrievedDocument(
            "nearby explanation",
            "sales.xlsx",
            0.82,
            {
                "doc_id": "doc-1",
                "title": "Sales Report",
                "block_type": "text",
                "block_id": "block_004",
                "block_index": 4,
                "section_id": "section_sales",
            },
            es_index="idx-sales",
            es_doc_id="es-text-1",
        ),
    ]


def _agent_table_metadata():
    return {
        "doc_id": "doc-1",
        "title": "Sales Report",
        "block_type": "table",
        "block_id": "table_001_chunk_001",
        "block_index": 3,
        "table_id": "table_001",
        "row_range": [0, 20],
        "table": {"title": "Quarterly Sales", "row_count": 120},
        "display_ref": "doc-1/structured_excel/tables/table_001.html",
        "table_json_ref": "doc-1/structured_excel/tables/table_001.json",
        "llm_table_ref": "doc-1/structured_excel/tables/table_001.llm.md",
    }


def _assert_agent_evidence_response(test_case, agent_response):
    package = agent_response["packages"][0]
    test_case.assertEqual(package["doc_id"], "doc-1")
    test_case.assertEqual(package["title"], "Sales Report")
    test_case.assertEqual(len(package["evidence"]), 1)
    test_case.assertEqual([item["text"] for item in package["slices"]], ["first matched table row", "nearby explanation"])
    _assert_agent_table_slice(test_case, package["slices"][0])
    test_case.assertEqual(package["slices"][1]["slice_id"], "doc-1:block_004")
    test_case.assertEqual(package["slices"][1]["section_id"], "section_sales")


def _assert_agent_table_slice(test_case, table_slice):
    test_case.assertEqual(table_slice["slice_id"], "doc-1:table_001_chunk_001")
    test_case.assertEqual(table_slice["rank"], 1)
    test_case.assertEqual(table_slice["block_id"], "table_001_chunk_001")
    test_case.assertEqual(table_slice["block_index"], 3)
    test_case.assertEqual(table_slice["block_type"], "table")
    test_case.assertEqual(table_slice["table_id"], "table_001")
    test_case.assertEqual(table_slice["row_range"], (0, 20))
    test_case.assertEqual(
        table_slice["refs"],
        {
            "display": "doc-1/structured_excel/tables/table_001.html",
            "table_json": "doc-1/structured_excel/tables/table_001.json",
            "llm_table": "doc-1/structured_excel/tables/table_001.llm.md",
        },
    )
    test_case.assertEqual(table_slice["es_index"], "idx-sales")
    test_case.assertEqual(table_slice["es_doc_id"], "es-table-1")


def _related_section_candidates():
    return [
        FakeRetrievedDocument("sales intro", "report.md", 0.91, _section_metadata("block_001", 1, "Sales")),
        FakeRetrievedDocument("sales detail", "report.md", 0.9, _section_metadata("block_002", 2, "Sales")),
        FakeRetrievedDocument("ops unrelated", "report.md", 0.89, _section_metadata("block_003", 3, "Ops")),
    ]


def _section_metadata(block_id, block_index, title):
    return {
        "doc_id": "doc-a",
        "block_id": block_id,
        "block_index": block_index,
        "block_type": "text",
        "headers": ["Report", title],
        "title": title,
    }


def _assert_related_section_package(test_case, package):
    section_artifact = package.artifacts.sections[0]
    test_case.assertEqual(section_artifact.type, "section")
    test_case.assertEqual(section_artifact.mode, "merged_section")
    test_case.assertEqual(section_artifact.title, "Sales")
    test_case.assertEqual(section_artifact.headers, ["Report", "Sales"])
    test_case.assertEqual(section_artifact.hit_blocks, 2)
    test_case.assertEqual(section_artifact.merged_block_ids, ["block_001", "block_002"])
    test_case.assertEqual(package.evidence[0].evidence_type, "section")
    test_case.assertEqual(package.evidence[0].mode, "merged_section")
    test_case.assertEqual(package.evidence[0].headers, ["Report", "Sales"])
    test_case.assertEqual(package.evidence[0].merged_block_ids, ["block_001", "block_002"])
    test_case.assertEqual(package.evidence[0].text, "sales intro\n\nsales detail")
    test_case.assertEqual(package.evidence[1].text, "ops unrelated")


def _covered_table_chunk_candidates():
    table_summary = {"table_id": "table_001", "title": "Sales detail", "row_count": 10, "col_count": 2}
    return [
        FakeRetrievedDocument(
            f"row chunk {index}",
            "sales.xlsx",
            0.9 - index * 0.01,
            _covered_table_metadata(index, table_summary),
        )
        for index in range(4)
    ]


def _covered_table_metadata(index, table_summary):
    return {
        "doc_id": "doc-a",
        "block_id": f"table_001_chunk_{index + 1:03d}",
        "table_id": "table_001",
        "row_range": (index, index + 1),
        "table": table_summary,
        "display_ref": "doc-a/structured_excel/tables/table_001.html",
        "table_json_ref": "doc-a/structured_excel/tables/table_001.json",
        "llm_table_ref": "doc-a/structured_excel/tables/table_001.llm.md",
    }


def _covered_table_text(key):
    return "FULL TABLE TEXT" if key.endswith(".llm.md") else ""


def _assert_inline_table_evidence(test_case, evidence):
    test_case.assertEqual(len(evidence), 1)
    table_evidence = evidence[0]
    test_case.assertEqual(table_evidence.text, "FULL TABLE TEXT")
    test_case.assertEqual(table_evidence.mode, "full_table_inline")
    test_case.assertEqual(table_evidence.table_id, "table_001")
    test_case.assertEqual(table_evidence.hit_ratio, 0.4)
    test_case.assertEqual(table_evidence.hit_rows, 4)
    test_case.assertEqual(table_evidence.hit_row_ranges, [[0, 4]])
    test_case.assertEqual(table_evidence.row_count, 10)
    test_case.assertEqual(
        table_evidence.merged_block_ids,
        ["table_001_chunk_001", "table_001_chunk_002", "table_001_chunk_003", "table_001_chunk_004"],
    )


def _assert_inline_table_artifact(test_case, table_artifact):
    test_case.assertEqual(table_artifact.type, "table")
    test_case.assertEqual(table_artifact.mode, "full_table_inline")
    test_case.assertEqual(table_artifact.table_id, "table_001")
    test_case.assertEqual(table_artifact.hit_row_ranges, [[0, 4]])
    test_case.assertEqual(table_artifact.hit_rows, 4)
    test_case.assertEqual(table_artifact.row_count, 10)
    test_case.assertEqual(table_artifact.hit_ratio, 0.4)
    test_case.assertEqual(table_artifact.refs["table_json"], "doc-a/structured_excel/tables/table_001.json")
    test_case.assertTrue(table_artifact.expanded)


def _class_annotation_names(tree, class_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
            }
    return set()


if __name__ == "__main__":
    unittest.main()
