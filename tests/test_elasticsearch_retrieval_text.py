import sys
import types
import unittest
import importlib.util
from pathlib import Path

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()


class ElasticsearchRetrievalTextTests(unittest.TestCase):
    def setUp(self):
        _install_es_model_stubs()

    def test_index_mapping_defines_content_and_retrieval_fields(self):
        es_model = _load_es_model()

        properties = es_model.es_index_mapping(3)["mappings"]["properties"]

        self.assertEqual(properties["content_text"]["type"], "text")
        self.assertEqual(properties["retrieval_text"]["type"], "text")
        self.assertEqual(properties["retrieval_text_vector"]["dims"], 3)
        self.assertEqual(properties["section_path"]["type"], "text")
        self.assertEqual(properties["section_title"]["type"], "text")
        self.assertEqual(properties["table_headers"]["type"], "text")

    def test_hybrid_query_searches_retrieval_fields_and_vector(self):
        es_model = _load_es_model()

        query = es_model.es_match_query("Risk", [0.1, 0.2, 0.3], 10, _FakeSearchInfo())
        searched_fields = _match_fields(query["query"]["bool"]["should"])

        self.assertIn("retrieval_text", searched_fields)
        self.assertIn("section_path", searched_fields)
        self.assertIn("section_title", searched_fields)
        self.assertIn("table_title", searched_fields)
        self.assertIn("table_headers", searched_fields)
        self.assertEqual(query["knn"]["field"], "retrieval_text_vector")

    def test_document_text_fields_keep_clean_content_and_structured_retrieval_text(self):
        es_model = _load_es_model()

        fields = es_model.es_document_text_fields(
            "section: Report > Risk\n\nRisk detail.",
            {
                "content_text": "Risk detail.",
                "retrieval_text": "section: Report > Risk\n\nRisk detail.",
                "section_path": "Report > Risk",
                "section_title": "Risk",
                "title": "Risk Register",
                "table": {"title": "Risk Table", "flatten_headers": ["Owner", "Status"]},
            },
        )

        self.assertEqual(fields["general_text"], "Risk detail.")
        self.assertEqual(fields["retrieval_text"], "section: Report > Risk\n\nRisk detail.")
        self.assertEqual(fields["section_path"], "Report > Risk")
        self.assertEqual(fields["section_title"], "Risk")
        self.assertEqual(fields["doc_title"], "Risk Register")
        self.assertEqual(fields["table_title"], "Risk Table")
        self.assertEqual(fields["table_headers"], "Owner Status")


class _FakeSearchInfo:
    document_close_sources = []


def _install_es_model_stubs():
    api_models = types.ModuleType("rag_service.models.api.models")
    api_models.RetrieveMetadata = type("RetrieveMetadata", (), {})
    sys.modules["rag_service.models.api.models"] = api_models

    generic_models = types.ModuleType("rag_service.models.generic.models")
    generic_models.VectorStoreElasticSearchQueryInfo = type("VectorStoreElasticSearchQueryInfo", (), {})
    sys.modules["rag_service.models.generic.models"] = generic_models


def _load_es_model():
    module_name = "es_model_under_test"
    path = Path("rag_service/vectorstore/elasticsearch/es_model.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _match_fields(should_clauses):
    fields = []
    for clause in should_clauses:
        match = clause.get("match")
        if match:
            fields.extend(match.keys())
    return fields


if __name__ == "__main__":
    unittest.main()
