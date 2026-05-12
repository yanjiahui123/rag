import threading
import ast
import sys
import types
import unittest
from pathlib import Path


class VectorStoreParallelRetrievalTests(unittest.TestCase):
    def test_runs_ordered_tasks_in_parallel(self):
        _install_vectorstore_import_stubs()
        from rag_service.vectorstore.parallel_retrieval import run_ordered_parallel_tasks

        barrier = threading.Barrier(2)

        def first():
            barrier.wait(timeout=1)
            return ["first"]

        def second():
            barrier.wait(timeout=1)
            return ["second"]

        self.assertEqual(run_ordered_parallel_tasks([first, second]), ["first", "second"])

    def test_full_text_strategy_does_not_require_embedding(self):
        _install_vectorstore_import_stubs()
        from rag_service.vectorstore.parallel_retrieval import query_strategy_requires_embedding

        full_text_strategy = types.SimpleNamespace(name="FULL_TEXT_QUERY")
        hybrid_strategy = types.SimpleNamespace(name="HYBRID_QUERY")

        self.assertFalse(query_strategy_requires_embedding(full_text_strategy))
        self.assertTrue(query_strategy_requires_embedding(hybrid_strategy))

    def test_elasticsearch_manager_retrieve_uses_parallel_tasks(self):
        source = Path("rag_service/vectorstore/elasticsearch/elasticsearch.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        retrieve = _find_class_method(tree, "ElasticsearchManager", "retrieve")
        called_names = {
            getattr(call.func, "id", getattr(call.func, "attr", ""))
            for call in ast.walk(retrieve)
            if isinstance(call, ast.Call)
        }

        self.assertIn("run_ordered_parallel_tasks", called_names)


def _install_vectorstore_import_stubs():
    config = sys.modules.get("rag_service.config", types.ModuleType("rag_service.config"))
    config.VECTOR_STORE_TYPE = "ELASTICSEARCH"
    sys.modules["rag_service.config"] = config

    fastapi = sys.modules.get("fastapi", types.ModuleType("fastapi"))
    fastapi.BackgroundTasks = type("BackgroundTasks", (), {})
    sys.modules["fastapi"] = fastapi

    langchain = sys.modules.get("langchain", types.ModuleType("langchain"))
    langchain_schema = sys.modules.get("langchain.schema", types.ModuleType("langchain.schema"))
    langchain_schema.Document = type("Document", (), {})
    sys.modules["langchain"] = langchain
    sys.modules["langchain.schema"] = langchain_schema

    api_models = sys.modules.get("rag_service.models.api.models", types.ModuleType("rag_service.models.api.models"))
    api_models.RetrievedDocument = type("RetrievedDocument", (), {})
    sys.modules["rag_service.models.api.models"] = api_models

    telemetry = sys.modules.get("rag_service.telemetry", types.ModuleType("rag_service.telemetry"))
    collect_answer_info = sys.modules.get(
        "rag_service.telemetry.collect_answer_info",
        types.ModuleType("rag_service.telemetry.collect_answer_info"),
    )
    collect_answer_info.collect_time_use_info = identity_decorator
    sys.modules["rag_service.telemetry"] = telemetry
    sys.modules["rag_service.telemetry.collect_answer_info"] = collect_answer_info

    enums = sys.modules.get("rag_service.models.enums", types.ModuleType("rag_service.models.enums"))

    class VectorStoreType:
        ELASTICSEARCH = "ELASTICSEARCH"

        @classmethod
        def __class_getitem__(cls, name):
            return getattr(cls, name)

    enums.Analyzer = type("Analyzer", (), {"IK_ANALYZER": "IK_ANALYZER"})
    enums.EmbeddingModel = type("EmbeddingModel", (), {})
    enums.QueryStrategy = type("QueryStrategy", (), {"HYBRID_QUERY": "HYBRID_QUERY"})
    enums.VectorStoreType = VectorStoreType
    sys.modules["rag_service.models.enums"] = enums


def identity_decorator(func):
    return func


def _find_class_method(tree, class_name, method_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"{class_name}.{method_name} not found")


if __name__ == "__main__":
    unittest.main()
