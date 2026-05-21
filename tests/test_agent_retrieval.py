import ast
import base64
import contextlib
import hashlib
import hmac
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeRetrieveMetadata:
    def __init__(self, title="", file_address=""):
        self.title = title
        self.file_address = file_address


class FakeMetadata:
    def __init__(self, source, extended_metadata=None, retrieve_metadata=None):
        self.source = source
        self.extended_metadata = extended_metadata or {}
        self.retrieve_metadata = retrieve_metadata or FakeRetrieveMetadata(
            title=self.extended_metadata.get("title", ""),
            file_address=source,
        )


class FakeRetrievedDocument:
    def __init__(self, text, source, score, extended_metadata=None, es_index=None, es_doc_id=None):
        self.text = text
        self.score = score
        self.metadata = FakeMetadata(source, extended_metadata)
        self.es_index = es_index
        self.es_doc_id = es_doc_id


class FakeAPIRouter:
    def __init__(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return lambda func: func

    def post(self, *args, **kwargs):
        return lambda func: func


class FakeHTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def import_agent_router_with_fastapi_stub():
    fastapi_stub = ModuleType("fastapi")
    fastapi_stub.APIRouter = FakeAPIRouter
    fastapi_stub.Depends = lambda dependency=None: dependency
    fastapi_stub.HTTPException = FakeHTTPException
    fastapi_stub.Request = object
    with patch.dict(sys.modules, {"fastapi": fastapi_stub}):
        sys.modules.pop("rag_service.agent_retrieval.router", None)
        import rag_service.agent_retrieval.router as router_module
    return router_module


def signed_handle(payload, uid="user-1", secret="agent-retrieval-local-secret"):
    body = {"uid": uid, "exp": 2000000000, "payload": payload}
    body_token = b64encode(json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = b64encode(hmac.new(secret.encode("utf-8"), body_token.encode("ascii"), hashlib.sha256).digest())
    return f"{body_token}.{signature}"


def b64encode(value):
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


class AgentRetrievalTests(unittest.TestCase):
    def test_obs_key_payload_accepts_asset_scoped_artifact_paths(self):
        from rag_service.agent_retrieval import service as svc

        kba_id = "11111111-1111-1111-1111-111111111111"
        doc_id = "22222222-2222-2222-2222-222222222222"
        prefix = f"{kba_id}/{doc_id}/artifacts/structured_docx/"

        document_payload = svc._obs_payload(prefix + "manifest.json", "document")
        section_payload = svc._obs_payload(prefix + "sections/s1.md", "section")
        table_payload = svc._obs_payload(prefix + "tables/t1.html", "table")

        self.assertEqual(document_payload["doc_id"], doc_id)
        self.assertEqual(document_payload["knowledge_base_asset_id"], kba_id)
        self.assertEqual(document_payload["manifest_key"], prefix + "manifest.json")
        self.assertEqual(section_payload["doc_id"], doc_id)
        self.assertEqual(section_payload["knowledge_base_asset_id"], kba_id)
        self.assertEqual(section_payload["section_id"], "s1")
        self.assertEqual(table_payload["doc_id"], doc_id)
        self.assertEqual(table_payload["knowledge_base_asset_id"], kba_id)
        self.assertEqual(table_payload["table_id"], "t1")

    def test_search_slices_projects_documents_with_action_handles(self):
        from rag_service.agent_retrieval.models import SearchSlicesRequest
        from rag_service.agent_retrieval.service import AgentRetrievalService

        artifact_map = {
            "doc-1/structured_markdown/sections/section_001.md": "# Risk\n\nFull section text",
            "doc-1/structured_markdown/manifest.json": json.dumps(
                {
                    "sections": [
                        {
                            "section_id": "section_001",
                            "title": "Risk",
                            "headers": ["Report", "Risk"],
                            "block_ids": ["block_001"],
                            "section_ref": "doc-1/structured_markdown/sections/section_001.md",
                        }
                    ],
                    "blocks": [
                        {
                            "block_id": "block_001",
                            "type": "text",
                            "title": "Risk",
                            "headers": ["Report", "Risk"],
                            "section_id": "section_001",
                            "section_ref": "doc-1/structured_markdown/sections/section_001.md",
                            "text": "Risk slice text",
                        }
                    ],
                }
            ),
        }
        fake_documents = [
            FakeRetrievedDocument(
                "Risk slice text",
                "report.md",
                0.88,
                {
                    "kb_sn": "kb-1",
                    "asset_name": "asset-1",
                    "doc_id": "doc-1",
                    "title": "Risk Report",
                    "block_type": "text",
                    "block_id": "block_001",
                    "block_index": 3,
                    "section_id": "section_001",
                    "section_ref": "doc-1/structured_markdown/sections/section_001.md",
                    "section_path": "Report > Risk",
                    "section_title": "Risk",
                    "headers": ["Report", "Risk"],
                    "structured_markdown": {
                        "manifest_key": "doc-1/structured_markdown/manifest.json",
                        "document_markdown_key": "doc-1/structured_markdown/document.md",
                    },
                },
                es_index="idx-a",
                es_doc_id="es-1",
            )
        ]
        service = AgentRetrievalService(
            retrieve_documents=lambda request, uid, session=None: fake_documents,
            artifact_text_resolver=lambda key: artifact_map.get(key),
        )

        response = service.search_slices(
            SearchSlicesRequest(query="risk", kb_sn_list=["kb-1"], top_k=5),
            uid="user-1",
        )

        self.assertEqual(response.query, "risk")
        self.assertEqual(response.kb_sn_list, ["kb-1"])
        self.assertEqual(len(response.slices), 1)
        result = response.slices[0]
        self.assertEqual(result.text, "Risk slice text")
        self.assertEqual(result.doc.doc_id, "doc-1")
        self.assertEqual(result.location.section_path, "Report > Risk")
        self.assertTrue(result.actions.can_get_section)
        self.assertFalse(result.actions.can_get_table)
        self.assertTrue(result.actions.can_get_original_text)
        self.assertEqual(result.handles.section_handle, "doc-1/structured_markdown/sections/section_001.md")
        self.assertEqual(result.handles.document_handle, "doc-1/structured_markdown/manifest.json")

        section = service.get_section(
            {"section_handle": result.handles.section_handle, "max_chars": 100},
            uid="user-1",
        )
        self.assertEqual(section.text, "# Risk\n\nFull section text")
        self.assertFalse(section.truncated)

    def test_signed_handles_are_rejected_by_agent_retrieval_service(self):
        from rag_service.agent_retrieval.models import SectionRequest
        from rag_service.agent_retrieval.service import AgentRetrievalService

        service = AgentRetrievalService(
            retrieve_documents=lambda request, uid, session=None: [],
            artifact_text_resolver=lambda key: "legacy section",
        )
        legacy_handle = signed_handle(
            {"kind": "section", "section_ref": "doc-1/structured_markdown/sections/section_001.md"},
            uid="user-1",
        )

        with self.assertRaises(PermissionError):
            service.get_section(SectionRequest(section_handle=legacy_handle), uid="user-1")

    def test_obs_key_handles_resolve_document_table_and_original_text(self):
        from rag_service.agent_retrieval.models import (
            DocumentOutlineRequest,
            OriginalTextRequest,
            TableRequest,
        )
        from rag_service.agent_retrieval.service import AgentRetrievalService

        manifest = {
            "sections": [
                {
                    "section_id": "section_001",
                    "title": "Sales",
                    "headers": ["Report", "Sales"],
                    "block_ids": ["block_001", "table_001_chunk_001", "block_002"],
                    "section_ref": "doc-1/structured_excel/sections/section_001.md",
                }
            ],
            "blocks": [
                {
                    "block_id": "block_001",
                    "type": "text",
                    "section_id": "section_001",
                    "title": "Sales intro",
                    "text": "before table",
                },
                {
                    "block_id": "table_001_chunk_001",
                    "type": "table",
                    "title": "Sales table",
                    "table_id": "table_001",
                    "display_ref": "doc-1/structured_excel/tables/table_001.html",
                    "table_json_ref": "doc-1/structured_excel/tables/table_001.json",
                    "llm_table_ref": "doc-1/structured_excel/tables/table_001.llm.md",
                    "row_count": 2,
                    "text": "table slice",
                },
                {
                    "block_id": "block_002",
                    "type": "text",
                    "section_id": "section_001",
                    "title": "Sales outro",
                    "text": "after table",
                },
            ],
        }
        artifact_map = {
            "doc-1/structured_excel/manifest.json": json.dumps(manifest),
            "doc-1/structured_excel/document.md": "whole document",
            "doc-1/structured_excel/tables/table_001.llm.md": "| Region | Sales |",
            "doc-1/structured_excel/tables/table_001.json": json.dumps({"rows": [["CN", "10"]]}),
            "doc-1/structured_excel/tables/table_001.html": "<table></table>",
        }
        service = AgentRetrievalService(
            retrieve_documents=lambda request, uid, session=None: [],
            artifact_text_resolver=lambda key: artifact_map.get(key),
        )

        outline = service.get_document_outline(
            DocumentOutlineRequest(document_handle="doc-1/structured_excel/manifest.json"),
            uid="user-1",
        )
        self.assertEqual(outline.doc_id, "doc-1")
        self.assertEqual(outline.tables[0].table_handle, "doc-1/structured_excel/tables/table_001.llm.md")

        table = service.get_table(
            TableRequest(table_handle="doc-1/structured_excel/tables/table_001.llm.md", mode="json"),
            uid="user-1",
        )
        self.assertEqual(table.table_id, "table_001")
        self.assertEqual(table.content, json.dumps({"rows": [["CN", "10"]]}))

        original = service.get_original_text(
            OriginalTextRequest(
                document_handle="doc-1/structured_excel/manifest.json",
                center_block_id="table_001_chunk_001",
                before=1,
                after=1,
                max_chars=100,
            ),
            uid="user-1",
        )
        self.assertEqual([block.block_id for block in original.blocks], ["block_001", "table_001_chunk_001", "block_002"])

    def test_skill_package_has_opencode_files_and_no_token_requirement(self):
        from rag_service.agent_retrieval.skill_generator import generate_opencode_skill_package

        package = generate_opencode_skill_package(base_url="https://rag.example.com", kb_sn_list=["kb-1"])
        config = json.loads(package.files["config.json"])

        self.assertIn("SKILL.md", package.files)
        self.assertIn("config.json", package.files)
        self.assertNotIn("config.example.json", package.files)
        self.assertIn("references/api_schema.md", package.files)
        self.assertIn("scripts/kb_retrieval.py", package.files)
        self.assertIn("scripts/kb_retrieval.sh", package.files)
        self.assertIn("scripts/kb_retrieval_request.py", package.files)
        self.assertIn("scripts/kb_retrieval_request.sh", package.files)
        self.assertIn("request.json", package.files)
        joined = "\n".join(package.files.values())
        self.assertIn("KB_RETRIEVAL_BASE_URL", joined)
        self.assertIn("KB_SN_LIST", joined)
        self.assertIn("KB_RETRIEVAL_UID", joined)
        self.assertNotIn("KB_RETRIEVAL_" + "TOKEN", joined)
        self.assertIn("uid", config)
        self.assertTrue(json.loads(package.files["request.json"])["utf8_output"])
        self.assertIn(".opencode/skills/kb-retrieval/request.json", package.files["SKILL.md"])
        self.assertIn(".opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py", package.files["SKILL.md"])
        self.assertIn("Do not prepend a Windows drive-path `cd`", package.files["SKILL.md"])
        self.assertIn("forward slashes", package.files["SKILL.md"])
        self.assertNotIn("python scripts/kb_retrieval.py search", package.files["SKILL.md"])
        self.assertIn("config.json", package.files["SKILL.md"])
        self.assertIn("default_config_path", package.files["scripts/kb_retrieval.py"])
        self.assertIn('"uid": config.get("uid")', package.files["scripts/kb_retrieval.py"])

    def test_generated_request_runner_dispatches_unicode_query_without_shell_args(self):
        from rag_service.agent_retrieval.skill_generator import generate_opencode_skill_package

        package = generate_opencode_skill_package(base_url="https://rag.example.com", kb_sn_list=["kb-1"])
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            for relative_path, content in package.files.items():
                path = skill_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            (skill_dir / "request.json").write_text(
                json.dumps(
                    {
                        "command": "search",
                        "pretty": True,
                        "query": "\u7f51\u5361 \u964d\u901f \u539f\u56e0",
                        "top_k": 3,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            captured = {}
            sys.modules.pop("kb_retrieval", None)
            sys.path.insert(0, str(skill_dir / "scripts"))
            try:
                import kb_retrieval

                def fake_post_json(base_url, path, payload, timeout):
                    captured.update(
                        {
                            "base_url": base_url,
                            "path": path,
                            "payload": payload,
                            "timeout": timeout,
                        }
                    )
                    return {"ok": True, "query": payload["query"]}

                kb_retrieval.post_json = fake_post_json
                runner_globals = runpy.run_path(str(skill_dir / "scripts" / "kb_retrieval_request.py"))
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    runner_globals["main"]()
            finally:
                sys.path.remove(str(skill_dir / "scripts"))
                sys.modules.pop("kb_retrieval", None)

        self.assertEqual(captured["base_url"], "https://rag.example.com")
        self.assertEqual(captured["path"], "/agent/retrieval/search_slices")
        self.assertEqual(captured["payload"]["query"], "\u7f51\u5361 \u964d\u901f \u539f\u56e0")
        self.assertEqual(captured["payload"]["top_k"], 3)
        self.assertIn("\u7f51\u5361", stdout.getvalue())
        self.assertNotIn(r"\u7f51\u5361", stdout.getvalue())

    def test_generated_client_prints_ascii_safe_json_by_default(self):
        from rag_service.agent_retrieval.skill_generator import generate_opencode_skill_package

        package = generate_opencode_skill_package(base_url="https://rag.example.com", kb_sn_list=["kb-1"])
        module = ModuleType("generated_kb_retrieval_client")
        exec(package.files["scripts/kb_retrieval.py"], module.__dict__)

        default_output = io.StringIO()
        with contextlib.redirect_stdout(default_output):
            module.print_json({"query": "\u7f51\u5361 \u964d\u901f \u539f\u56e0"})

        utf8_output = io.StringIO()
        with contextlib.redirect_stdout(utf8_output):
            module.print_json({"query": "\u7f51\u5361 \u964d\u901f \u539f\u56e0"}, utf8_output=True)

        self.assertIn(r"\u7f51\u5361", default_output.getvalue())
        self.assertNotIn("\u7f51\u5361", default_output.getvalue())
        self.assertIn("\u7f51\u5361", utf8_output.getvalue())

    def test_request_uid_falls_back_to_body_uid_for_agent_clients(self):
        from rag_service.agent_retrieval.models import SearchSlicesRequest

        router_module = import_agent_router_with_fastapi_stub()
        request = SimpleNamespace(state=SimpleNamespace())

        uid = router_module._request_uid(request, SearchSlicesRequest(query="risk", uid="agent-user"))

        self.assertEqual(uid, "agent-user")

    def test_request_state_uid_takes_precedence_over_body_uid(self):
        from rag_service.agent_retrieval.models import SearchSlicesRequest

        router_module = import_agent_router_with_fastapi_stub()
        request = SimpleNamespace(state=SimpleNamespace(uid="web-user"))

        uid = router_module._request_uid(request, SearchSlicesRequest(query="risk", uid="agent-user"))

        self.assertEqual(uid, "web-user")

    def test_router_returns_403_for_invalid_retrieval_handle(self):
        from rag_service.agent_retrieval.models import TableRequest

        router_module = import_agent_router_with_fastapi_stub()
        request = SimpleNamespace(state=SimpleNamespace(uid="agent-user"))

        with self.assertRaises(FakeHTTPException) as ctx:
            router_module.get_table(request, TableRequest(table_handle="not-a-signed-handle"))

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("invalid retrieval handle", ctx.exception.detail)

    def test_router_is_standalone_agent_retrieval_surface(self):
        router_source = Path("rag_service/agent_retrieval/router.py").read_text(encoding="utf-8")

        self.assertIn('prefix="/agent/retrieval"', router_source)
        self.assertIn('"/search_slices"', router_source)
        self.assertIn('"/opencode/skill-package"', router_source)
        self.assertNotIn("knowledge_base_api", router_source)
        self.assertNotIn("KB_RETRIEVAL_" + "TOKEN", router_source)

    def test_app_router_autodiscovery_has_agent_retrieval_bridge(self):
        router_registry_source = Path("rag_service/rag_app/router/__init__.py").read_text(encoding="utf-8")
        bridge_path = Path("rag_service/rag_app/router/agent_retrieval_api.py")

        self.assertIn("rglob", router_registry_source)
        self.assertTrue(bridge_path.exists())
        self.assertIn("from rag_service.agent_retrieval.router import router", bridge_path.read_text(encoding="utf-8"))

    def test_app_configures_routers_when_imported_by_uvicorn(self):
        app_tree = ast.parse(Path("rag_service/rag_app/app.py").read_text(encoding="utf-8"))
        top_level_configure_calls = [
            node
            for node in app_tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "configure"
        ]

        self.assertTrue(top_level_configure_calls)


if __name__ == "__main__":
    unittest.main()
