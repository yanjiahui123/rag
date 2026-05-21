import json
import posixpath
import unittest

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()

from rag_service.document_loaders.parsed_blocks import ParsedBlock, ParsedDocument
from rag_service.document_loaders.structured_artifacts import (
    PARSED_MARKDOWN_METADATA_KEY,
    STRUCTURED_MARKDOWN_ARTIFACTS_KEY,
    STRUCTURED_MARKDOWN_METADATA_KEY,
    STRUCTURED_EXCEL_ARTIFACTS_KEY,
    STRUCTURED_EXCEL_METADATA_KEY,
    STRUCTURED_HTML_ARTIFACTS_KEY,
    STRUCTURED_HTML_METADATA_KEY,
    STRUCTURED_DOCX_ARTIFACTS_KEY,
    STRUCTURED_DOCX_METADATA_KEY,
    build_document_artifact_images_prefix,
    build_document_artifacts_prefix,
    build_document_download_key,
    build_document_download_prefix,
    build_document_resource_prefix,
    build_knowledge_base_asset_artifact_prefix,
    build_parsed_markdown_artifact_prefix,
    build_structured_docx_artifact_prefix,
    build_structured_excel_artifact_prefix,
    build_structured_html_artifact_prefix,
    build_structured_markdown_artifact_prefix,
    extract_parsed_markdown_artifact_prefix,
    extract_structured_docx_artifact_prefix,
    extract_structured_excel_artifact_prefix,
    extract_structured_html_artifact_prefix,
    extract_structured_markdown_artifact_prefix,
    extract_structured_image_object_keys,
    is_safe_knowledge_base_asset_artifact_prefix,
    is_safe_structured_artifact_prefix,
    persist_parsed_markdown_artifact,
    persist_structured_docx_artifacts,
    persist_structured_excel_artifacts,
    persist_structured_html_artifacts,
    persist_structured_markdown_artifacts,
)

KBA_ID = "11111111-1111-1111-1111-111111111111"
DOC_ID = "22222222-2222-2222-2222-222222222222"
DOC_PREFIX = f"{KBA_ID}/{DOC_ID}"
ARTIFACTS_PREFIX = f"{DOC_PREFIX}/artifacts"


class StructuredArtifactsTests(unittest.TestCase):
    def test_persist_structured_docx_artifacts_uploads_payloads_and_adds_refs(self):
        uploaded = {}
        parsed_document = ParsedDocument(
            blocks=[
                ParsedBlock(text="intro", metadata={"block_id": "block_001"}),
                ParsedBlock(
                    text="table text",
                    metadata={"block_id": "block_002", "table_id": "table_001", "table": {}},
                ),
            ],
            metadata={
                STRUCTURED_DOCX_ARTIFACTS_KEY: {
                    "document_markdown": "intro\n\n| A | B |",
                    "manifest": {"blocks": [{"block_id": "block_002", "table_id": "table_001"}]},
                    "tables": [
                        {
                            "table_id": "table_001",
                            "html": "<table></table>",
                            "json": {"rows": [["1"]]},
                            "llm_markdown": "| A | B |",
                        }
                    ],
                }
            },
        )

        summary = persist_structured_docx_artifacts(
            parsed_document,
            KBA_ID,
            "asset",
            DOC_ID,
            upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
        )

        self.assertEqual(summary["table_count"], 1)
        self.assertEqual(summary["block_count"], 2)
        self.assertNotIn(STRUCTURED_DOCX_ARTIFACTS_KEY, parsed_document.metadata)
        self.assertIn(f"{ARTIFACTS_PREFIX}/structured_docx/document.md", uploaded)
        self.assertIn(f"{ARTIFACTS_PREFIX}/structured_docx/manifest.json", uploaded)
        self.assertIn(f"{ARTIFACTS_PREFIX}/structured_docx/tables/table_001.html", uploaded)
        manifest = _uploaded_json(uploaded, f"{ARTIFACTS_PREFIX}/structured_docx/manifest.json")
        self.assertEqual(
            manifest["blocks"][1]["display_ref"],
            f"{ARTIFACTS_PREFIX}/structured_docx/tables/table_001.html",
        )
        self.assertEqual(
            parsed_document.blocks[1].metadata["display_ref"],
            f"{ARTIFACTS_PREFIX}/structured_docx/tables/table_001.html",
        )
        self.assertEqual(
            parsed_document.blocks[1].metadata["llm_table_ref"],
            f"{ARTIFACTS_PREFIX}/structured_docx/tables/table_001.llm.md",
        )

    def test_persist_structured_artifacts_preserves_image_object_keys_for_cleanup(self):
        uploaded = {}
        parsed_document = ParsedDocument(
            blocks=[ParsedBlock(text="![](https://example.test/image.png)", metadata={"block_id": "block_001"})],
            metadata={
                STRUCTURED_HTML_ARTIFACTS_KEY: {
                    "document_markdown": "![](https://example.test/image.png)",
                    "tables": [],
                    "image_object_keys": ["image-key-1"],
                }
            },
        )

        summary = persist_structured_html_artifacts(
            parsed_document,
            "kb",
            "asset",
            "doc-1",
            upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
        )

        self.assertEqual(summary["image_object_keys"], ["image-key-1"])
        extended_metadata = {STRUCTURED_HTML_METADATA_KEY: summary}
        self.assertEqual(extract_structured_image_object_keys(extended_metadata), ["image-key-1"])

    def test_build_and_extract_structured_docx_artifact_prefix(self):
        prefix = build_structured_docx_artifact_prefix("kb", "asset", "doc-1")

        self.assertEqual(prefix, "kb/doc-1/artifacts/structured_docx/")
        self.assertEqual(
            extract_structured_docx_artifact_prefix(
                {STRUCTURED_DOCX_METADATA_KEY: {"artifact_prefix": prefix}}
            ),
            prefix,
        )
        self.assertIsNone(extract_structured_docx_artifact_prefix({}))

    def test_persist_structured_excel_artifacts_uses_excel_prefix_and_metadata_key(self):
        uploaded = {}
        parsed_document = ParsedDocument(
            blocks=[ParsedBlock(text="chunk", metadata={"block_id": "block_001", "table_id": "table_001"})],
            metadata={
                STRUCTURED_EXCEL_ARTIFACTS_KEY: {
                    "document_markdown": "excel text",
                    "tables": [
                        {
                            "table_id": "table_001",
                            "html": "<table></table>",
                            "json": {"rows": []},
                            "llm_markdown": "chunk",
                        }
                    ],
                }
            },
        )

        summary = persist_structured_excel_artifacts(
            parsed_document,
            KBA_ID,
            "asset",
            DOC_ID,
            upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
        )

        self.assertEqual(summary["artifact_prefix"], f"{ARTIFACTS_PREFIX}/structured_excel/")
        self.assertIn(STRUCTURED_EXCEL_METADATA_KEY, parsed_document.metadata)
        self.assertEqual(
            parsed_document.blocks[0].metadata["display_ref"],
            f"{ARTIFACTS_PREFIX}/structured_excel/tables/table_001.html",
        )
        self.assertEqual(
            extract_structured_excel_artifact_prefix(
                {STRUCTURED_EXCEL_METADATA_KEY: {"artifact_prefix": summary["artifact_prefix"]}}
            ),
            summary["artifact_prefix"],
        )
        self.assertEqual(build_structured_excel_artifact_prefix(KBA_ID, "asset", DOC_ID), summary["artifact_prefix"])

    def test_structured_artifact_prefixes_are_document_scoped_and_guarded(self):
        self.assertEqual(build_knowledge_base_asset_artifact_prefix(KBA_ID), f"{KBA_ID}/")
        self.assertEqual(build_document_resource_prefix(KBA_ID, DOC_ID), f"{DOC_PREFIX}/")
        self.assertEqual(build_document_download_prefix(KBA_ID, DOC_ID), f"{DOC_PREFIX}/download/")
        self.assertEqual(
            build_document_download_key(KBA_ID, DOC_ID, "Folder/My File.pdf"),
            f"{DOC_PREFIX}/download/My_File.pdf",
        )
        self.assertEqual(build_document_artifacts_prefix(KBA_ID, DOC_ID), f"{ARTIFACTS_PREFIX}/")
        self.assertEqual(
            build_document_artifact_images_prefix(KBA_ID, DOC_ID, "structured_docx"),
            f"{ARTIFACTS_PREFIX}/structured_docx/images/",
        )
        self.assertTrue(is_safe_knowledge_base_asset_artifact_prefix(f"{KBA_ID}/"))
        self.assertFalse(is_safe_knowledge_base_asset_artifact_prefix("not-a-uuid/"))
        self.assertEqual(
            build_structured_docx_artifact_prefix(KBA_ID, "asset/path", DOC_ID),
            f"{ARTIFACTS_PREFIX}/structured_docx/",
        )
        self.assertEqual(
            build_structured_excel_artifact_prefix(KBA_ID, "asset/path", DOC_ID),
            f"{ARTIFACTS_PREFIX}/structured_excel/",
        )
        self.assertEqual(
            build_structured_html_artifact_prefix(KBA_ID, "asset/path", DOC_ID),
            f"{ARTIFACTS_PREFIX}/structured_html/",
        )
        self.assertEqual(
            build_structured_markdown_artifact_prefix(KBA_ID, "asset/path", DOC_ID),
            f"{ARTIFACTS_PREFIX}/structured_markdown/",
        )
        self.assertEqual(
            build_parsed_markdown_artifact_prefix(KBA_ID, "asset/path", DOC_ID),
            f"{ARTIFACTS_PREFIX}/parsed_markdown/",
        )
        self.assertTrue(is_safe_structured_artifact_prefix(f"{ARTIFACTS_PREFIX}/structured_docx/"))
        self.assertTrue(is_safe_structured_artifact_prefix(f"{ARTIFACTS_PREFIX}/structured_excel/"))
        self.assertTrue(is_safe_structured_artifact_prefix(f"{ARTIFACTS_PREFIX}/structured_html/"))
        self.assertTrue(is_safe_structured_artifact_prefix(f"{ARTIFACTS_PREFIX}/structured_markdown/"))
        self.assertTrue(is_safe_structured_artifact_prefix(f"{ARTIFACTS_PREFIX}/parsed_markdown/"))
        self.assertTrue(is_safe_structured_artifact_prefix(f"{DOC_PREFIX}/structured_docx/"))
        self.assertTrue(is_safe_structured_artifact_prefix("doc-1/structured_docx/"))
        self.assertTrue(is_safe_structured_artifact_prefix("doc-1/structured_excel/"))
        self.assertTrue(is_safe_structured_artifact_prefix("doc-1/structured_html/"))
        self.assertTrue(is_safe_structured_artifact_prefix("doc-1/structured_markdown/"))
        self.assertTrue(is_safe_structured_artifact_prefix("doc-1/parsed_markdown/"))
        self.assertFalse(is_safe_structured_artifact_prefix("kb/asset/"))
        self.assertFalse(is_safe_structured_artifact_prefix("doc-1/"))
        self.assertFalse(is_safe_structured_artifact_prefix("kb/not-a-doc/structured_docx/"))

    def test_persist_structured_html_artifacts_uses_html_prefix_and_metadata_key(self):
        uploaded = {}
        parsed_document = ParsedDocument(
            blocks=[ParsedBlock(text="chunk", metadata={"block_id": "block_001", "table_id": "table_001"})],
            metadata={
                STRUCTURED_HTML_ARTIFACTS_KEY: {
                    "document_markdown": "html text",
                    "tables": [
                        {
                            "table_id": "table_001",
                            "html": "<table></table>",
                            "json": {"rows": []},
                            "llm_markdown": "chunk",
                        }
                    ],
                }
            },
        )

        summary = persist_structured_html_artifacts(
            parsed_document,
            KBA_ID,
            "asset",
            DOC_ID,
            upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
        )

        self.assertEqual(summary["artifact_prefix"], f"{ARTIFACTS_PREFIX}/structured_html/")
        self.assertIn(STRUCTURED_HTML_METADATA_KEY, parsed_document.metadata)
        self.assertEqual(
            parsed_document.blocks[0].metadata["display_ref"],
            f"{ARTIFACTS_PREFIX}/structured_html/tables/table_001.html",
        )
        self.assertEqual(
            extract_structured_html_artifact_prefix(
                {STRUCTURED_HTML_METADATA_KEY: {"artifact_prefix": summary["artifact_prefix"]}}
            ),
            summary["artifact_prefix"],
        )

    def test_persist_structured_markdown_artifacts_uploads_document_and_manifest(self):
        uploaded = {}
        parsed_document = ParsedDocument(
            blocks=[ParsedBlock(text="# Title\n\nBody", metadata={"block_id": "block_001"})],
            metadata={
                STRUCTURED_MARKDOWN_ARTIFACTS_KEY: {
                    "document_markdown": "# Title\n\nBody",
                    "tables": [],
                }
            },
        )

        summary = persist_structured_markdown_artifacts(
            parsed_document,
            KBA_ID,
            "asset",
            DOC_ID,
            upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
        )

        self.assertEqual(summary["artifact_prefix"], f"{ARTIFACTS_PREFIX}/structured_markdown/")
        self.assertEqual(_uploaded_text(uploaded, f"{ARTIFACTS_PREFIX}/structured_markdown/document.md"), "# Title\n\nBody")
        self.assertIn(f"{ARTIFACTS_PREFIX}/structured_markdown/manifest.json", uploaded)
        self.assertIn(STRUCTURED_MARKDOWN_METADATA_KEY, parsed_document.metadata)
        self.assertNotIn(STRUCTURED_MARKDOWN_ARTIFACTS_KEY, parsed_document.metadata)
        self.assertEqual(
            extract_structured_markdown_artifact_prefix(
                {STRUCTURED_MARKDOWN_METADATA_KEY: {"artifact_prefix": summary["artifact_prefix"]}}
            ),
            summary["artifact_prefix"],
        )

    def test_persist_structured_artifacts_uploads_sections_for_docx_html_and_markdown(self):
        for case in _section_artifact_cases():
            persist_func, artifacts_key, metadata_key, artifact_type = case
            with self.subTest(artifact_type=artifact_type):
                uploaded, parsed_document, summary = _persist_section_artifact_case(
                    persist_func,
                    artifacts_key,
                )

                _assert_section_artifact_upload(self, uploaded, parsed_document, summary, metadata_key, artifact_type)

    def test_persist_parsed_markdown_artifact_uploads_joined_blocks(self):
        uploaded = {}
        parsed_document = ParsedDocument(
            blocks=[
                ParsedBlock(text="first", metadata={"block_id": "block_001"}),
                ParsedBlock(text="second", metadata={"block_id": "block_002"}),
            ],
        )

        summary = persist_parsed_markdown_artifact(
            parsed_document,
            KBA_ID,
            "asset",
            DOC_ID,
            upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
        )

        self.assertEqual(summary["artifact_prefix"], f"{ARTIFACTS_PREFIX}/parsed_markdown/")
        self.assertEqual(summary["document_markdown_key"], f"{ARTIFACTS_PREFIX}/parsed_markdown/document.md")
        self.assertEqual(_uploaded_text(uploaded, f"{ARTIFACTS_PREFIX}/parsed_markdown/document.md"), "first\n\nsecond")
        self.assertEqual(parsed_document.metadata[PARSED_MARKDOWN_METADATA_KEY], summary)
        self.assertEqual(
            extract_parsed_markdown_artifact_prefix(
                {PARSED_MARKDOWN_METADATA_KEY: {"artifact_prefix": summary["artifact_prefix"]}}
            ),
            summary["artifact_prefix"],
        )


def _uploaded_text(uploaded, key):
    value = uploaded.get(key)
    if value is None:
        raise AssertionError(f"Missing uploaded artifact: {key}")
    return value


def _uploaded_json(uploaded, key):
    return json.loads(_uploaded_text(uploaded, key))


def _section_artifact_cases():
    return [
        (
            persist_structured_docx_artifacts,
            STRUCTURED_DOCX_ARTIFACTS_KEY,
            STRUCTURED_DOCX_METADATA_KEY,
            "structured_docx",
        ),
        (
            persist_structured_html_artifacts,
            STRUCTURED_HTML_ARTIFACTS_KEY,
            STRUCTURED_HTML_METADATA_KEY,
            "structured_html",
        ),
        (
            persist_structured_markdown_artifacts,
            STRUCTURED_MARKDOWN_ARTIFACTS_KEY,
            STRUCTURED_MARKDOWN_METADATA_KEY,
            "structured_markdown",
        ),
    ]


def _persist_section_artifact_case(persist_func, artifacts_key):
    uploaded = {}
    parsed_document = _section_artifact_document(artifacts_key)
    summary = persist_func(
        parsed_document,
        KBA_ID,
        "asset",
        DOC_ID,
        upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
    )
    return uploaded, parsed_document, summary


def _section_artifact_document(artifacts_key):
    return ParsedDocument(
        blocks=[
            ParsedBlock(text="sales intro", metadata={"block_id": "block_001", "headers": ["Report", "Sales"], "title": "Sales"}),
            ParsedBlock(text="sales detail", metadata={"block_id": "block_002", "headers": ["Report", "Sales"], "title": "Sales"}),
            ParsedBlock(text="ops note", metadata={"block_id": "block_003", "headers": ["Report", "Ops"], "title": "Ops"}),
        ],
        metadata={artifacts_key: {"document_markdown": "full document", "tables": []}},
    )


def _assert_section_artifact_upload(test_case, uploaded, parsed_document, summary, metadata_key, artifact_type):
    manifest_key = posixpath.join(KBA_ID, DOC_ID, "artifacts", artifact_type, "manifest.json")
    section_key = posixpath.join(KBA_ID, DOC_ID, "artifacts", artifact_type, "sections", "section_001.md")
    manifest = _uploaded_json(uploaded, manifest_key)
    sections = manifest.get("sections") or []
    first_section = sections[0]
    test_case.assertEqual(summary["artifact_prefix"], f"{ARTIFACTS_PREFIX}/{artifact_type}/")
    test_case.assertIn(metadata_key, parsed_document.metadata)
    test_case.assertEqual(len(sections), 2)
    test_case.assertEqual(first_section.get("section_id"), "section_001")
    test_case.assertEqual(first_section.get("headers"), ["Report", "Sales"])
    test_case.assertEqual(first_section.get("block_ids"), ["block_001", "block_002"])
    test_case.assertEqual(
        first_section.get("section_ref"),
        posixpath.join(KBA_ID, DOC_ID, "artifacts", artifact_type, "sections", "section_001.md"),
    )
    test_case.assertEqual(_uploaded_text(uploaded, section_key), "sales intro\n\nsales detail")
    test_case.assertEqual(parsed_document.blocks[0].metadata["section_id"], "section_001")
    test_case.assertEqual(parsed_document.blocks[0].metadata["section_ref"], section_key)


if __name__ == "__main__":
    unittest.main()
