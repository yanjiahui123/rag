import json
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
    is_safe_structured_artifact_prefix,
    persist_parsed_markdown_artifact,
    persist_structured_docx_artifacts,
    persist_structured_excel_artifacts,
    persist_structured_html_artifacts,
    persist_structured_markdown_artifacts,
)


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
            "kb",
            "asset",
            "doc-1",
            upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
        )

        self.assertEqual(summary["table_count"], 1)
        self.assertEqual(summary["block_count"], 2)
        self.assertNotIn(STRUCTURED_DOCX_ARTIFACTS_KEY, parsed_document.metadata)
        self.assertIn("doc-1/structured_docx/document.md", uploaded)
        self.assertIn("doc-1/structured_docx/manifest.json", uploaded)
        self.assertIn("doc-1/structured_docx/tables/table_001.html", uploaded)
        manifest = json.loads(uploaded["doc-1/structured_docx/manifest.json"])
        self.assertEqual(
            manifest["blocks"][1]["display_ref"],
            "doc-1/structured_docx/tables/table_001.html",
        )
        self.assertEqual(
            parsed_document.blocks[1].metadata["display_ref"],
            "doc-1/structured_docx/tables/table_001.html",
        )
        self.assertEqual(
            parsed_document.blocks[1].metadata["llm_table_ref"],
            "doc-1/structured_docx/tables/table_001.llm.md",
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

        self.assertEqual(prefix, "doc-1/structured_docx/")
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
            "kb",
            "asset",
            "doc-1",
            upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
        )

        self.assertEqual(summary["artifact_prefix"], "doc-1/structured_excel/")
        self.assertIn(STRUCTURED_EXCEL_METADATA_KEY, parsed_document.metadata)
        self.assertEqual(
            parsed_document.blocks[0].metadata["display_ref"],
            "doc-1/structured_excel/tables/table_001.html",
        )
        self.assertEqual(
            extract_structured_excel_artifact_prefix(
                {STRUCTURED_EXCEL_METADATA_KEY: {"artifact_prefix": summary["artifact_prefix"]}}
            ),
            summary["artifact_prefix"],
        )
        self.assertEqual(build_structured_excel_artifact_prefix("kb", "asset", "doc-1"), summary["artifact_prefix"])

    def test_structured_artifact_prefixes_are_document_scoped_and_guarded(self):
        self.assertEqual(build_structured_docx_artifact_prefix("kb", "asset/path", "doc-1"), "doc-1/structured_docx/")
        self.assertEqual(build_structured_excel_artifact_prefix("kb", "asset/path", "doc-1"), "doc-1/structured_excel/")
        self.assertEqual(build_structured_html_artifact_prefix("kb", "asset/path", "doc-1"), "doc-1/structured_html/")
        self.assertEqual(
            build_structured_markdown_artifact_prefix("kb", "asset/path", "doc-1"),
            "doc-1/structured_markdown/",
        )
        self.assertEqual(
            build_parsed_markdown_artifact_prefix("kb", "asset/path", "doc-1"),
            "doc-1/parsed_markdown/",
        )
        self.assertTrue(is_safe_structured_artifact_prefix("doc-1/structured_docx/"))
        self.assertTrue(is_safe_structured_artifact_prefix("doc-1/structured_excel/"))
        self.assertTrue(is_safe_structured_artifact_prefix("doc-1/structured_html/"))
        self.assertTrue(is_safe_structured_artifact_prefix("doc-1/structured_markdown/"))
        self.assertTrue(is_safe_structured_artifact_prefix("doc-1/parsed_markdown/"))
        self.assertFalse(is_safe_structured_artifact_prefix("kb/asset/"))
        self.assertFalse(is_safe_structured_artifact_prefix("doc-1/"))

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
            "kb",
            "asset",
            "doc-1",
            upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
        )

        self.assertEqual(summary["artifact_prefix"], "doc-1/structured_html/")
        self.assertIn(STRUCTURED_HTML_METADATA_KEY, parsed_document.metadata)
        self.assertEqual(
            parsed_document.blocks[0].metadata["display_ref"],
            "doc-1/structured_html/tables/table_001.html",
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
            "kb",
            "asset",
            "doc-1",
            upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
        )

        self.assertEqual(summary["artifact_prefix"], "doc-1/structured_markdown/")
        self.assertEqual(uploaded["doc-1/structured_markdown/document.md"], "# Title\n\nBody")
        self.assertIn("doc-1/structured_markdown/manifest.json", uploaded)
        self.assertIn(STRUCTURED_MARKDOWN_METADATA_KEY, parsed_document.metadata)
        self.assertNotIn(STRUCTURED_MARKDOWN_ARTIFACTS_KEY, parsed_document.metadata)
        self.assertEqual(
            extract_structured_markdown_artifact_prefix(
                {STRUCTURED_MARKDOWN_METADATA_KEY: {"artifact_prefix": summary["artifact_prefix"]}}
            ),
            summary["artifact_prefix"],
        )

    def test_persist_structured_artifacts_uploads_sections_for_docx_html_and_markdown(self):
        cases = [
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
        for persist_func, artifacts_key, metadata_key, artifact_type in cases:
            with self.subTest(artifact_type=artifact_type):
                uploaded = {}
                parsed_document = ParsedDocument(
                    blocks=[
                        ParsedBlock(
                            text="sales intro",
                            metadata={"block_id": "block_001", "headers": ["Report", "Sales"], "title": "Sales"},
                        ),
                        ParsedBlock(
                            text="sales detail",
                            metadata={"block_id": "block_002", "headers": ["Report", "Sales"], "title": "Sales"},
                        ),
                        ParsedBlock(
                            text="ops note",
                            metadata={"block_id": "block_003", "headers": ["Report", "Ops"], "title": "Ops"},
                        ),
                    ],
                    metadata={artifacts_key: {"document_markdown": "full document", "tables": []}},
                )

                summary = persist_func(
                    parsed_document,
                    "kb",
                    "asset",
                    "doc-1",
                    upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
                )

                manifest = json.loads(uploaded[f"doc-1/{artifact_type}/manifest.json"])
                self.assertEqual(summary["artifact_prefix"], f"doc-1/{artifact_type}/")
                self.assertIn(metadata_key, parsed_document.metadata)
                self.assertEqual(len(manifest["sections"]), 2)
                self.assertEqual(manifest["sections"][0]["section_id"], "section_001")
                self.assertEqual(manifest["sections"][0]["headers"], ["Report", "Sales"])
                self.assertEqual(manifest["sections"][0]["block_ids"], ["block_001", "block_002"])
                self.assertEqual(
                    manifest["sections"][0]["section_ref"],
                    f"doc-1/{artifact_type}/sections/section_001.md",
                )
                self.assertEqual(uploaded[f"doc-1/{artifact_type}/sections/section_001.md"], "sales intro\n\nsales detail")
                self.assertEqual(parsed_document.blocks[0].metadata["section_id"], "section_001")
                self.assertEqual(
                    parsed_document.blocks[0].metadata["section_ref"],
                    f"doc-1/{artifact_type}/sections/section_001.md",
                )

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
            "kb",
            "asset",
            "doc-1",
            upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
        )

        self.assertEqual(summary["artifact_prefix"], "doc-1/parsed_markdown/")
        self.assertEqual(summary["document_markdown_key"], "doc-1/parsed_markdown/document.md")
        self.assertEqual(uploaded["doc-1/parsed_markdown/document.md"], "first\n\nsecond")
        self.assertEqual(parsed_document.metadata[PARSED_MARKDOWN_METADATA_KEY], summary)
        self.assertEqual(
            extract_parsed_markdown_artifact_prefix(
                {PARSED_MARKDOWN_METADATA_KEY: {"artifact_prefix": summary["artifact_prefix"]}}
            ),
            summary["artifact_prefix"],
        )


if __name__ == "__main__":
    unittest.main()
