import json
import unittest

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()

from rag_service.document_loaders.parsed_blocks import ParsedBlock, ParsedDocument
from rag_service.document_loaders import structured_artifacts as sa


KBA_ID = "11111111-1111-1111-1111-111111111111"
DOC_ID = "22222222-2222-2222-2222-222222222222"
DOC_PREFIX = f"{KBA_ID}/{DOC_ID}"
ARTIFACTS_PREFIX = f"{DOC_PREFIX}/artifacts"


class StructuredArtifactsBranchTests(unittest.TestCase):
    def test_prefix_and_download_helpers_accept_only_supported_shapes(self):
        self.assertEqual(sa.build_knowledge_base_asset_artifact_prefix(KBA_ID), f"{KBA_ID}/")
        self.assertEqual(sa.build_document_resource_prefix(KBA_ID, DOC_ID), f"{DOC_PREFIX}/")
        self.assertEqual(sa.build_document_download_prefix(KBA_ID, DOC_ID), f"{DOC_PREFIX}/download/")
        self.assertEqual(
            sa.build_document_download_key(KBA_ID, DOC_ID, r"folder\report.pdf"),
            f"{DOC_PREFIX}/download/source.pdf",
        )
        self.assertEqual(sa.build_document_download_key(KBA_ID, DOC_ID, "README"), f"{DOC_PREFIX}/download/source")
        self.assertEqual(sa.build_document_download_key(KBA_ID, DOC_ID, "report.bad ext"), f"{DOC_PREFIX}/download/source.bad_ext")
        self.assertEqual(sa.build_document_artifacts_prefix(KBA_ID, DOC_ID), f"{ARTIFACTS_PREFIX}/")
        self.assertEqual(
            sa.build_document_artifact_images_prefix(KBA_ID, DOC_ID, "structured_html"),
            f"{ARTIFACTS_PREFIX}/structured_html/images/",
        )

        self.assertFalse(sa.is_safe_knowledge_base_asset_artifact_prefix(None))
        self.assertFalse(sa.is_safe_knowledge_base_asset_artifact_prefix("kb/"))
        self.assertTrue(sa.is_safe_knowledge_base_asset_artifact_prefix(f"{KBA_ID}/"))
        self.assertFalse(sa.is_safe_structured_artifact_prefix(None))
        self.assertTrue(sa.is_safe_structured_artifact_prefix(f"{ARTIFACTS_PREFIX}/structured_html/"))
        self.assertTrue(sa.is_safe_structured_artifact_prefix(f"{DOC_PREFIX}/structured_docx/"))
        self.assertTrue(sa.is_safe_structured_artifact_prefix("doc-1/parsed_markdown/"))
        self.assertFalse(sa.is_safe_structured_artifact_prefix("doc-1/not_supported/"))

    def test_metadata_helpers_handle_missing_and_duplicate_image_keys(self):
        prefix = f"{ARTIFACTS_PREFIX}/structured_docx/"
        metadata = {sa.STRUCTURED_DOCX_METADATA_KEY: {"artifact_prefix": prefix}}

        self.assertIsNone(sa._extract_artifact_prefix(None, sa.STRUCTURED_DOCX_METADATA_KEY))
        self.assertIsNone(sa.extract_structured_html_artifact_prefix({}))
        self.assertEqual(sa.extract_structured_docx_artifact_prefix(metadata), prefix)
        self.assertEqual(sa.extract_structured_image_object_keys(None), [])
        self.assertEqual(
            sa.extract_structured_image_object_keys(
                {
                    sa.STRUCTURED_DOCX_METADATA_KEY: {sa.IMAGE_OBJECT_KEYS_METADATA_KEY: ["docx", "", "docx"]},
                    sa.STRUCTURED_HTML_METADATA_KEY: {sa.IMAGE_OBJECT_KEYS_METADATA_KEY: ["html", 3]},
                    sa.STRUCTURED_EXCEL_METADATA_KEY: {},
                }
            ),
            ["docx", "html", "3"],
        )
        self.assertEqual(sa._image_object_keys(None), [])
        self.assertEqual(
            sa._image_object_keys({sa.IMAGE_OBJECT_KEYS_METADATA_KEY: ["a", "", "a", 3]}),
            ["a", "3"],
        )

        self.assertEqual(
            sa.merge_structured_docx_metadata({"keep": True}, {"artifact_prefix": prefix}),
            {"keep": True, sa.STRUCTURED_DOCX_METADATA_KEY: {"artifact_prefix": prefix}},
        )
        self.assertEqual(
            sa.merge_structured_html_metadata(None, {"artifact_prefix": "html"})[sa.STRUCTURED_HTML_METADATA_KEY],
            {"artifact_prefix": "html"},
        )

    def test_excel_and_markdown_wrappers_persist_extract_and_merge_metadata(self):
        cases = [
            (
                sa.STRUCTURED_EXCEL_ARTIFACTS_KEY,
                sa.STRUCTURED_EXCEL_METADATA_KEY,
                sa.persist_structured_excel_artifacts,
                sa.extract_structured_excel_artifact_prefix,
                sa.merge_structured_excel_metadata,
            ),
            (
                sa.STRUCTURED_MARKDOWN_ARTIFACTS_KEY,
                sa.STRUCTURED_MARKDOWN_METADATA_KEY,
                sa.persist_structured_markdown_artifacts,
                sa.extract_structured_markdown_artifact_prefix,
                sa.merge_structured_markdown_metadata,
            ),
        ]

        for artifacts_key, metadata_key, persist, extract, merge in cases:
            with self.subTest(metadata_key=metadata_key):
                parsed_document = ParsedDocument(
                    blocks=[ParsedBlock(text="body", metadata={"block_id": "b1"})],
                    metadata={artifacts_key: {"document_markdown": "body", "tables": []}},
                )
                summary = persist(parsed_document, KBA_ID, "asset", DOC_ID, lambda key, content: key)
                self.assertEqual(extract({metadata_key: summary}), summary["artifact_prefix"])
                self.assertEqual(merge({}, summary)[metadata_key], summary)

    def test_persist_structured_artifacts_uploads_sections_tables_and_image_summary(self):
        uploaded = {}
        parsed_document = ParsedDocument(
            blocks=[
                ParsedBlock(text="custom intro", metadata={"block_id": "b1", "section_id": "custom/id", "title": "Custom"}),
                ParsedBlock(text="", metadata={"section_id": "custom/id", "title": "Custom"}),
                ParsedBlock(text="sales intro", metadata={"block_id": "b3", "headers": ["Report", "", "Sales"], "title": "Sales"}),
                ParsedBlock(
                    text="table rows",
                    metadata={"block_id": "b4", "headers": ["Report", "", "Sales"], "title": "Sales", "table_id": "t1"},
                ),
                ParsedBlock(text="loose tail", metadata={"block_id": "b5", "title": "Loose", "table_id": "missing"}),
            ],
            metadata={
                sa.STRUCTURED_HTML_ARTIFACTS_KEY: {
                    "document_markdown": "whole document",
                    "tables": [
                        {"table_id": "t1", "html": "<table></table>", "json": {"rows": [["1"]]}, "llm_markdown": "rows"}
                    ],
                    sa.IMAGE_OBJECT_KEYS_METADATA_KEY: ["image-1", "", "image-1", 2],
                }
            },
        )

        summary = sa.persist_structured_html_artifacts(
            parsed_document,
            KBA_ID,
            "unused-asset-name",
            DOC_ID,
            upload_content=lambda key, content: uploaded.setdefault(key, content) or key,
        )

        prefix = f"{ARTIFACTS_PREFIX}/structured_html/"
        manifest = json.loads(uploaded[prefix + "manifest.json"])
        self.assertEqual(summary["artifact_prefix"], prefix)
        self.assertEqual(summary["table_count"], 1)
        self.assertEqual(summary[sa.IMAGE_OBJECT_KEYS_METADATA_KEY], ["image-1", "2"])
        self.assertNotIn(sa.STRUCTURED_HTML_ARTIFACTS_KEY, parsed_document.metadata)
        self.assertEqual(uploaded[prefix + "sections/custom_id.md"], "custom intro")
        self.assertEqual(uploaded[prefix + "sections/section_002.md"], "sales intro\n\ntable rows")
        self.assertEqual(uploaded[prefix + "sections/section_003.md"], "loose tail")
        self.assertEqual(parsed_document.blocks[3].metadata["display_ref"], prefix + "tables/t1.html")
        self.assertNotIn("display_ref", parsed_document.blocks[4].metadata)
        self.assertEqual([section["section_id"] for section in manifest["sections"]], ["custom/id", "section_002", "section_003"])
        self.assertEqual(manifest["sections"][0]["block_ids"], ["b1"])

    def test_persist_without_images_omits_image_summary_field(self):
        parsed_document = ParsedDocument(
            blocks=[ParsedBlock(text="body", metadata={"block_id": "b1"})],
            metadata={sa.STRUCTURED_DOCX_ARTIFACTS_KEY: {"document_markdown": "body", "tables": []}},
        )

        summary = sa.persist_structured_docx_artifacts(parsed_document, "kb", "asset", "doc", lambda key, content: key)

        self.assertNotIn(sa.IMAGE_OBJECT_KEYS_METADATA_KEY, summary)
        self.assertEqual(summary["table_count"], 0)

    def test_empty_and_parsed_markdown_persistence_paths(self):
        self.assertEqual(
            sa.persist_structured_docx_artifacts(ParsedDocument(), KBA_ID, "asset", DOC_ID, lambda key, content: key),
            {},
        )
        self.assertEqual(
            sa.persist_parsed_markdown_artifact(ParsedDocument(), KBA_ID, "asset", DOC_ID, lambda key, content: key),
            {},
        )

        direct_uploads = {}
        direct = ParsedDocument(text="  direct markdown  ")
        direct_summary = sa.persist_parsed_markdown_artifact(
            direct,
            KBA_ID,
            "asset",
            DOC_ID,
            lambda key, content: direct_uploads.setdefault(key, content) or key,
        )
        self.assertEqual(direct_uploads[direct_summary["document_markdown_key"]], "direct markdown")
        self.assertEqual(
            sa.extract_parsed_markdown_artifact_prefix({sa.PARSED_MARKDOWN_METADATA_KEY: direct_summary}),
            direct_summary["artifact_prefix"],
        )

        block_uploads = {}
        blocks = ParsedDocument(blocks=[ParsedBlock(text="one"), ParsedBlock(text=""), ParsedBlock(text="two")])
        block_summary = sa.persist_parsed_markdown_artifact(
            blocks,
            KBA_ID,
            "asset",
            DOC_ID,
            lambda key, content: block_uploads.setdefault(key, content) or key,
        )
        self.assertEqual(block_uploads[block_summary["document_markdown_key"]], "one\n\ntwo")
        self.assertEqual(block_summary["block_count"], 3)


if __name__ == "__main__":
    unittest.main()
