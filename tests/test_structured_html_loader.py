import unittest

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()


class StructuredHtmlLoaderTests(unittest.TestCase):
    def test_parse_html_outputs_text_and_table_blocks_without_duplicate_table_text(self):
        from rag_service.document_loaders.parsed_blocks import (
            BLOCK_TYPE_TABLE,
            BLOCK_TYPE_TEXT,
            SPLIT_POLICY_MARKDOWN_HEADINGS,
            SPLIT_POLICY_NO_SPLIT,
        )
        from rag_service.document_loaders.structured_artifacts import STRUCTURED_HTML_ARTIFACTS_KEY
        from rag_service.document_loaders.structured_html_loader import StructuredHtmlLoader

        html = """
        <html><body>
          <h1>Report</h1>
          <p>Intro before table.</p>
          <table>
            <caption>Sales</caption>
            <tr><th>Region</th><th>Q1</th><th>Q2</th></tr>
            <tr><td>East</td><td>100</td><td></td></tr>
          </table>
          <p>After table.</p>
        </body></html>
        """

        parsed_document = StructuredHtmlLoader("demo.html").parse_html(html)

        self.assertEqual(len(parsed_document.blocks), 3)
        text_block, table_block, tail_block = parsed_document.blocks
        self.assertEqual(text_block.block_type, BLOCK_TYPE_TEXT)
        self.assertEqual(text_block.split_policy, SPLIT_POLICY_MARKDOWN_HEADINGS)
        self.assertIn("# Report", text_block.text)
        self.assertIn("Intro before table.", text_block.text)
        self.assertNotIn("East", text_block.text)
        self.assertEqual(table_block.block_type, BLOCK_TYPE_TABLE)
        self.assertEqual(table_block.split_policy, SPLIT_POLICY_NO_SPLIT)
        self.assertIn("section: Report", table_block.text)
        self.assertIn("1. Region=East; Q1=100", table_block.text)
        self.assertEqual(tail_block.text, "After table.")

        artifacts = parsed_document.metadata[STRUCTURED_HTML_ARTIFACTS_KEY]
        self.assertEqual(artifacts["tables"][0]["llm_markdown"], table_block.text)
        self.assertIn("<table>", artifacts["tables"][0]["html"])

    def test_parse_html_splits_text_blocks_on_headings_with_correct_headers(self):
        from rag_service.document_loaders.structured_html_loader import StructuredHtmlLoader

        html = """
        <html><body>
          <h1>Report</h1>
          <p>Overview.</p>
          <h2>Risk</h2>
          <p>Risk detail.</p>
        </body></html>
        """

        parsed_document = StructuredHtmlLoader("demo.html").parse_html(html)

        self.assertEqual([block.text for block in parsed_document.blocks], ["# Report\n\nOverview.", "## Risk\n\nRisk detail."])
        self.assertEqual(parsed_document.blocks[0].metadata["headers"], ["Report"])
        self.assertEqual(parsed_document.blocks[1].metadata["headers"], ["Report", "Risk"])

    def test_parse_html_preserves_img_tags_as_markdown_urls(self):
        from rag_service.document_loaders.structured_artifacts import STRUCTURED_HTML_ARTIFACTS_KEY
        from rag_service.document_loaders.structured_html_loader import StructuredHtmlLoader

        html = """
        <html><body>
          <h1>Report</h1>
          <p><img src="https://example.test/diagram.png"></p>
        </body></html>
        """

        parsed_document = StructuredHtmlLoader("demo.html").parse_html(html)

        self.assertEqual(len(parsed_document.blocks), 1)
        self.assertEqual(parsed_document.blocks[0].text, "# Report\n\n![](https://example.test/diagram.png)")
        artifacts = parsed_document.metadata[STRUCTURED_HTML_ARTIFACTS_KEY]
        self.assertIn("![](https://example.test/diagram.png)", artifacts["document_markdown"])

    def test_parse_html_uploads_base64_images_and_records_object_keys(self):
        from unittest.mock import patch

        from rag_service.document_loaders.image_markdown import ImageMarkdown
        import rag_service.document_loaders.structured_html_loader as structured_html_loader
        from rag_service.document_loaders.structured_artifacts import STRUCTURED_HTML_ARTIFACTS_KEY
        from rag_service.document_loaders.structured_html_loader import StructuredHtmlLoader

        html = """
        <html><body>
          <h1>Report</h1>
          <p><img src="data:image/png;base64,aW1hZ2UtYnl0ZXM="></p>
        </body></html>
        """

        with patch.object(
            structured_html_loader,
            "_upload_html_image_bytes",
            lambda content, extension, object_key_prefix="": ImageMarkdown(
                markdown="![](https://example.test/image.png)",
                object_key="image-key-1",
            ),
            create=True,
        ):
            parsed_document = StructuredHtmlLoader("demo.html").parse_html(html)

        self.assertEqual(parsed_document.blocks[0].text, "# Report\n\n![](https://example.test/image.png)")
        artifacts = parsed_document.metadata[STRUCTURED_HTML_ARTIFACTS_KEY]
        self.assertEqual(artifacts["image_object_keys"], ["image-key-1"])

    def test_parse_html_uploads_images_under_configured_prefix(self):
        from unittest.mock import patch

        from rag_service.document_loaders.image_markdown import ImageMarkdown
        import rag_service.document_loaders.structured_html_loader as structured_html_loader
        from rag_service.document_loaders.structured_artifacts import STRUCTURED_HTML_ARTIFACTS_KEY
        from rag_service.document_loaders.structured_html_loader import StructuredHtmlLoader

        calls = []
        html = """
        <html><body>
          <h1>Report</h1>
          <p><img src="data:image/png;base64,aW1hZ2UtYnl0ZXM="></p>
        </body></html>
        """

        def fake_upload(content, extension, object_key_prefix=""):
            calls.append((content, extension, object_key_prefix))
            return ImageMarkdown(markdown="![](image)", object_key=object_key_prefix + "image.png")

        with patch.object(structured_html_loader, "upload_image_bytes", fake_upload):
            parsed_document = StructuredHtmlLoader(
                "demo.html",
                image_upload_prefix="asset/doc/artifacts/structured_html/images/",
            ).parse_html(html)

        self.assertEqual(calls, [(b"image-bytes", "png", "asset/doc/artifacts/structured_html/images/")])
        artifacts = parsed_document.metadata[STRUCTURED_HTML_ARTIFACTS_KEY]
        self.assertEqual(artifacts["image_object_keys"], ["asset/doc/artifacts/structured_html/images/image.png"])


if __name__ == "__main__":
    unittest.main()
