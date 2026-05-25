import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()

from rag_service.document_loaders.image_markdown import ImageMarkdown
from rag_service.document_loaders.parsed_blocks import BLOCK_TYPE_TABLE, SPLIT_POLICY_NO_SPLIT
from rag_service.document_loaders.structured_artifacts import STRUCTURED_HTML_ARTIFACTS_KEY
from rag_service.document_loaders import structured_html_loader as sh


class StructuredHtmlLoaderBranchTests(unittest.TestCase):
    def test_parse_html_dispatches_heading_table_text_and_empty_flush_paths(self):
        loader = sh.StructuredHtmlLoader("", source="source.html")
        parsed = loader.parse_html(
            """
            <html><body>
              <h1>Report</h1>
              <p>Intro.</p>
              <table>
                <caption>Sales</caption>
                <tr><th>Region</th><th>Value</th></tr>
                <tr><td>East</td><td>1</td></tr>
              </table>
              <p>After table.</p>
            </body></html>
            """
        )

        self.assertEqual(len(parsed.blocks), 3)
        self.assertEqual(parsed.blocks[0].text, "# Report\n\nIntro.")
        self.assertEqual(parsed.blocks[1].block_type, BLOCK_TYPE_TABLE)
        self.assertEqual(parsed.blocks[1].split_policy, SPLIT_POLICY_NO_SPLIT)
        self.assertEqual(parsed.blocks[2].text, "After table.")
        self.assertIn("1. Region=East; Value=1", parsed.blocks[1].text)
        artifacts = parsed.metadata[STRUCTURED_HTML_ARTIFACTS_KEY]
        self.assertEqual(artifacts["tables"][0]["llm_markdown"], parsed.blocks[1].text)

        empty_state = sh._ParseState("source.html")
        loader._append_text_block(empty_state)
        self.assertEqual(empty_state.blocks, [])
        no_header_state = sh._ParseState("source.html")
        no_header_state.text_parts = ["", "Loose text"]
        loader._append_text_block(no_header_state)
        self.assertEqual(no_header_state.blocks[0].metadata["title"], "")

    def test_content_events_skip_non_content_and_recurse_into_nested_nodes(self):
        no_body_root = sh._content_root(sh.parse_html("<div>Fallback root</div>"))
        self.assertEqual(list(sh._content_events(no_body_root)), [("text", "Fallback root")])

        root = sh._content_root(
            sh.parse_html(
                """
                <html><body>
                  Loose text
                  <script>ignore script</script>
                  <style>ignore style</style>
                  <caption>ignore caption</caption>
                  <table><tr><td>A</td></tr></table>
                  <img src="//cdn.example/a.png">
                  <h2>Nested</h2>
                  <p>Paragraph <span>inside</span><img src=""></p>
                  <li>Item</li>
                  <div><span>Deep</span></div>
                </body></html>
                """
            )
        )
        events = list(sh._content_events(root))

        self.assertIn(("text", "Loose text"), events)
        self.assertIn(("text", "![](//cdn.example/a.png)"), events)
        self.assertIn(("heading", (2, "Nested")), events)
        self.assertIn(("text", "Paragraph inside"), events)
        self.assertIn(("text", "Item"), events)
        self.assertIn(("text", "Deep"), events)
        self.assertTrue(any(event_type == "table" for event_type, _ in events))
        self.assertFalse(any("ignore" in str(payload) for _, payload in events))
        self.assertEqual(sh._replace_header(["Report"], 3, "Detail"), ["Report", "", "Detail"])

    def test_image_sources_handle_missing_remote_data_and_upload_results(self):
        self.assertEqual(sh._image_markdown(sh.HtmlNode(tag="img", attrs={}), None, []), "")
        self.assertEqual(
            sh._image_markdown(sh.HtmlNode(tag="img", attrs={"src": "https://example.test/a.png"}), None, []),
            "![](https://example.test/a.png)",
        )
        self.assertEqual(sh._data_image_markdown("not-a-data-image", []), "")
        self.assertEqual(sh._data_image_markdown("data:image/png;base64,not-valid", []), "")

        object_keys = []
        with patch.object(
            sh,
            "_upload_html_image_bytes",
            return_value=ImageMarkdown(markdown="![](uploaded.jpg)", object_key="images/uploaded.jpg"),
        ) as upload:
            markdown = sh._image_markdown(
                sh.HtmlNode(tag="img", attrs={"src": "data:image/jpeg;base64,aW1n"}),
                None,
                object_keys,
                "images/",
            )

        self.assertEqual(markdown, "![](uploaded.jpg)")
        self.assertEqual(object_keys, ["images/uploaded.jpg"])
        upload.assert_called_once_with(b"img", "jpg", "images/")
        self.assertEqual(
            sh._uploaded_image_markdown(ImageMarkdown(markdown="![](plain)", object_key=""), object_keys),
            "![](plain)",
        )
        self.assertEqual(object_keys, ["images/uploaded.jpg"])

    def test_local_images_reject_unsafe_paths_and_upload_existing_files(self):
        with tempfile.TemporaryDirectory(dir=".") as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "image").write_bytes(b"no-extension")
            (base_dir / "photo.png").write_bytes(b"png-bytes")

            self.assertEqual(sh._local_image_markdown("photo.png", None, []), "")
            self.assertIsNone(sh._safe_local_image_path("https://example.test/photo.png", base_dir))
            self.assertIsNone(sh._safe_local_image_path("", base_dir))
            self.assertIsNone(sh._safe_local_image_path("../outside.png", base_dir))
            self.assertEqual(sh._local_image_markdown("missing.png", base_dir, []), "")

            calls = []

            def fake_upload(content, extension, object_key_prefix=""):
                calls.append((content, extension, object_key_prefix))
                return ImageMarkdown(markdown=f"![]({extension})", object_key=f"{object_key_prefix}{extension}")

            keys = []
            with patch.object(sh, "_upload_html_image_bytes", side_effect=fake_upload):
                self.assertEqual(sh._local_image_markdown("image", base_dir, keys, "local/"), "![](png)")
                self.assertEqual(
                    sh._image_markdown(sh.HtmlNode(tag="img", attrs={"src": "photo.png?version=1"}), base_dir, keys, "local/"),
                    "![](png)",
                )

        self.assertEqual(calls, [(b"no-extension", "png", "local/"), (b"png-bytes", "png", "local/")])
        self.assertEqual(keys, ["local/png", "local/png"])

    def test_markdown_part_joining_places_images_on_separate_lines(self):
        self.assertEqual(sh._node_markdown_text(sh.HtmlNode(tag="p", children=[]), None, []), "")
        paragraph = sh.HtmlNode(
            tag="p",
            children=[
                "A",
                sh.HtmlNode(tag="script", children=["skip"]),
                sh.HtmlNode(tag="img", attrs={"src": "https://example.test/x.png"}),
                sh.HtmlNode(tag="span", children=["B"]),
            ],
        )
        self.assertEqual(sh._node_markdown_text(paragraph, None, []), "A\n![](https://example.test/x.png) B")
        self.assertEqual(sh._join_markdown_parts([]), "")
        self.assertEqual(sh._join_markdown_parts(["A", "", "![](x)", "B"]), "A\n![](x) B")
        self.assertEqual(sh._join_markdown_parts(["![](x)", "![](y)"]), "![](x)\n![](y)")

    def test_file_entrypoints_and_image_upload_wrapper_delegate_to_helpers(self):
        with tempfile.TemporaryDirectory(dir=".") as temp_dir:
            html_path = Path(temp_dir) / "input.html"
            html_path.write_text("<body><p>From file</p></body>", encoding="utf-8")
            self.assertEqual(sh.StructuredHtmlLoader(str(html_path)).parse_blocks()[0].text, "From file")

        with patch.object(sh, "upload_image_bytes", return_value=ImageMarkdown(markdown="![](x)", object_key="key")) as upload:
            result = sh._upload_html_image_bytes(b"bytes", "png", "prefix/")
        self.assertEqual(result.object_key, "key")
        upload.assert_called_once_with(b"bytes", "png", object_key_prefix="prefix/")


if __name__ == "__main__":
    unittest.main()
