import unittest

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()


class FakeStyle:
    def __init__(self, name):
        self.name = name


class FakeParagraph:
    def __init__(self, text, style_name="Normal", xml=""):
        self.text = text
        self.style = FakeStyle(style_name)
        self._p = type("FakeXml", (), {"xml": xml})()


class FakeCell:
    def __init__(self, text):
        self.text = text


class FakeRow:
    def __init__(self, cells):
        self.cells = cells


class FakeTable:
    def __init__(self, rows):
        self.rows = rows


class FakeDocument:
    def __init__(self, blocks, related_parts=None):
        self.blocks = blocks
        self.part = type("FakePart", (), {"related_parts": related_parts or {}})()


class FakeImagePart:
    def __init__(self, partname="/word/media/image1.png", blob=b"image-bytes"):
        self.partname = partname
        self.blob = blob
        self.content_type = "image/png"


class StructuredDocxLoaderTests(unittest.TestCase):
    def test_parse_document_returns_markdown_text_block(self):
        from rag_service.document_loaders.parsed_blocks import SPLIT_POLICY_MARKDOWN_HEADINGS
        from rag_service.document_loaders.structured_docx_loader import StructuredDocxLoader

        document = FakeDocument(
            [
                FakeParagraph("Asset List", "Heading 1"),
                FakeParagraph("First paragraph"),
                FakeParagraph("Second paragraph"),
                FakeParagraph("Risk List", "Heading 1"),
                FakeParagraph("Risk paragraph"),
            ]
        )

        parsed_document = StructuredDocxLoader("demo.docx").parse_document(document)

        self.assertEqual(parsed_document.text, "")
        self.assertEqual(len(parsed_document.blocks), 2)
        self.assertEqual(
            [block.text for block in parsed_document.blocks],
            ["# Asset List\n\nFirst paragraph\n\nSecond paragraph", "# Risk List\n\nRisk paragraph"],
        )
        self.assertEqual(parsed_document.blocks[0].split_policy, SPLIT_POLICY_MARKDOWN_HEADINGS)
        self.assertEqual(parsed_document.blocks[0].metadata["headers"], ["Asset List"])
        self.assertEqual(parsed_document.blocks[1].metadata["headers"], ["Risk List"])

    def test_parse_document_keeps_table_artifacts_out_of_block_metadata(self):
        from rag_service.document_loaders.parsed_blocks import BLOCK_TYPE_TABLE, SPLIT_POLICY_NO_SPLIT
        from rag_service.document_loaders.structured_artifacts import STRUCTURED_DOCX_ARTIFACTS_KEY
        from rag_service.document_loaders.structured_docx_loader import StructuredDocxLoader

        table = FakeTable(
            [
                FakeRow([FakeCell("Name"), FakeCell("Value")]),
                FakeRow([FakeCell("A"), FakeCell("1")]),
            ]
        )
        document = FakeDocument([FakeParagraph("Asset List", "Heading 1"), FakeParagraph("Intro"), table])

        parsed_document = StructuredDocxLoader("demo.docx").parse_document(document)

        self.assertEqual(len(parsed_document.blocks), 2)
        text_block, table_block = parsed_document.blocks
        self.assertIn("# Asset List", text_block.text)
        self.assertIn("Intro", text_block.text)
        self.assertEqual(table_block.block_type, BLOCK_TYPE_TABLE)
        self.assertEqual(table_block.split_policy, SPLIT_POLICY_NO_SPLIT)
        self.assertIn("fields:\n- Name\n- Value", table_block.text)
        self.assertIn("1. Name=A; Value=1", table_block.text)
        self.assertNotIn("|", table_block.text)
        self.assertEqual(table_block.metadata["table_id"], "table_001")
        self.assertEqual(table_block.metadata["table"]["flatten_headers"], ["Name", "Value"])
        self.assertNotIn("display", table_block.metadata)
        self.assertNotIn("expanded_rows", table_block.metadata["table"])
        self.assertNotIn("cell_spans", table_block.metadata["table"])

        artifacts = parsed_document.metadata[STRUCTURED_DOCX_ARTIFACTS_KEY]
        self.assertIn("# Asset List", artifacts["document_markdown"])
        self.assertEqual(table_block.text, artifacts["tables"][0]["llm_markdown"])
        self.assertIn("<table>", artifacts["tables"][0]["html"])

    def test_parse_document_preserves_image_paragraph_as_markdown_url(self):
        from unittest.mock import patch

        from rag_service.document_loaders.image_markdown import ImageMarkdown
        import rag_service.document_loaders.structured_docx_loader as structured_docx_loader
        from rag_service.document_loaders.structured_artifacts import STRUCTURED_DOCX_ARTIFACTS_KEY
        from rag_service.document_loaders.structured_docx_loader import StructuredDocxLoader

        image_xml = '<w:p><pic:pic><a:blip r:embed="rId9"/></pic:pic></w:p>'
        image_part = FakeImagePart()
        document = FakeDocument(
            [
                FakeParagraph("Asset List", "Heading 1"),
                FakeParagraph("", xml=image_xml),
            ],
            related_parts={"rId9": image_part},
        )
        with patch.object(
            structured_docx_loader,
            "_upload_image_part",
            lambda part, object_key_prefix="": ImageMarkdown(
                markdown="![](https://example.test/image1.png)",
                object_key="image-key-1",
            ),
            create=True,
        ):
            parsed_document = StructuredDocxLoader("demo.docx").parse_document(document)

        self.assertEqual(len(parsed_document.blocks), 1)
        self.assertEqual(parsed_document.blocks[0].text, "# Asset List\n\n![](https://example.test/image1.png)")
        artifacts = parsed_document.metadata[STRUCTURED_DOCX_ARTIFACTS_KEY]
        self.assertIn("![](https://example.test/image1.png)", artifacts["document_markdown"])
        self.assertEqual(artifacts["image_object_keys"], ["image-key-1"])

    def test_parse_document_uploads_images_under_configured_prefix(self):
        from unittest.mock import patch

        from rag_service.document_loaders.image_markdown import ImageMarkdown
        import rag_service.document_loaders.structured_docx_loader as structured_docx_loader
        from rag_service.document_loaders.structured_artifacts import STRUCTURED_DOCX_ARTIFACTS_KEY
        from rag_service.document_loaders.structured_docx_loader import StructuredDocxLoader

        calls = []
        image_xml = '<w:p><pic:pic><a:blip r:embed="rId9"/></pic:pic></w:p>'
        document = FakeDocument(
            [FakeParagraph("Report", "Heading 1"), FakeParagraph("", xml=image_xml)],
            related_parts={"rId9": FakeImagePart()},
        )

        def fake_upload(content, extension, object_key_prefix=""):
            calls.append((content, extension, object_key_prefix))
            return ImageMarkdown(markdown="![](image)", object_key=object_key_prefix + "image.png")

        with patch.object(structured_docx_loader, "upload_image_bytes", fake_upload):
            parsed_document = StructuredDocxLoader(
                "demo.docx",
                image_upload_prefix="asset/doc/artifacts/structured_docx/images/",
            ).parse_document(document)

        self.assertEqual(calls, [(b"image-bytes", "png", "asset/doc/artifacts/structured_docx/images/")])
        artifacts = parsed_document.metadata[STRUCTURED_DOCX_ARTIFACTS_KEY]
        self.assertEqual(artifacts["image_object_keys"], ["asset/doc/artifacts/structured_docx/images/image.png"])


if __name__ == "__main__":
    unittest.main()
