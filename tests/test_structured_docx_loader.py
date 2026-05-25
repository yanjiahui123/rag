import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()

from rag_service.document_loaders.image_markdown import ImageMarkdown
from rag_service.document_loaders.parsed_blocks import BLOCK_TYPE_TABLE, SPLIT_POLICY_NO_SPLIT
from rag_service.document_loaders.structured_artifacts import STRUCTURED_DOCX_ARTIFACTS_KEY
from rag_service.document_loaders import structured_docx_loader as sd


class FakeParagraph:
    def __init__(self, text="", style_name="Normal", xml=""):
        self.text = text
        self.style = SimpleNamespace(name=style_name)
        self._p = SimpleNamespace(xml=xml)


class FakeCell:
    def __init__(self, text):
        self.text = text


class FakeRow:
    def __init__(self, *values):
        self.cells = [FakeCell(value) for value in values]


class FakeTable:
    def __init__(self, rows):
        self.rows = rows


class FakeDocument:
    def __init__(self, blocks=None, related_parts=None):
        self.blocks = list(blocks or [])
        self.part = SimpleNamespace(related_parts=related_parts or {})


class FakeImagePart:
    def __init__(self, partname="/word/media/image.png", content_type="image/png", blob=b"image-bytes"):
        self.partname = partname
        self.content_type = content_type
        self.blob = blob


class StructuredDocxLoaderBranchTests(unittest.TestCase):
    def test_parse_document_handles_empty_headings_text_and_tables(self):
        loader = sd.StructuredDocxLoader("demo.docx", source="source.docx")

        empty = loader.parse_document(FakeDocument())
        self.assertEqual(empty.blocks, [])
        self.assertEqual(empty.metadata["loader"], "structured_docx")
        self.assertNotIn(STRUCTURED_DOCX_ARTIFACTS_KEY, empty.metadata)
        with patch.object(loader, "_open_document", return_value=FakeDocument([FakeParagraph("opened")])):
            self.assertEqual(loader.parse_blocks()[0].text, "opened")

        table = FakeTable([FakeRow("Name", "Value"), FakeRow("A", "1")])
        parsed = loader.parse_document(
            FakeDocument(
                [
                    FakeParagraph(""),
                    FakeParagraph("Preamble"),
                    FakeParagraph("Report", "Heading 1"),
                    FakeParagraph("Body"),
                    table,
                    FakeParagraph("After table"),
                ]
            )
        )

        self.assertEqual([block.text for block in parsed.blocks[:2]], ["Preamble", "# Report\n\nBody"])
        self.assertEqual(parsed.blocks[2].block_type, BLOCK_TYPE_TABLE)
        self.assertEqual(parsed.blocks[2].split_policy, SPLIT_POLICY_NO_SPLIT)
        self.assertEqual(parsed.blocks[2].metadata["table_id"], "table_001")
        self.assertEqual(parsed.blocks[3].text, "After table")
        artifacts = parsed.metadata[STRUCTURED_DOCX_ARTIFACTS_KEY]
        self.assertEqual(len(artifacts["tables"]), 1)
        self.assertIn("# Report", artifacts["document_markdown"])

    def test_paragraph_paths_skip_empty_content_and_append_heading_images(self):
        loader = sd.StructuredDocxLoader("demo.docx")
        elements = []

        self.assertEqual(loader._append_paragraph_element(elements, FakeParagraph(), [], 0, FakeDocument()), [])
        self.assertEqual(elements, [])
        self.assertEqual(loader._append_paragraph_element(elements, FakeParagraph("body"), [], 1, FakeDocument()), [])
        self.assertEqual(elements[0].text, "body")

        heading_elements = []
        with patch.object(loader, "_paragraph_image_markdown", return_value=["![](diagram.png)"]):
            headers = loader._append_paragraph_element(
                heading_elements,
                FakeParagraph("Risk", "Heading 2"),
                ["Report"],
                2,
                FakeDocument(),
            )

        self.assertEqual(headers, ["Report", "Risk"])
        self.assertEqual([element.text for element in heading_elements], ["## Risk", "![](diagram.png)"])
        blocks = []
        loader._append_text_block(blocks, [])
        self.assertEqual(blocks, [])

    def test_image_relation_paths_skip_non_images_and_record_uploaded_images(self):
        xml = (
            '<w:p><pic:pic><a:blip r:embed="missing"/><a:blip r:embed="text"/>'
            '<a:blip r:embed="blank"/><a:blip r:embed="good"/><a:blip r:embed="good"/></pic:pic></w:p>'
        )
        blank_part = FakeImagePart("/word/media/blank.png")
        good_part = FakeImagePart("/word/media/good.png")
        document = FakeDocument(
            related_parts={
                "text": FakeImagePart("/word/media/file.txt", "text/plain"),
                "blank": blank_part,
                "good": good_part,
            }
        )
        loader = sd.StructuredDocxLoader("demo.docx", image_upload_prefix="images/")
        uploads = []

        def fake_upload(part, object_key_prefix=""):
            uploads.append((part.partname, object_key_prefix))
            if part is blank_part:
                return ImageMarkdown(markdown="", object_key="")
            return ImageMarkdown(markdown="![](good.png)", object_key="images/good.png")

        self.assertEqual(loader._paragraph_image_markdown(FakeParagraph(xml="<w:p/>"), document), [])
        with patch.object(sd, "_upload_image_part", side_effect=fake_upload):
            links = loader._paragraph_image_markdown(FakeParagraph(xml=xml), document)

        self.assertEqual(links, ["![](good.png)"])
        self.assertEqual(loader.image_object_keys, ["images/good.png"])
        self.assertEqual(uploads, [("/word/media/blank.png", "images/"), ("/word/media/good.png", "images/")])

    def test_heading_relationship_and_image_helpers_cover_fallbacks(self):
        loader = sd.StructuredDocxLoader("demo.docx")
        xml = '<pic:pic><a:blip r:embed="rId1"/><a:blip r:id="rId1"/></pic:pic>'

        self.assertEqual(loader._heading_level(FakeParagraph("Title", "Title")), 1)
        self.assertEqual(loader._heading_level(FakeParagraph("Third", "Heading 3")), 3)
        self.assertIsNone(loader._heading_level(FakeParagraph("Body", "Normal")))
        self.assertIsNone(loader._parse_heading_level("Heading Bad"))
        self.assertEqual(loader._replace_header(["Report"], 3, "Detail"), ["Report", "", "Detail"])
        self.assertTrue(sd._contains_image_marker(xml))
        self.assertFalse(sd._contains_image_marker("<w:p/>"))
        self.assertEqual(sd._image_relation_ids(xml), ["rId1"])

        self.assertEqual(sd._related_parts(FakeDocument(related_parts={"doc": 1}), FakeParagraph()), {"doc": 1})
        owned = FakeParagraph()
        owned.part = SimpleNamespace(related_parts={"block": 2})
        self.assertEqual(sd._related_parts(object(), owned), {"block": 2})
        self.assertEqual(sd._related_parts(object(), object()), {})
        fake_section_loader = ModuleType("rag_service.document_loaders.docx_section_loader")

        def iter_fallback_blocks(_document):
            return ["fallback"]

        fake_section_loader.iter_block_items = iter_fallback_blocks
        with patch.dict(sys.modules, {"rag_service.document_loaders.docx_section_loader": fake_section_loader}):
            self.assertEqual(loader._iter_blocks(object()), ["fallback"])

        self.assertFalse(sd._is_image_part(None))
        self.assertTrue(sd._is_image_part(FakeImagePart()))
        self.assertTrue(sd._is_image_part(FakeImagePart("/word/media/vector.svg", "application/octet-stream")))
        self.assertFalse(sd._is_image_part(FakeImagePart("/word/media/file.txt", "text/plain")))
        self.assertEqual(sd._combine_text_and_images("Text", ["![](x)"]), "Text\n![](x)")
        self.assertEqual(sd._combine_text_and_images("", []), "")

    def test_upload_image_part_forwards_prefix_and_extension(self):
        with patch.object(sd, "upload_image_bytes", return_value=ImageMarkdown(markdown="![](x)", object_key="prefix/x.png")) as upload:
            result = sd._upload_image_part(FakeImagePart(), "prefix/")

        self.assertEqual(result.object_key, "prefix/x.png")
        upload.assert_called_once_with(b"image-bytes", "png", object_key_prefix="prefix/")

        opened = []
        fake_docx = ModuleType("docx")

        def open_document(path):
            opened.append(path)
            return "opened document"

        fake_docx.Document = open_document
        with patch.dict(sys.modules, {"docx": fake_docx}):
            self.assertEqual(sd.StructuredDocxLoader("input.docx")._open_document(), "opened document")
        self.assertEqual(opened, ["input.docx"])


if __name__ == "__main__":
    unittest.main()
