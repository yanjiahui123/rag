import unittest

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()


class ImageMarkdownTests(unittest.TestCase):
    def test_image_markdown_is_pydantic_model(self):
        from pydantic import BaseModel

        from rag_service.document_loaders.image_markdown import ImageMarkdown

        self.assertTrue(issubclass(ImageMarkdown, BaseModel))
        payload = ImageMarkdown(markdown="![](url)", object_key="key", url="url")
        self.assertEqual(payload.markdown, "![](url)")
        self.assertEqual(payload.object_key, "key")
        self.assertEqual(payload.url, "url")

    def test_upload_image_bytes_uses_optional_object_key_prefix(self):
        from unittest.mock import patch

        from rag_service.document_loaders import image_markdown

        calls = []

        def fake_upload(object_key, content):
            calls.append((object_key, content))
            return object_key

        with patch.dict(
            "sys.modules",
            {
                "rag_service.utils.his_util.obs_util": type(
                    "FakeObs",
                    (),
                    {"upload_file_as_bytes": staticmethod(fake_upload)},
                ),
            },
        ), patch.object(image_markdown.uuid, "uuid4", lambda: "image-id"):
            result = image_markdown.upload_image_bytes(
                b"image",
                ".PNG",
                object_key_prefix="asset/doc/artifacts/structured_docx/images/",
            )

        self.assertEqual(calls, [("asset/doc/artifacts/structured_docx/images/image-id.png", b"image")])
        self.assertEqual(result.object_key, "asset/doc/artifacts/structured_docx/images/image-id.png")


if __name__ == "__main__":
    unittest.main()
