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


if __name__ == "__main__":
    unittest.main()
