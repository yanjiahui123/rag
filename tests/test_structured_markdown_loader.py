import unittest

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()


class StructuredMarkdownLoaderTests(unittest.TestCase):
    def test_parse_markdown_outputs_section_blocks_and_preserves_document_markdown(self):
        from rag_service.document_loaders.parsed_blocks import SPLIT_POLICY_MARKDOWN_HEADINGS
        from rag_service.document_loaders.structured_artifacts import STRUCTURED_MARKDOWN_ARTIFACTS_KEY
        from rag_service.document_loaders.structured_markdown_loader import StructuredMarkdownLoader

        markdown = """# Report

Intro paragraph.

```python
# Not a heading
```

## Details

- item
"""

        parsed_document = StructuredMarkdownLoader("demo.md").parse_markdown(markdown)

        self.assertEqual(parsed_document.text, "")
        self.assertEqual(len(parsed_document.blocks), 2)
        intro_block = parsed_document.blocks[0]
        detail_block = parsed_document.blocks[1]
        self.assertIn("Intro paragraph.", intro_block.text)
        self.assertIn("# Not a heading", intro_block.text)
        self.assertEqual(intro_block.split_policy, SPLIT_POLICY_MARKDOWN_HEADINGS)
        self.assertEqual(intro_block.metadata["loader"], "structured_markdown")
        self.assertEqual(intro_block.metadata["block_id"], "block_001")
        self.assertEqual(intro_block.metadata["title"], "Report")
        self.assertEqual(intro_block.metadata["headers"], ["Report"])
        self.assertEqual(detail_block.metadata["block_id"], "block_002")
        self.assertEqual(detail_block.metadata["title"], "Details")
        self.assertEqual(detail_block.metadata["headers"], ["Report", "Details"])
        self.assertEqual(detail_block.text, "## Details\n\n- item")
        artifacts = parsed_document.metadata[STRUCTURED_MARKDOWN_ARTIFACTS_KEY]
        self.assertEqual(artifacts["document_markdown"], markdown.strip())
        self.assertEqual(artifacts["tables"], [])


if __name__ == "__main__":
    unittest.main()
