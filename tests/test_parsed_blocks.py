import unittest

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()

from rag_service.document_loaders.parsed_blocks import (
    DISPLAY_METADATA_KEY,
    SPLIT_POLICY_MARKDOWN_HEADINGS,
    SPLIT_POLICY_NO_SPLIT,
    ParsedBlock,
    ParsedDocument,
    blocks_to_documents,
    documents_to_parsed_blocks,
)


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeSplitter:
    def split_text(self, text):
        return [part.strip() for part in text.split("|") if part.strip()]


class FakeSizedSplitter:
    def __init__(self, chunk_size):
        self._chunk_size = chunk_size

    def split_text(self, text):
        if len(text) <= self._chunk_size:
            return [text]
        return [text[:self._chunk_size], text[self._chunk_size:]]


class ParsedBlockTests(unittest.TestCase):
    def test_documents_to_parsed_blocks_preserves_content_and_metadata(self):
        source_metadata = {
            "source": "demo.xlsx",
            DISPLAY_METADATA_KEY: {"type": "table", "format": "html", "content": "<table></table>"},
        }
        docs = [FakeDocument("hello", source_metadata)]

        blocks = documents_to_parsed_blocks(docs)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "hello")
        self.assertEqual(blocks[0].block_type, "text")
        self.assertEqual(blocks[0].metadata[DISPLAY_METADATA_KEY]["type"], "table")
        self.assertIsNot(blocks[0].metadata, source_metadata)

    def test_blocks_to_documents_splits_text_blocks_and_keeps_metadata_isolated(self):
        block = ParsedBlock(text="alpha | beta", metadata={"source": "demo.txt", "headers": ["Intro"]})

        docs = blocks_to_documents([block], FakeDocument, text_splitter=FakeSplitter())

        self.assertEqual([doc.page_content for doc in docs], ["alpha", "beta"])
        self.assertEqual(docs[0].metadata["headers"], ["Intro"])
        docs[0].metadata["headers"].append("Changed")
        self.assertEqual(block.metadata["headers"], ["Intro"])

    def test_structured_text_splits_use_section_aware_retrieval_text(self):
        block = ParsedBlock(
            text="alpha | beta",
            metadata={"source": "demo.md", "loader": "structured_markdown", "headers": ["Report", "Risk"]},
        )

        docs = blocks_to_documents([block], FakeDocument, text_splitter=FakeSplitter())

        self.assertEqual([doc.metadata["content_text"] for doc in docs], ["alpha", "beta"])
        self.assertEqual(docs[0].metadata["section_path"], "Report > Risk")
        self.assertEqual(docs[0].metadata["section_title"], "Risk")
        self.assertEqual(docs[0].page_content, "section: Report > Risk\n\nalpha")
        self.assertEqual(docs[1].metadata["retrieval_text"], "section: Report > Risk\n\nbeta")

    def test_table_blocks_keep_existing_retrieval_text_without_extra_section_prefix(self):
        block = ParsedBlock(
            text="section: Report\n\ntable: Sales",
            metadata={"source": "demo.html", "loader": "structured_html", "headers": ["Report"]},
            block_type="table",
            split_policy=SPLIT_POLICY_NO_SPLIT,
        )

        docs = blocks_to_documents([block], FakeDocument, text_splitter=FakeSplitter())

        self.assertEqual(docs[0].page_content, "section: Report\n\ntable: Sales")
        self.assertNotIn("content_text", docs[0].metadata)
        self.assertNotIn("retrieval_text", docs[0].metadata)

    def test_blocks_to_documents_does_not_split_no_split_blocks(self):
        block = ParsedBlock(
            text="alpha | beta",
            metadata={"source": "demo.txt"},
            split_policy=SPLIT_POLICY_NO_SPLIT,
        )

        docs = blocks_to_documents([block], FakeDocument, text_splitter=FakeSplitter())

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "alpha | beta")
        self.assertEqual(docs[0].metadata["split_policy"], SPLIT_POLICY_NO_SPLIT)

    def test_blocks_to_documents_splits_markdown_by_deepest_heading_policy(self):
        block = ParsedBlock(
            text="# A\n\ntext a\n\n# B\n\ntext b",
            metadata={"source": "demo.docx"},
            split_policy=SPLIT_POLICY_MARKDOWN_HEADINGS,
        )

        docs = blocks_to_documents([block], FakeDocument)

        self.assertEqual([doc.page_content for doc in docs], ["# A\n\ntext a", "# B\n\ntext b"])

    def test_blocks_to_documents_merges_short_markdown_sections_before_length_split(self):
        block = ParsedBlock(
            text="# A\n\nalpha\n\n# B\n\nbeta\n\n# C\n\ngamma",
            metadata={"source": "demo.docx"},
            split_policy=SPLIT_POLICY_MARKDOWN_HEADINGS,
        )

        docs = blocks_to_documents([block], FakeDocument, text_splitter=FakeSizedSplitter(chunk_size=25))

        self.assertEqual(
            [doc.page_content for doc in docs],
            ["# A\n\nalpha\n\n# B\n\nbeta", "# C\n\ngamma"],
        )

    def test_parsed_document_full_text_converts_to_one_block(self):
        parsed_document = ParsedDocument(
            text="# A\n\ntext",
            metadata={"source": "demo.docx"},
            split_policy=SPLIT_POLICY_MARKDOWN_HEADINGS,
        )

        blocks = parsed_document.to_blocks()

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "# A\n\ntext")
        self.assertEqual(blocks[0].split_policy, SPLIT_POLICY_MARKDOWN_HEADINGS)

    def test_parsed_document_legacy_blocks_remain_blocks(self):
        parsed_document = ParsedDocument(
            blocks=[ParsedBlock(text="section", metadata={"source": "demo.txt"})],
            metadata={"source": "demo.txt"},
        )

        blocks = parsed_document.to_blocks()

        self.assertEqual([block.text for block in blocks], ["section"])
        self.assertIsNot(blocks[0], parsed_document.blocks[0])


if __name__ == "__main__":
    unittest.main()
