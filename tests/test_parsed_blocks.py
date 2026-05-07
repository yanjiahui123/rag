import unittest

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()

from rag_service.document_loaders.parsed_blocks import (
    DISPLAY_METADATA_KEY,
    SPLIT_POLICY_NO_SPLIT,
    ParsedBlock,
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


if __name__ == "__main__":
    unittest.main()
