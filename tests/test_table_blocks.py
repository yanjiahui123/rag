import unittest

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()

from rag_service.document_loaders.parsed_blocks import BLOCK_TYPE_TABLE, SPLIT_POLICY_TABLE_ROWS
from rag_service.document_loaders.table.models import TableBlock


class TableBlockTests(unittest.TestCase):
    def test_table_block_to_parsed_block_adds_table_and_display_metadata(self):
        table = TableBlock(
            title="Sales",
            source_type="excel",
            headers=["Region", "Q1"],
            rows=[["East", "100"], ["West", "80"]],
            display_html="<table><tr><th>Region</th></tr></table>",
            metadata={"source": "demo.xlsx", "sheet_name": "Sheet1", "cell_range": "A1:B3"},
        )

        block = table.to_parsed_block()

        self.assertEqual(block.block_type, BLOCK_TYPE_TABLE)
        self.assertEqual(block.split_policy, SPLIT_POLICY_TABLE_ROWS)
        self.assertIn("| Region | Q1 |", block.text)
        self.assertIn("| East | 100 |", block.text)
        self.assertEqual(block.metadata["display"]["format"], "html")
        self.assertEqual(block.metadata["table"]["title"], "Sales")
        self.assertEqual(block.metadata["table"]["sheet_name"], "Sheet1")
        self.assertEqual(block.metadata["table"]["flatten_headers"], ["Region", "Q1"])


if __name__ == "__main__":
    unittest.main()
