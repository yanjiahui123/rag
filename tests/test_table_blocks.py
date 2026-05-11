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
        self.assertIn("fields:", block.text)
        self.assertIn("- Region", block.text)
        self.assertIn("1. Region=East; Q1=100", block.text)
        self.assertEqual(block.metadata["display"]["format"], "html")
        self.assertEqual(block.metadata["table"]["title"], "Sales")
        self.assertEqual(block.metadata["table"]["sheet_name"], "Sheet1")
        self.assertEqual(block.metadata["table"]["flatten_headers"], ["Region", "Q1"])

    def test_table_block_llm_text_skips_blank_cells_and_markdown_table_noise(self):
        table = TableBlock(
            title="Sales",
            source_type="docx_table",
            headers=["Region", "Q1", "Q2", ""],
            rows=[["East", "100", "", ""], ["West", "", "80", "owner"]],
        )

        text = table.to_llm_text(section_headers=["Asset List"])

        self.assertIn("section: Asset List", text)
        self.assertIn("fields:\n- Region\n- Q1\n- Q2\n- Column 4", text)
        self.assertIn("1. Region=East; Q1=100", text)
        self.assertIn("2. Region=West; Q2=80; Column 4=owner", text)
        self.assertNotIn("|", text)
        self.assertNotIn("Q2=", text.splitlines()[-2])


if __name__ == "__main__":
    unittest.main()
