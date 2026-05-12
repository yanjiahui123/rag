import unittest

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()


class HtmlTableParserTests(unittest.TestCase):
    def test_parse_complex_header_table_to_compact_llm_text_and_artifacts(self):
        from rag_service.document_loaders.table.html_parser import HtmlTableParser

        html = """
        <table>
          <caption>Sales</caption>
          <tr><th rowspan="2">Region</th><th colspan="2">Sales</th></tr>
          <tr><th>Q1</th><th>Q2</th></tr>
          <tr><td>East</td><td>100</td><td></td></tr>
        </table>
        """

        table = HtmlTableParser().parse_html(
            html,
            metadata={"source": "demo.html"},
        )

        self.assertEqual(table.title, "Sales")
        self.assertEqual(table.headers, ["Region", "Sales/Q1", "Sales/Q2"])
        self.assertEqual(table.rows, [["East", "100", ""]])
        self.assertIn('rowspan="2"', table.display_html)
        self.assertIn('colspan="2"', table.display_html)
        self.assertIn("fields:\n- Region\n- Sales/Q1\n- Sales/Q2", table.to_llm_text())
        self.assertIn("1. Region=East; Sales/Q1=100", table.to_llm_text())
        self.assertNotIn("Sales/Q2=", table.to_llm_text())


if __name__ == "__main__":
    unittest.main()
