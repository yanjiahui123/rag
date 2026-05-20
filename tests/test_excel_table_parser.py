import unittest
import time

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()


class ExcelTableParserTests(unittest.TestCase):
    def test_parse_detects_description_multi_headers_and_multiple_tables(self):
        from rag_service.document_loaders.table.excel_parser import ExcelTableParser

        grid = [
            ["Sales report, unit: USD", "", "", ""],
            ["Region", "Sales", "Sales", "Owner"],
            ["", "Q1", "Q2", ""],
            ["East", 100, 120, "Ann"],
            ["West", 80, 90, "Ben"],
            ["", "", "", ""],
            ["Inventory snapshot", "", "", ""],
            ["Product", "Stock", "", ""],
            ["A", 12, "", ""],
            ["B", 8, "", ""],
        ]

        tables = ExcelTableParser().parse_grid(grid, sheet_name="Sheet1", source="demo.xlsx")

        self.assertEqual(len(tables), 2)
        self.assertEqual(tables[0].description, "Sales report, unit: USD")
        self.assertEqual(tables[0].header_rows, [["Region", "Sales", "Sales", "Owner"], ["", "Q1", "Q2", ""]])
        self.assertEqual(tables[0].flatten_headers, ["Region", "Sales/Q1", "Sales/Q2", "Owner"])
        self.assertEqual(tables[0].cell_range, "A1:D5")
        self.assertIn('<th colspan="2">Sales</th>', tables[0].display_html)
        self.assertEqual(tables[1].description, "Inventory snapshot")
        self.assertEqual(tables[1].flatten_headers, ["Product", "Stock"])

    def test_parse_expands_sparse_merged_header_cells(self):
        from rag_service.document_loaders.table.excel_parser import ExcelTableParser

        grid = [
            ["Region", "Sales", "", "Owner"],
            ["", "Q1", "Q2", ""],
            ["East", 100, 120, "Ann"],
        ]

        table = ExcelTableParser().parse_grid(grid, sheet_name="Sheet1", source="demo.xlsx")[0]

        self.assertEqual(table.header_rows, [["Region", "Sales", "Sales", "Owner"], ["", "Q1", "Q2", ""]])
        self.assertEqual(table.flatten_headers, ["Region", "Sales/Q1", "Sales/Q2", "Owner"])
        self.assertIn('<th colspan="2">Sales</th>', table.display_html)

    def test_parse_uses_explicit_merge_ranges_for_single_row_headers(self):
        from rag_service.document_loaders.table.excel_parser import ExcelTableParser

        grid = [
            ["Product", "Metrics", "", "Owner"],
            ["A", 10, 20, "Ann"],
        ]

        table = ExcelTableParser().parse_grid(
            grid,
            sheet_name="Sheet1",
            source="demo.xlsx",
            merge_ranges=[(0, 1, 1, 3)],
        )[0]

        self.assertEqual(table.header_rows, [["Product", "Metrics", "Metrics", "Owner"]])
        self.assertEqual(table.flatten_headers, ["Product", "Metrics", "Metrics", "Owner"])
        self.assertIn('<th colspan="2">Metrics</th>', table.display_html)

    def test_to_chunks_keeps_headers_and_merges_small_tail(self):
        from rag_service.document_loaders.table.excel_parser import ExcelTableParser

        rows = [["R" + str(i), "value " + str(i)] for i in range(12)]
        grid = [["Region", "Value"]] + rows
        table = ExcelTableParser().parse_grid(grid, sheet_name="Sheet1", source="demo.xlsx")[0]

        chunks = table.to_llm_chunks(max_chars=170, min_tail_chars=80)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 170)
            self.assertIn("字段:", chunk.text)
            self.assertIn("- Region", chunk.text)
            self.assertIn("- Value", chunk.text)
        self.assertEqual(chunks[-1].row_range[1], len(rows))
        self.assertGreaterEqual(len(chunks[-1].text), 80)

    def test_to_chunks_scales_near_linearly_when_table_fits_one_chunk(self):
        from rag_service.document_loaders.table.excel_parser import ExcelTable

        def build_table(row_count):
            return ExcelTable(
                table_id="table_001",
                source="large.xlsx",
                sheet_name="Sheet1",
                cell_range=f"A1:C{row_count + 1}",
                flatten_headers=["A", "B", "C"],
                rows=[[str(index), "x" * 20, "y" * 20] for index in range(row_count)],
            )

        def elapsed_for(row_count):
            start = time.perf_counter()
            chunks = build_table(row_count).to_llm_chunks(max_chars=10**9)
            elapsed = time.perf_counter() - start
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].row_range, (0, row_count))
            return elapsed

        small_elapsed = elapsed_for(1000)
        large_elapsed = elapsed_for(4000)

        self.assertLess(large_elapsed, small_elapsed * 8 + 0.1)


if __name__ == "__main__":
    unittest.main()
