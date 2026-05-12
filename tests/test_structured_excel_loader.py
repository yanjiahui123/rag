import unittest
import sys
import types

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()


class StructuredExcelLoaderTests(unittest.TestCase):
    def test_parse_sheets_outputs_llm_chunks_and_excel_artifacts(self):
        from rag_service.document_loaders.parsed_blocks import BLOCK_TYPE_TABLE, SPLIT_POLICY_NO_SPLIT
        from rag_service.document_loaders.structured_artifacts import STRUCTURED_EXCEL_ARTIFACTS_KEY
        from rag_service.document_loaders.structured_excel_loader import StructuredExcelLoader

        rows = [["R" + str(i), "value " + str(i)] for i in range(12)]
        sheets = {"Sheet1": [["Report description", ""], ["Region", "Value"]] + rows}

        parsed_document = StructuredExcelLoader("demo.xlsx", chunk_size=170).parse_sheets(sheets)

        self.assertGreater(len(parsed_document.blocks), 1)
        first_block = parsed_document.blocks[0]
        self.assertEqual(first_block.block_type, BLOCK_TYPE_TABLE)
        self.assertEqual(first_block.split_policy, SPLIT_POLICY_NO_SPLIT)
        self.assertEqual(first_block.metadata["table_id"], "table_001")
        self.assertEqual(first_block.metadata["row_range"][0], 0)
        self.assertIn("字段:", first_block.text)
        self.assertIn("- Region", first_block.text)
        self.assertIn("Region=R0", first_block.text)
        self.assertNotIn("<table>", first_block.text)
        self.assertNotIn("display", first_block.metadata)

        artifacts = parsed_document.metadata[STRUCTURED_EXCEL_ARTIFACTS_KEY]
        self.assertIn("Report description", artifacts["document_markdown"])
        self.assertIn("<table>", artifacts["tables"][0]["html"])
        self.assertIn("Region=R0", artifacts["tables"][0]["llm_markdown"])

    def test_parse_to_document_preserves_openpyxl_merged_headers(self):
        from rag_service.document_loaders.structured_artifacts import STRUCTURED_EXCEL_ARTIFACTS_KEY
        from rag_service.document_loaders.structured_excel_loader import StructuredExcelLoader

        previous_modules = install_fake_openpyxl_module()
        try:
            parsed_document = StructuredExcelLoader("merged.xlsx").parse_to_document()
        finally:
            restore_modules(previous_modules)

        table = parsed_document.metadata[STRUCTURED_EXCEL_ARTIFACTS_KEY]["tables"][0]
        self.assertIn("- Metrics", parsed_document.blocks[0].text)
        self.assertIn('<th colspan="2">Metrics</th>', table["html"])


def install_fake_openpyxl_module():
    module_names = ["openpyxl", "openpyxl.reader", "openpyxl.reader.excel"]
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    openpyxl = types.ModuleType("openpyxl")
    reader = types.ModuleType("openpyxl.reader")
    excel = types.ModuleType("openpyxl.reader.excel")
    excel.load_workbook = lambda *args, **kwargs: FakeWorkbook()
    sys.modules["openpyxl"] = openpyxl
    sys.modules["openpyxl.reader"] = reader
    sys.modules["openpyxl.reader.excel"] = excel
    return previous_modules


def restore_modules(previous_modules):
    for name, module in previous_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class FakeWorkbook:
    worksheets = []

    def __init__(self):
        self.worksheets = [FakeWorksheet()]


class FakeWorksheet:
    title = "Sheet1"
    merged_cells = types.SimpleNamespace(ranges=[types.SimpleNamespace(min_row=1, min_col=2, max_row=1, max_col=3)])

    def iter_rows(self, values_only=True):
        return iter([
            ["Product", "Metrics", None, "Owner"],
            ["A", 10, 20, "Ann"],
        ])


if __name__ == "__main__":
    unittest.main()
