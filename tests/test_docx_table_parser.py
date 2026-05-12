import unittest
import types

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()

MISSING = object()


class FakeCell:
    def __init__(self, text, rowspan=1, colspan=1):
        self.text = text
        self.rowspan = rowspan
        self.colspan = colspan


class FakeRow:
    def __init__(self, cells):
        self.cells = cells


class FakeTable:
    def __init__(self, rows):
        self.rows = rows


class FakeOoxmlValue:
    def __init__(self, value):
        self.val = value


class FakeOoxmlTcPr:
    def __init__(self, colspan=1, vmerge=MISSING):
        self.gridSpan = FakeOoxmlValue(colspan) if colspan > 1 else None
        self.vMerge = None if vmerge is MISSING else FakeOoxmlValue(vmerge)


class FakeOoxmlTc:
    def __init__(self, text, colspan=1, vmerge=MISSING):
        self.text = text
        self.tcPr = FakeOoxmlTcPr(colspan, vmerge)

    def xpath(self, _pattern):
        return [types.SimpleNamespace(text=self.text)] if self.text else []


class FakeOoxmlTrPr:
    def __init__(self, grid_before=0, grid_after=0):
        self.gridBefore = FakeOoxmlValue(grid_before) if grid_before else None
        self.gridAfter = FakeOoxmlValue(grid_after) if grid_after else None


class FakeOoxmlTr:
    def __init__(self, cells, grid_before=0, grid_after=0):
        self.tc_lst = cells
        self.trPr = FakeOoxmlTrPr(grid_before, grid_after)


class FakeOoxmlTbl:
    def __init__(self, rows):
        self.tr_lst = rows


class FakeOoxmlTable:
    def __init__(self, rows):
        self._tbl = FakeOoxmlTbl(rows)


class DocxTableParserTests(unittest.TestCase):
    def test_parse_keeps_merge_spans_for_display_and_flattens_search_rows(self):
        from rag_service.document_loaders.table.docx_parser import DocxTableParser

        table = FakeTable(
            [
                FakeRow([FakeCell("Region", rowspan=2), FakeCell("Group", colspan=2)]),
                FakeRow([FakeCell("Revenue"), FakeCell("Cost")]),
            ]
        )

        table_block = DocxTableParser().parse(table, metadata={"source": "demo.docx"})
        parsed_block = table_block.to_parsed_block()

        self.assertEqual(table_block.headers, ["Region", "Group", "Group"])
        self.assertEqual(table_block.rows, [["Region", "Revenue", "Cost"]])
        self.assertIn('rowspan="2"', parsed_block.metadata["display"]["content"])
        self.assertIn('colspan="2"', parsed_block.metadata["display"]["content"])
        self.assertIn("fields:\n- Region\n- Group\n- Group", parsed_block.text)
        self.assertIn("1. Region=Region; Group=Revenue; Group=Cost", parsed_block.text)
        self.assertNotIn("|", parsed_block.text)

    def test_parse_ooxml_table_keeps_cell_spans_and_expanded_grid(self):
        from rag_service.document_loaders.table.docx_parser import DocxTableParser

        table = FakeOoxmlTable(
            [
                FakeOoxmlTr([FakeOoxmlTc("Region", vmerge="restart"), FakeOoxmlTc("Metric", colspan=2)]),
                FakeOoxmlTr([FakeOoxmlTc("", vmerge=None), FakeOoxmlTc("Revenue"), FakeOoxmlTc("Cost")]),
                FakeOoxmlTr([FakeOoxmlTc("East"), FakeOoxmlTc("100"), FakeOoxmlTc("60")]),
            ]
        )

        table_block = DocxTableParser().parse(table, metadata={"source": "demo.docx"})
        parsed_block = table_block.to_parsed_block()

        self.assertEqual(table_block.headers, ["Region", "Metric", "Metric"])
        self.assertEqual(table_block.rows, [["Region", "Revenue", "Cost"], ["East", "100", "60"]])
        self.assertEqual(
            parsed_block.metadata["table"]["expanded_rows"],
            [["Region", "Metric", "Metric"], ["Region", "Revenue", "Cost"], ["East", "100", "60"]],
        )
        self.assertEqual(parsed_block.metadata["table"]["cell_spans"][0]["rowspan"], 2)
        self.assertEqual(parsed_block.metadata["table"]["cell_spans"][1]["colspan"], 2)
        self.assertIn('rowspan="2"', parsed_block.metadata["display"]["content"])
        self.assertIn('colspan="2"', parsed_block.metadata["display"]["content"])

    def test_parse_ooxml_table_respects_omitted_leading_grid_columns(self):
        from rag_service.document_loaders.table.docx_parser import DocxTableParser

        table = FakeOoxmlTable(
            [
                FakeOoxmlTr([FakeOoxmlTc("A"), FakeOoxmlTc("B"), FakeOoxmlTc("C")]),
                FakeOoxmlTr([FakeOoxmlTc("B1"), FakeOoxmlTc("C1")], grid_before=1),
            ]
        )

        table_block = DocxTableParser().parse(table, metadata={"source": "demo.docx"})
        parsed_block = table_block.to_parsed_block()

        self.assertEqual(table_block.rows, [["", "B1", "C1"]])
        self.assertEqual(
            parsed_block.metadata["table"]["expanded_rows"],
            [["A", "B", "C"], ["", "B1", "C1"]],
        )
        self.assertEqual(parsed_block.metadata["table"]["cell_spans"][3]["col"], 0)
        self.assertEqual(parsed_block.metadata["table"]["cell_spans"][4]["col"], 1)


if __name__ == "__main__":
    unittest.main()
