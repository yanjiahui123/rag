from __future__ import annotations

from html import escape
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel

from rag_service.document_loaders.table.models import TableBlock


class DocxCellSpan(BaseModel):
    text: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "row": self.row,
            "col": self.col,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
        }


class DocxTableParser:
    def parse(self, table: Any, metadata: Optional[Dict[str, Any]] = None) -> TableBlock:
        cell_spans = self._collect_spans(table)
        expanded_rows = self._to_grid(cell_spans)
        headers, rows = self._to_table_values(expanded_rows)
        block_metadata = dict(metadata or {})
        return TableBlock(
            title=block_metadata.get("title", ""),
            source_type="docx_table",
            headers=headers,
            rows=rows,
            metadata=block_metadata,
            display_html=self._to_html(cell_spans),
            cell_spans=[span.to_metadata() for span in cell_spans],
            expanded_rows=expanded_rows,
        )

    def _collect_spans(self, table: Any) -> List[DocxCellSpan]:
        tr_list = getattr(getattr(table, "_tbl", None), "tr_lst", None)
        if tr_list:
            return self._collect_ooxml_spans(tr_list)
        return self._collect_row_spans(table)

    def _collect_row_spans(self, table: Any) -> List[DocxCellSpan]:
        spans = []
        occupied = set()
        for row_index, row in enumerate(getattr(table, "rows", [])):
            col_index = 0
            for cell in getattr(row, "cells", []):
                col_index = self._next_open_col(row_index, col_index, occupied)
                span = self._span_from_cell(cell, row_index, col_index)
                spans.append(span)
                self._mark_occupied(occupied, span)
                col_index += span.colspan
        return spans

    def _collect_ooxml_spans(self, tr_list: Iterable[Any]) -> List[DocxCellSpan]:
        spans = []
        active_merges = {}
        for row_index, tr in enumerate(tr_list):
            col_index = self._append_omitted_cells(spans, tr, row_index, 0, "gridBefore")
            for tc in getattr(tr, "tc_lst", []):
                colspan = self._ooxml_colspan(tc)
                vmerge = self._ooxml_vmerge(tc)
                if vmerge == "continue" and col_index in active_merges:
                    self._extend_rowspan(active_merges, col_index, colspan)
                else:
                    span = DocxCellSpan(text=self._ooxml_text(tc), row=row_index, col=col_index, colspan=colspan)
                    spans.append(span)
                    self._update_active_merges(active_merges, span, vmerge)
                col_index += colspan
            self._append_omitted_cells(spans, tr, row_index, col_index, "gridAfter")
        return spans

    def _append_omitted_cells(self, spans, tr, row_index, col_index, attr_name) -> int:
        omitted_count = self._ooxml_omitted_grid_count(tr, attr_name)
        for offset in range(omitted_count):
            spans.append(DocxCellSpan(text="", row=row_index, col=col_index + offset))
        return col_index + omitted_count

    def _span_from_cell(self, cell: Any, row_index: int, col_index: int) -> DocxCellSpan:
        return DocxCellSpan(
            text=self._cell_text(cell),
            row=row_index,
            col=col_index,
            rowspan=self._cell_rowspan(cell),
            colspan=self._cell_colspan(cell),
        )

    @staticmethod
    def _next_open_col(row_index: int, col_index: int, occupied: set[Tuple[int, int]]) -> int:
        while (row_index, col_index) in occupied:
            col_index += 1
        return col_index

    @staticmethod
    def _mark_occupied(occupied: set[Tuple[int, int]], span: DocxCellSpan):
        for row_index in range(span.row, span.row + span.rowspan):
            for col_index in range(span.col, span.col + span.colspan):
                if row_index != span.row or col_index != span.col:
                    occupied.add((row_index, col_index))

    @staticmethod
    def _cell_text(cell: Any) -> str:
        return str(getattr(cell, "text", "") or "").replace("\r", "\n").strip()

    def _cell_colspan(self, cell: Any) -> int:
        explicit_colspan = getattr(cell, "colspan", None)
        if explicit_colspan:
            return int(explicit_colspan)
        return self._ooxml_colspan(getattr(cell, "_tc", None))

    @staticmethod
    def _cell_rowspan(cell: Any) -> int:
        explicit_rowspan = getattr(cell, "rowspan", None)
        return int(explicit_rowspan) if explicit_rowspan else 1

    @staticmethod
    def _ooxml_colspan(tc: Any) -> int:
        grid_span = getattr(getattr(tc, "tcPr", None), "gridSpan", None)
        value = getattr(grid_span, "val", None)
        return int(value) if value else 1

    @staticmethod
    def _ooxml_vmerge(tc: Any) -> str:
        vmerge = getattr(getattr(tc, "tcPr", None), "vMerge", None)
        if not vmerge:
            return ""
        value = str(getattr(vmerge, "val", "") or "").lower()
        return "restart" if value == "restart" else "continue"

    @staticmethod
    def _ooxml_text(tc: Any) -> str:
        try:
            return "".join(node.text or "" for node in tc.xpath(".//w:t")).strip()
        except Exception:
            return str(getattr(tc, "text", "") or "").strip()

    @staticmethod
    def _ooxml_omitted_grid_count(tr: Any, attr_name: str) -> int:
        value = getattr(getattr(getattr(tr, "trPr", None), attr_name, None), "val", None)
        return int(value) if value else 0

    @staticmethod
    def _extend_rowspan(active_merges: Dict[int, DocxCellSpan], col_index: int, colspan: int):
        extended_ids = set()
        for offset in range(colspan):
            span = active_merges.get(col_index + offset)
            if span and id(span) not in extended_ids:
                span.rowspan += 1
                extended_ids.add(id(span))

    @staticmethod
    def _update_active_merges(active_merges: Dict[int, DocxCellSpan], span: DocxCellSpan, vmerge: str):
        for col_index in range(span.col, span.col + span.colspan):
            if vmerge == "restart":
                active_merges[col_index] = span
            else:
                active_merges.pop(col_index, None)

    @staticmethod
    def _to_table_values(expanded_rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        if not expanded_rows:
            return [], []
        return expanded_rows[0], expanded_rows[1:]

    @staticmethod
    def _to_grid(spans: List[DocxCellSpan]) -> List[List[str]]:
        if not spans:
            return []
        height = max(span.row + span.rowspan for span in spans)
        width = max(span.col + span.colspan for span in spans)
        grid = [["" for _ in range(width)] for _ in range(height)]
        for span in spans:
            for row_index in range(span.row, span.row + span.rowspan):
                for col_index in range(span.col, span.col + span.colspan):
                    grid[row_index][col_index] = span.text
        return grid

    def _to_html(self, spans: List[DocxCellSpan]) -> str:
        rows = self._group_by_row(spans)
        html_rows = []
        for row_index in sorted(rows):
            tag = "th" if row_index == 0 else "td"
            cells = [self._cell_to_html(tag, span) for span in rows[row_index]]
            html_rows.append("<tr>" + "".join(cells) + "</tr>")
        return "<table>" + "".join(html_rows) + "</table>"

    @staticmethod
    def _group_by_row(spans: List[DocxCellSpan]) -> Dict[int, List[DocxCellSpan]]:
        rows = {}
        for span in spans:
            rows.setdefault(span.row, []).append(span)
        for row in rows.values():
            row.sort(key=lambda span: span.col)
        return rows

    @staticmethod
    def _cell_to_html(tag: str, span: DocxCellSpan) -> str:
        attrs = []
        if span.rowspan > 1:
            attrs.append(f'rowspan="{span.rowspan}"')
        if span.colspan > 1:
            attrs.append(f'colspan="{span.colspan}"')
        attr_text = " " + " ".join(attrs) if attrs else ""
        return f"<{tag}{attr_text}>{escape(span.text)}</{tag}>"
