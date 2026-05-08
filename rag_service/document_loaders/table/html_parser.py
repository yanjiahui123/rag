from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Dict, List, Optional, Tuple

from rag_service.document_loaders.html_dom import HtmlNode, find_first, iter_nodes, parse_html
from rag_service.document_loaders.table.models import TableBlock


@dataclass
class HtmlCellSpan:
    text: str
    row: int
    col: int
    tag: str
    rowspan: int = 1
    colspan: int = 1

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "row": self.row,
            "col": self.col,
            "tag": self.tag,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
        }


class HtmlTableParser:
    def parse_html(self, html: str, metadata: Optional[Dict[str, Any]] = None) -> TableBlock:
        table = find_first(parse_html(html), "table")
        if table is None:
            return TableBlock(title="", source_type="html_table", headers=[], rows=[], metadata=metadata or {})
        return self.parse(table, metadata)

    def parse(self, table: HtmlNode, metadata: Optional[Dict[str, Any]] = None) -> TableBlock:
        spans = _collect_spans(table)
        expanded_rows = _to_grid(spans)
        header_count = _header_row_count(spans)
        headers = _flatten_headers(expanded_rows[:header_count])
        rows = expanded_rows[header_count:]
        title = _table_title(table, metadata or {})
        return TableBlock(
            title=title,
            source_type="html_table",
            headers=headers,
            rows=rows,
            metadata=dict(metadata or {}),
            display_html=_to_html(table, spans, header_count),
            cell_spans=[span.to_metadata() for span in spans],
            expanded_rows=expanded_rows,
        )


def _collect_spans(table: HtmlNode) -> List[HtmlCellSpan]:
    spans, occupied = [], set()
    for row_index, row in enumerate(_table_rows(table)):
        col_index = 0
        for cell in _row_cells(row):
            col_index = _next_open_col(row_index, col_index, occupied)
            span = _span_from_cell(cell, row_index, col_index)
            spans.append(span)
            _mark_occupied(occupied, span)
            col_index += span.colspan
    return spans


def _table_rows(table: HtmlNode) -> List[HtmlNode]:
    return [node for node in iter_nodes(table) if node.tag == "tr"]


def _row_cells(row: HtmlNode) -> List[HtmlNode]:
    return [child for child in row.children if isinstance(child, HtmlNode) and child.tag in {"th", "td"}]


def _span_from_cell(cell: HtmlNode, row: int, col: int) -> HtmlCellSpan:
    return HtmlCellSpan(
        text=cell.text(),
        row=row,
        col=col,
        tag=cell.tag,
        rowspan=_span_attr(cell, "rowspan"),
        colspan=_span_attr(cell, "colspan"),
    )


def _span_attr(cell: HtmlNode, attr_name: str) -> int:
    try:
        return max(int(cell.attrs.get(attr_name, "1")), 1)
    except ValueError:
        return 1


def _next_open_col(row_index: int, col_index: int, occupied: set[Tuple[int, int]]) -> int:
    while (row_index, col_index) in occupied:
        col_index += 1
    return col_index


def _mark_occupied(occupied: set[Tuple[int, int]], span: HtmlCellSpan) -> None:
    for row_index in range(span.row, span.row + span.rowspan):
        for col_index in range(span.col, span.col + span.colspan):
            if row_index != span.row or col_index != span.col:
                occupied.add((row_index, col_index))


def _to_grid(spans: List[HtmlCellSpan]) -> List[List[str]]:
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


def _header_row_count(spans: List[HtmlCellSpan]) -> int:
    rows = _group_by_row(spans)
    count = 0
    for row_index in sorted(rows):
        if any(span.tag == "th" for span in rows[row_index]):
            count += 1
            continue
        break
    return count or (1 if spans else 0)


def _flatten_headers(header_rows: List[List[str]]) -> List[str]:
    if not header_rows:
        return []
    headers = []
    for col_index in range(len(header_rows[0])):
        parts = [row[col_index] for row in header_rows if row[col_index]]
        headers.append("/".join(dict.fromkeys(parts)) or f"Column {col_index + 1}")
    return headers


def _table_title(table: HtmlNode, metadata: Dict[str, Any]) -> str:
    caption = find_first(table, "caption")
    return caption.text() if caption else metadata.get("title", "")


def _to_html(table: HtmlNode, spans: List[HtmlCellSpan], header_count: int) -> str:
    rows = _group_by_row(spans)
    html_rows = [_caption_html(table)]
    for row_index in sorted(rows):
        cells = [_cell_to_html(_html_tag(row_index, span, header_count), span) for span in rows[row_index]]
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(row for row in html_rows if row) + "</table>"


def _caption_html(table: HtmlNode) -> str:
    caption = find_first(table, "caption")
    return f"<caption>{escape(caption.text())}</caption>" if caption and caption.text() else ""


def _html_tag(row_index: int, span: HtmlCellSpan, header_count: int) -> str:
    return "th" if row_index < header_count or span.tag == "th" else "td"


def _cell_to_html(tag: str, span: HtmlCellSpan) -> str:
    attrs = []
    if span.rowspan > 1:
        attrs.append(f'rowspan="{span.rowspan}"')
    if span.colspan > 1:
        attrs.append(f'colspan="{span.colspan}"')
    attr_text = " " + " ".join(attrs) if attrs else ""
    return f"<{tag}{attr_text}>{escape(span.text)}</{tag}>"


def _group_by_row(spans: List[HtmlCellSpan]) -> Dict[int, List[HtmlCellSpan]]:
    rows = {}
    for span in spans:
        rows.setdefault(span.row, []).append(span)
    for row in rows.values():
        row.sort(key=lambda span: span.col)
    return rows
