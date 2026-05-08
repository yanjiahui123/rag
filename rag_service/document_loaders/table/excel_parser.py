from __future__ import annotations

from html import escape
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

CellRange = Tuple[int, int, int, int]


class ExcelTableChunk(BaseModel):
    text: str
    row_range: Tuple[int, int]
    oversized_row: bool = False


class ExcelTable(BaseModel):
    table_id: str
    title: str = ""
    source: str
    sheet_name: str
    cell_range: str
    description: str = ""
    header_rows: List[List[str]] = Field(default_factory=list)
    flatten_headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    display_html: str = ""
    merged_ranges: List[str] = Field(default_factory=list)

    def to_llm_chunks(self, max_chars: int = 1500, min_tail_chars: int = 300) -> List[ExcelTableChunk]:
        prefix = self._llm_prefix()
        chunks = []
        current_rows = []
        current_start = 0
        for row_index, row in enumerate(self.rows):
            candidate_rows = current_rows + [self._row_text(row_index, row)]
            if current_rows and len(prefix + "\n".join(candidate_rows)) > max_chars:
                chunks.append(self._chunk(prefix, current_rows, current_start, row_index))
                current_rows, current_start = [self._row_text(row_index, row)], row_index
            else:
                current_rows = candidate_rows
        if current_rows:
            chunks.append(self._chunk(prefix, current_rows, current_start, len(self.rows)))
        return self._merge_small_tail(chunks, max_chars, min_tail_chars)

    def to_artifact_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "source_type": "excel_table",
            "sheet_name": self.sheet_name,
            "cell_range": self.cell_range,
            "description": self.description,
            "header_rows": self.header_rows,
            "flatten_headers": self.flatten_headers,
            "rows": self.rows,
            "merged_ranges": self.merged_ranges,
        }

    def _llm_prefix(self) -> str:
        lines = [f"文件: {self.source}", f"Sheet: {self.sheet_name}", f"区域: {self.cell_range}"]
        if self.title:
            lines.append(f"表格: {self.title}")
        if self.description:
            lines.append(f"说明: {self.description}")
        lines.append("字段:")
        lines.extend([f"- {header}" for header in self.flatten_headers])
        lines.append("数据:")
        return "\n".join(lines) + "\n"

    def _row_text(self, row_index: int, row: List[str]) -> str:
        pairs = [
            f"{header}={value}"
            for header, value in zip(self.flatten_headers, row)
            if str(value).strip()
        ]
        return f"{row_index + 1}. " + "；".join(pairs)

    @staticmethod
    def _chunk(prefix: str, rows: List[str], start: int, end: int) -> ExcelTableChunk:
        text = prefix + "\n".join(rows)
        return ExcelTableChunk(text=text, row_range=(start, end), oversized_row=len(rows) == 1 and len(text) > 1500)

    def _merge_small_tail(self, chunks: List[ExcelTableChunk], max_chars: int, min_tail_chars: int) -> List[ExcelTableChunk]:
        if len(chunks) < 2 or len(chunks[-1].text) >= min_tail_chars:
            return chunks
        merged_text = chunks[-2].text + "\n" + self._data_part(chunks[-1].text)
        if len(merged_text) > max_chars:
            return chunks
        chunks[-2] = ExcelTableChunk(text=merged_text, row_range=(chunks[-2].row_range[0], chunks[-1].row_range[1]))
        return chunks[:-1]

    @staticmethod
    def _data_part(text: str) -> str:
        return text.split("数据:\n", 1)[-1]


class ExcelTableParser:
    def parse_grid(
        self,
        grid: List[List[Any]],
        sheet_name: str,
        source: str,
        merge_ranges: Optional[List[CellRange]] = None,
    ) -> List[ExcelTable]:
        normalized_grid = _normalize_grid(grid)
        tables = []
        for index, region in enumerate(_table_regions(normalized_grid)):
            table = self._table_from_region(region, index, sheet_name, source, merge_ranges or [])
            if table:
                tables.append(table)
        return tables

    def _table_from_region(
        self,
        region,
        table_index: int,
        sheet_name: str,
        source: str,
        merge_ranges: List[CellRange],
    ) -> Optional[ExcelTable]:
        rows, start_row, start_col = region
        region_merges = _local_merge_ranges(merge_ranges, start_row, start_col, rows)
        description, table_rows, table_row_offset = _extract_description(rows)
        header_count = _header_row_count(table_rows)
        if not table_rows or len(table_rows) <= header_count:
            return None
        header_merges = _header_merge_ranges(region_merges, table_row_offset, header_count)
        header_rows = _normalize_header_rows(table_rows[:header_count], header_merges)
        data_rows = table_rows[header_count:]
        flatten_headers = _flatten_headers(header_rows)
        table = ExcelTable(
            table_id=f"table_{table_index + 1:03d}",
            title=description,
            source=source,
            sheet_name=sheet_name,
            cell_range=_cell_range(start_row, start_col, rows),
            description=description,
            header_rows=header_rows,
            flatten_headers=flatten_headers,
            rows=data_rows,
            merged_ranges=_range_refs(region_merges, start_row, start_col),
        )
        table.display_html = _table_to_html(header_rows, data_rows)
        return table


def _normalize_grid(grid: List[List[Any]]) -> List[List[str]]:
    width = max([len(row) for row in grid] or [0])
    return [[_cell_text(row[col]) if col < len(row) else "" for col in range(width)] for row in grid]


def _cell_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _table_regions(grid: List[List[str]]):
    row_regions = _non_empty_row_regions(grid)
    for start, end in row_regions:
        yield from _column_regions(grid[start:end], start)


def _non_empty_row_regions(grid: List[List[str]]) -> List[Tuple[int, int]]:
    regions, start = [], None
    for index, row in enumerate(grid):
        if any(row):
            start = index if start is None else start
        elif start is not None:
            regions.append((start, index))
            start = None
    if start is not None:
        regions.append((start, len(grid)))
    return regions


def _column_regions(rows: List[List[str]], start_row: int):
    non_empty_cols = [col for col in range(len(rows[0])) if any(row[col] for row in rows)]
    for start_col, end_col in _contiguous_ranges(non_empty_cols):
        yield [[row[start_col:end_col] for row in rows], start_row, start_col]


def _contiguous_ranges(values: List[int]) -> List[Tuple[int, int]]:
    if not values:
        return []
    ranges, start, prev = [], values[0], values[0]
    for value in values[1:]:
        if value != prev + 1:
            ranges.append((start, prev + 1))
            start = value
        prev = value
    ranges.append((start, prev + 1))
    return ranges


def _extract_description(rows: List[List[str]]) -> Tuple[str, List[List[str]], int]:
    if len(rows) < 2:
        return "", rows, 0
    non_empty = [cell for cell in rows[0] if cell]
    if len(non_empty) == 1 and _non_empty_count(rows[1]) >= 2:
        return non_empty[0], rows[1:], 1
    return "", rows, 0


def _header_row_count(rows: List[List[str]]) -> int:
    if len(rows) >= 3 and (_has_repeated_adjacent(rows[0]) or _has_sparse_header_child(rows[1])):
        return 2
    return 1


def _has_repeated_adjacent(row: List[str]) -> bool:
    return any(cell and cell == row[index + 1] for index, cell in enumerate(row[:-1]))


def _has_sparse_header_child(row: List[str]) -> bool:
    return 0 < _non_empty_count(row) < len(row)


def _non_empty_count(row: List[str]) -> int:
    return len([cell for cell in row if cell])


def _flatten_headers(header_rows: List[List[str]]) -> List[str]:
    headers = []
    for col in range(len(header_rows[0])):
        parts = [row[col] for row in header_rows if row[col]]
        headers.append("/".join(dict.fromkeys(parts)))
    return headers


def _normalize_header_rows(
    header_rows: List[List[str]],
    merge_ranges: Optional[List[CellRange]] = None,
) -> List[List[str]]:
    header_rows = _apply_header_merges(header_rows, merge_ranges or [])
    if len(header_rows) < 2:
        return header_rows
    normalized = [list(row) for row in header_rows]
    for row_index in range(len(normalized) - 1):
        lower_rows = normalized[row_index + 1:]
        normalized[row_index] = _fill_sparse_header_cells(normalized[row_index], lower_rows)
    return normalized


def _apply_header_merges(header_rows: List[List[str]], merge_ranges: List[CellRange]) -> List[List[str]]:
    normalized = [list(row) for row in header_rows]
    for start_row, start_col, _, end_col in merge_ranges:
        if not _is_header_cell(normalized, start_row, start_col):
            continue
        value = normalized[start_row][start_col]
        for col_index in range(start_col + 1, min(end_col, len(normalized[start_row]))):
            normalized[start_row][col_index] = value
    return normalized


def _is_header_cell(header_rows: List[List[str]], row: int, col: int) -> bool:
    return row < len(header_rows) and col < len(header_rows[row]) and bool(header_rows[row][col])


def _fill_sparse_header_cells(row: List[str], lower_rows: List[List[str]]) -> List[str]:
    filled, previous = [], ""
    for col_index, value in enumerate(row):
        if value:
            previous = value
            filled.append(value)
        elif previous and _has_child_header(lower_rows, col_index):
            filled.append(previous)
        else:
            filled.append(value)
    return filled


def _has_child_header(lower_rows: List[List[str]], col_index: int) -> bool:
    return any(row[col_index] for row in lower_rows)


def _local_merge_ranges(
    merge_ranges: List[CellRange],
    start_row: int,
    start_col: int,
    rows: List[List[str]],
) -> List[CellRange]:
    if not rows:
        return []
    end_row, end_col = start_row + len(rows), start_col + len(rows[0])
    ranges = [_local_range(cell_range, start_row, start_col, end_row, end_col) for cell_range in merge_ranges]
    return [cell_range for cell_range in ranges if cell_range and _is_multi_cell_range(cell_range)]


def _local_range(cell_range: CellRange, start_row: int, start_col: int, end_row: int, end_col: int):
    row_1, col_1, row_2, col_2 = cell_range
    clipped = max(row_1, start_row), max(col_1, start_col), min(row_2, end_row), min(col_2, end_col)
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None
    return clipped[0] - start_row, clipped[1] - start_col, clipped[2] - start_row, clipped[3] - start_col


def _header_merge_ranges(merge_ranges: List[CellRange], row_offset: int, header_count: int) -> List[CellRange]:
    ranges = [_shift_to_header_range(cell_range, row_offset, header_count) for cell_range in merge_ranges]
    return [cell_range for cell_range in ranges if cell_range and _is_multi_cell_range(cell_range)]


def _shift_to_header_range(cell_range: CellRange, row_offset: int, header_count: int):
    start_row, start_col, end_row, end_col = cell_range
    shifted = start_row - row_offset, start_col, end_row - row_offset, end_col
    if shifted[0] >= header_count or shifted[2] <= 0:
        return None
    return max(shifted[0], 0), start_col, min(shifted[2], header_count), end_col


def _is_multi_cell_range(cell_range: CellRange) -> bool:
    start_row, start_col, end_row, end_col = cell_range
    return end_row - start_row > 1 or end_col - start_col > 1


def _range_refs(merge_ranges: List[CellRange], start_row: int, start_col: int) -> List[str]:
    return [_range_ref(cell_range, start_row, start_col) for cell_range in merge_ranges]


def _range_ref(cell_range: CellRange, start_row: int, start_col: int) -> str:
    row_1, col_1, row_2, col_2 = cell_range
    return f"{_cell_ref(start_row + row_1, start_col + col_1)}:{_cell_ref(start_row + row_2 - 1, start_col + col_2 - 1)}"


def _cell_ref(row: int, col: int) -> str:
    return f"{_col_letter(col)}{row + 1}"


def _cell_range(start_row: int, start_col: int, rows: List[List[str]]) -> str:
    end_row = start_row + len(rows)
    end_col = start_col + len(rows[0])
    return f"{_col_letter(start_col)}{start_row + 1}:{_col_letter(end_col - 1)}{end_row}"


def _col_letter(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _table_to_html(header_rows: List[List[str]], data_rows: List[List[str]]) -> str:
    html_rows = _header_html_rows(header_rows)
    html_rows.extend(_data_html_rows(data_rows))
    return "<table>" + "".join(html_rows) + "</table>"


def _header_html_rows(header_rows: List[List[str]]) -> List[str]:
    covered = set()
    rows = []
    for row_index, row in enumerate(header_rows):
        cells = []
        for col_index, value in enumerate(row):
            if (row_index, col_index) in covered or not value:
                continue
            colspan = _colspan(row, col_index)
            rowspan = 1 if colspan > 1 else _rowspan(header_rows, row_index, col_index)
            _mark_covered(covered, row_index, col_index, rowspan, colspan)
            cells.append(_html_cell("th", value, rowspan, colspan))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return rows


def _data_html_rows(data_rows: List[List[str]]) -> List[str]:
    return ["<tr>" + "".join(_html_cell("td", cell, 1, 1) for cell in row) + "</tr>" for row in data_rows]


def _colspan(row: List[str], col_index: int) -> int:
    value = row[col_index]
    span = 1
    for cell in row[col_index + 1:]:
        if cell != value:
            break
        span += 1
    return span


def _rowspan(header_rows: List[List[str]], row_index: int, col_index: int) -> int:
    span = 1
    for row in header_rows[row_index + 1:]:
        if row[col_index]:
            break
        span += 1
    return span


def _mark_covered(covered: set, row: int, col: int, rowspan: int, colspan: int) -> None:
    for row_offset in range(rowspan):
        for col_offset in range(colspan):
            if row_offset or col_offset:
                covered.add((row + row_offset, col + col_offset))


def _html_cell(tag: str, value: str, rowspan: int, colspan: int) -> str:
    attrs = []
    if rowspan > 1:
        attrs.append(f'rowspan="{rowspan}"')
    if colspan > 1:
        attrs.append(f'colspan="{colspan}"')
    attr_text = " " + " ".join(attrs) if attrs else ""
    return f"<{tag}{attr_text}>{escape(str(value))}</{tag}>"
