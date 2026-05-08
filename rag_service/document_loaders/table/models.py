from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from rag_service.document_loaders.parsed_blocks import (
    BLOCK_TYPE_TABLE,
    SPLIT_POLICY_TABLE_ROWS,
    ParsedBlock,
)


class TableBlock(BaseModel):
    title: str
    source_type: str
    headers: List[str]
    rows: List[List[Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    display_html: Optional[str] = None
    cell_spans: List[Dict[str, Any]] = Field(default_factory=list)
    expanded_rows: List[List[Any]] = Field(default_factory=list)

    def to_parsed_block(self) -> ParsedBlock:
        metadata = copy.deepcopy(self.metadata)
        metadata["table"] = {
            "title": self.title,
            "source_type": self.source_type,
            "sheet_name": metadata.get("sheet_name"),
            "cell_range": metadata.get("cell_range"),
            "header_rows": [self.headers],
            "flatten_headers": self.headers,
            "expanded_rows": self._expanded_rows(),
            "cell_spans": copy.deepcopy(self.cell_spans),
        }
        if self.display_html:
            metadata["display"] = {
                "type": "table",
                "format": "html",
                "content": self.display_html,
            }
        return ParsedBlock(
            text=self.to_search_text(),
            metadata=metadata,
            block_type=BLOCK_TYPE_TABLE,
            split_policy=SPLIT_POLICY_TABLE_ROWS,
        )

    def to_search_text(self) -> str:
        return self.to_llm_text()

    def to_llm_text(self, section_headers: Optional[List[str]] = None) -> str:
        headers = _llm_headers(self.headers, self.rows)
        lines = []
        section = _section_text(section_headers)
        if section:
            lines.append(f"section: {section}")
        if self.title:
            lines.append(f"table: {self.title}")
        if self.source_type:
            lines.append(f"source_type: {self.source_type}")
        if headers:
            lines.append("fields:")
            lines.extend([f"- {header}" for header in headers])
        row_lines = _llm_row_lines(headers, self.rows)
        if row_lines:
            lines.append("rows:")
            lines.extend(row_lines)
        return "\n".join(lines)

    def to_artifact_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "source_type": self.source_type,
            "headers": copy.deepcopy(self.headers),
            "rows": copy.deepcopy(self.rows),
            "expanded_rows": self._expanded_rows(),
            "cell_spans": copy.deepcopy(self.cell_spans),
            "metadata": copy.deepcopy(self.metadata),
        }

    def _expanded_rows(self) -> List[List[Any]]:
        if self.expanded_rows:
            return copy.deepcopy(self.expanded_rows)
        if not self.headers:
            return copy.deepcopy(self.rows)
        return [copy.deepcopy(self.headers)] + copy.deepcopy(self.rows)


def _markdown_row(values: List[str]) -> str:
    escaped_values = [value.replace("\n", " ").replace("|", "\\|").strip() for value in values]
    return "| " + " | ".join(escaped_values) + " |"


def _section_text(section_headers: Optional[List[str]]) -> str:
    return " > ".join(_clean_table_value(header) for header in section_headers or [] if _clean_table_value(header))


def _llm_headers(headers: List[str], rows: List[List[Any]]) -> List[str]:
    width = max([len(headers)] + [len(row) for row in rows] or [0])
    return [_llm_header(headers, index) for index in range(width)]


def _llm_header(headers: List[str], index: int) -> str:
    if index < len(headers):
        header = _clean_table_value(headers[index])
        if header:
            return header
    return f"Column {index + 1}"


def _llm_row_lines(headers: List[str], rows: List[List[Any]]) -> List[str]:
    lines = []
    for row_index, row in enumerate(rows):
        pairs = _llm_row_pairs(headers, row)
        if pairs:
            lines.append(f"{row_index + 1}. " + "; ".join(pairs))
    return lines


def _llm_row_pairs(headers: List[str], row: List[Any]) -> List[str]:
    pairs = []
    for col_index, header in enumerate(headers):
        value = _clean_table_value(row[col_index]) if col_index < len(row) else ""
        if value:
            pairs.append(f"{header}={value}")
    return pairs


def _clean_table_value(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())
