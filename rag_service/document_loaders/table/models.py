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
        lines = []
        if self.title:
            lines.append(f"table: {self.title}")
        if self.source_type:
            lines.append(f"source_type: {self.source_type}")
        if self.headers:
            lines.append(_markdown_row(self.headers))
            lines.append(_markdown_row(["-"] * len(self.headers)))
        for row in self.rows:
            lines.append(_markdown_row([str(value) for value in row]))
        return "\n".join(lines)

    def _expanded_rows(self) -> List[List[Any]]:
        if self.expanded_rows:
            return copy.deepcopy(self.expanded_rows)
        if not self.headers:
            return copy.deepcopy(self.rows)
        return [copy.deepcopy(self.headers)] + copy.deepcopy(self.rows)


def _markdown_row(values: List[str]) -> str:
    escaped_values = [value.replace("\n", " ").replace("|", "\\|").strip() for value in values]
    return "| " + " | ".join(escaped_values) + " |"
