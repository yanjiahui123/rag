from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from rag_service.document_loaders.html_dom import HtmlNode, find_first, parse_html
from rag_service.document_loaders.parsed_blocks import (
    BLOCK_TYPE_TABLE,
    SPLIT_POLICY_MARKDOWN_HEADINGS,
    SPLIT_POLICY_NO_SPLIT,
    ParsedBlock,
    ParsedDocument,
)
from rag_service.document_loaders.structured_artifacts import STRUCTURED_HTML_ARTIFACTS_KEY
from rag_service.document_loaders.structured_loader import StructuredDocumentLoader
from rag_service.document_loaders.table.html_parser import HtmlTableParser
from rag_service.document_loaders.table.models import TableBlock


class StructuredHtmlLoader(StructuredDocumentLoader):
    def __init__(self, file_path: str, source: Optional[str] = None):
        super().__init__(file_path, source)
        self.table_parser = HtmlTableParser()

    def parse_blocks(self) -> List[ParsedBlock]:
        return self.parse_to_document().to_blocks()

    def parse_to_document(self) -> ParsedDocument:
        with open(self.file_path, encoding="utf-8") as file:
            return self.parse_html(file.read())

    def parse_html(self, html: str) -> ParsedDocument:
        state = _ParseState(source=self.source)
        for event_type, payload in _content_events(_content_root(parse_html(html))):
            self._handle_event(state, event_type, payload)
        self._append_text_block(state)
        return ParsedDocument(blocks=state.blocks, metadata=self._metadata(state.blocks, state.tables))

    def _handle_event(self, state, event_type: str, payload: Any) -> None:
        if event_type == "heading":
            level, text = payload
            self._append_heading(state, level, text)
        elif event_type == "table":
            self._append_table_block(state, payload)
        elif event_type == "text":
            state.text_parts.append(payload)

    def _append_heading(self, state, level: int, text: str) -> None:
        state.headers = _replace_header(state.headers, level, text)
        state.text_parts.append("#" * max(level, 1) + " " + text)

    def _append_table_block(self, state, table_node: HtmlNode) -> None:
        self._append_text_block(state)
        table_index = len(state.tables)
        table = self.table_parser.parse(table_node, metadata=self._base_metadata(state))
        table.metadata.update({"table_id": _table_id(table_index), "title": table.title})
        state.tables.append(table)
        state.blocks.append(self._table_block(table, state, table_index))

    def _append_text_block(self, state) -> None:
        text = "\n\n".join(part for part in state.text_parts if part).strip()
        if not text:
            state.text_parts = []
            return
        metadata = self._base_metadata(state)
        metadata.update({"block_id": _block_id(len(state.blocks)), "title": state.headers[-1] if state.headers else ""})
        state.blocks.append(ParsedBlock(text=text, metadata=metadata, split_policy=SPLIT_POLICY_MARKDOWN_HEADINGS))
        state.text_parts = []

    def _table_block(self, table: TableBlock, state, table_index: int) -> ParsedBlock:
        metadata = self._table_metadata(table, state, table_index)
        text = table.to_llm_text(section_headers=state.headers)
        return ParsedBlock(text=text, metadata=metadata, block_type=BLOCK_TYPE_TABLE, split_policy=SPLIT_POLICY_NO_SPLIT)

    def _table_metadata(self, table: TableBlock, state, table_index: int) -> Dict[str, Any]:
        metadata = self._base_metadata(state)
        metadata.update(
            {
                "block_id": _block_id(len(state.blocks)),
                "title": table.title,
                "table_id": _table_id(table_index),
                "table": _table_summary(table, _table_id(table_index)),
            }
        )
        return metadata

    def _base_metadata(self, state) -> Dict[str, Any]:
        return {"source": self.source, "headers": list(state.headers), "loader": "structured_html"}

    def _metadata(self, blocks: List[ParsedBlock], tables: List[TableBlock]) -> Dict[str, Any]:
        return {
            "source": self.source,
            STRUCTURED_HTML_ARTIFACTS_KEY: {
                "document_markdown": "\n\n".join(block.text for block in blocks),
                "tables": [self._table_artifact(table) for table in tables],
            },
        }

    @staticmethod
    def _table_artifact(table: TableBlock) -> Dict[str, Any]:
        return {
            "table_id": table.metadata.get("table_id"),
            "html": table.display_html or "",
            "json": table.to_artifact_dict(),
            "llm_markdown": table.to_llm_text(section_headers=table.metadata.get("headers", [])),
        }


class _ParseState:
    def __init__(self, source: str):
        self.source = source
        self.headers: List[str] = []
        self.text_parts: List[str] = []
        self.blocks: List[ParsedBlock] = []
        self.tables: List[TableBlock] = []


def _content_root(root: HtmlNode) -> HtmlNode:
    return find_first(root, "body") or root


def _content_events(node: HtmlNode) -> Iterable[Tuple[str, Any]]:
    for child in node.children:
        if isinstance(child, str):
            text = _clean_text(child)
            if text:
                yield "text", text
        elif child.tag in {"script", "style", "caption"}:
            continue
        elif child.tag == "table":
            yield "table", child
        elif child.tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            yield "heading", (int(child.tag[1]), child.text())
        elif child.tag in {"p", "li"}:
            text = child.text()
            if text:
                yield "text", text
        else:
            yield from _content_events(child)


def _replace_header(headers: List[str], level: int, text: str) -> List[str]:
    index = max(level - 1, 0)
    next_headers = list(headers[:index])
    while len(next_headers) < index:
        next_headers.append("")
    next_headers.append(text)
    return next_headers


def _table_summary(table: TableBlock, table_id: str) -> Dict[str, Any]:
    return {
        "table_id": table_id,
        "title": table.title,
        "source_type": table.source_type,
        "flatten_headers": list(table.headers),
        "row_count": len(table.rows),
        "col_count": len(table.headers),
    }


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _block_id(index: int) -> str:
    return f"block_{index + 1:03d}"


def _table_id(index: int) -> str:
    return f"table_{index + 1:03d}"
