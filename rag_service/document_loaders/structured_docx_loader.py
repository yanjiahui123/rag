from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from rag_service.document_loaders.parsed_blocks import (
    BLOCK_TYPE_TEXT,
    SPLIT_POLICY_MARKDOWN_HEADINGS,
    ParsedBlock,
)
from rag_service.document_loaders.structured_loader import StructuredDocumentLoader
from rag_service.document_loaders.table.models import TableBlock
from rag_service.document_loaders.table.docx_parser import DocxTableParser


class DocxMarkdownElement(BaseModel):
    text: str
    block_index: int
    headers: List[str] = Field(default_factory=list)
    heading_level: Optional[int] = None
    table_metadata: Optional[Dict[str, Any]] = None


class StructuredDocxLoader(StructuredDocumentLoader):
    def __init__(self, file_path: str, source: Optional[str] = None, table_parser: Optional[DocxTableParser] = None):
        super().__init__(file_path, source)
        self.table_parser = table_parser or DocxTableParser()

    def parse_blocks(self) -> List[ParsedBlock]:
        return self.parse_document(self._open_document())

    def parse_document(self, document: Any) -> List[ParsedBlock]:
        elements = self._document_to_markdown_elements(document)
        if not elements:
            return []
        return [self._document_to_block(elements)]

    def _document_to_markdown_elements(self, document: Any) -> List[DocxMarkdownElement]:
        elements = []
        headers = []
        table_index = 0
        for block_index, block in enumerate(self._iter_blocks(document)):
            if self._is_table(block):
                elements.append(self._table_element(block, headers, block_index, table_index))
                table_index += 1
                continue
            headers = self._append_paragraph_element(elements, block, headers, block_index)
        return elements

    def _append_paragraph_element(self, elements, block, headers, block_index):
        text = self._paragraph_text(block)
        if not text:
            return headers
        heading_level = self._heading_level(block)
        if heading_level is None:
            elements.append(self._text_element(text, headers, block_index))
            return headers
        next_headers = self._replace_header(headers, heading_level, text)
        elements.append(self._heading_element(text, heading_level, next_headers, block_index))
        return next_headers

    def _open_document(self) -> Any:
        import docx

        return docx.Document(self.file_path)

    def _iter_blocks(self, document: Any) -> Iterable[Any]:
        if hasattr(document, "blocks"):
            return document.blocks
        from rag_service.document_loaders.docx_section_loader import iter_block_items

        return iter_block_items(document)

    @staticmethod
    def _is_table(block: Any) -> bool:
        return hasattr(block, "rows")

    @staticmethod
    def _paragraph_text(block: Any) -> str:
        return str(getattr(block, "text", "") or "").strip()

    def _heading_level(self, block: Any) -> Optional[int]:
        style_name = getattr(getattr(block, "style", None), "name", "")
        if style_name.startswith("Title"):
            return 1
        if not style_name.startswith("Heading"):
            return None
        return self._parse_heading_level(style_name)

    @staticmethod
    def _parse_heading_level(style_name: str) -> Optional[int]:
        try:
            return int(style_name.split()[-1])
        except Exception:
            return None

    @staticmethod
    def _replace_header(headers: List[str], level: int, text: str) -> List[str]:
        index = max(level - 1, 0)
        next_headers = list(headers[:index])
        while len(next_headers) < index:
            next_headers.append("")
        next_headers.append(text)
        return next_headers

    @staticmethod
    def _heading_element(text: str, level: int, headers: List[str], block_index: int) -> DocxMarkdownElement:
        marker = "#" * max(level, 1)
        return DocxMarkdownElement(
            text=f"{marker} {text}",
            block_index=block_index,
            headers=list(headers),
            heading_level=level,
        )

    @staticmethod
    def _text_element(text: str, headers: List[str], block_index: int) -> DocxMarkdownElement:
        return DocxMarkdownElement(text=text, block_index=block_index, headers=list(headers))

    def _table_element(self, table: Any, headers: List[str], block_index: int, table_index: int) -> DocxMarkdownElement:
        metadata = self._base_metadata(headers, block_index)
        metadata["table_index"] = table_index
        metadata["title"] = headers[-1] if headers else ""
        table_block = self.table_parser.parse(table, metadata=metadata)
        return DocxMarkdownElement(
            text=self._table_to_markdown(table_block),
            block_index=block_index,
            headers=list(headers),
            table_metadata=table_block.to_parsed_block().metadata,
        )

    def _base_metadata(self, headers: List[str], block_index: int) -> dict:
        return {
            "source": self.source,
            "headers": list(headers),
            "block_index": block_index,
            "loader": "structured_docx",
        }

    def _document_to_block(self, elements: List[DocxMarkdownElement]) -> ParsedBlock:
        text = "\n\n".join(element.text for element in elements if element.text).strip()
        metadata = self._document_metadata(elements)
        return ParsedBlock(
            text=text,
            metadata=metadata,
            block_type=BLOCK_TYPE_TEXT,
            split_policy=SPLIT_POLICY_MARKDOWN_HEADINGS,
        )

    def _document_metadata(self, elements: List[DocxMarkdownElement]) -> Dict[str, Any]:
        metadata = self._base_metadata([], 0)
        tables = [copy.deepcopy(element.table_metadata) for element in elements if element.table_metadata]
        if tables:
            metadata["tables"] = tables
        return metadata

    @staticmethod
    def _table_to_markdown(table_block: TableBlock) -> str:
        lines = []
        if table_block.headers:
            lines.append(_markdown_row(table_block.headers))
            lines.append(_markdown_row(["---"] * len(table_block.headers)))
        for row in table_block.rows:
            lines.append(_markdown_row([str(value) for value in row]))
        return "\n".join(lines)


def _markdown_row(values: List[str]) -> str:
    escaped_values = [value.replace("\n", " ").replace("|", "\\|").strip() for value in values]
    return "| " + " | ".join(escaped_values) + " |"
