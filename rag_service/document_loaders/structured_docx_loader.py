from __future__ import annotations

from typing import Any, Iterable, List, Optional

from rag_service.document_loaders.parsed_blocks import BLOCK_TYPE_TEXT, ParsedBlock
from rag_service.document_loaders.structured_loader import StructuredDocumentLoader
from rag_service.document_loaders.table.docx_parser import DocxTableParser


class StructuredDocxLoader(StructuredDocumentLoader):
    def __init__(self, file_path: str, source: Optional[str] = None, table_parser: Optional[DocxTableParser] = None):
        super().__init__(file_path, source)
        self.table_parser = table_parser or DocxTableParser()

    def parse_blocks(self) -> List[ParsedBlock]:
        return self.parse_document(self._open_document())

    def parse_document(self, document: Any) -> List[ParsedBlock]:
        blocks = []
        headers = []
        table_index = 0
        for block_index, block in enumerate(self._iter_blocks(document)):
            if self._is_table(block):
                blocks.append(self._table_block(block, headers, block_index, table_index))
                table_index += 1
                continue
            text = self._paragraph_text(block)
            heading_level = self._heading_level(block)
            if heading_level is not None and text:
                headers = self._replace_header(headers, heading_level, text)
                continue
            if text:
                blocks.append(self._text_block(text, headers, block_index))
        return blocks

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

    def _text_block(self, text: str, headers: List[str], block_index: int) -> ParsedBlock:
        return ParsedBlock(
            text=self._with_headers(text, headers),
            metadata=self._base_metadata(headers, block_index),
            block_type=BLOCK_TYPE_TEXT,
        )

    def _table_block(self, table: Any, headers: List[str], block_index: int, table_index: int) -> ParsedBlock:
        metadata = self._base_metadata(headers, block_index)
        metadata["table_index"] = table_index
        metadata["title"] = headers[-1] if headers else ""
        return self.table_parser.parse(table, metadata=metadata).to_parsed_block()

    def _base_metadata(self, headers: List[str], block_index: int) -> dict:
        return {
            "source": self.source,
            "headers": list(headers),
            "block_index": block_index,
            "loader": "structured_docx",
        }

    @staticmethod
    def _with_headers(text: str, headers: List[str]) -> str:
        header_text = "\n".join(header for header in headers if header)
        return f"{header_text}\n{text}" if header_text else text
