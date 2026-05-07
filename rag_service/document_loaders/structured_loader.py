from __future__ import annotations

from typing import List, Optional

from rag_service.document_loaders.parsed_blocks import ParsedBlock, ParsedDocument


class StructuredDocumentLoader:
    def __init__(self, file_path: str, source: Optional[str] = None):
        self.file_path = file_path
        self.source = source or file_path

    def parse_blocks(self) -> List[ParsedBlock]:
        raise NotImplementedError

    def parse_to_document(self) -> ParsedDocument:
        return ParsedDocument(blocks=self.parse_blocks())
