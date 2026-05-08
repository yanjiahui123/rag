from __future__ import annotations

import re
from typing import Dict, List, Optional

from rag_service.document_loaders.parsed_blocks import (
    SPLIT_POLICY_MARKDOWN_HEADINGS,
    ParsedBlock,
    ParsedDocument,
)
from rag_service.document_loaders.structured_artifacts import STRUCTURED_MARKDOWN_ARTIFACTS_KEY
from rag_service.document_loaders.structured_loader import StructuredDocumentLoader


class StructuredMarkdownLoader(StructuredDocumentLoader):
    def parse_blocks(self) -> List[ParsedBlock]:
        return self.parse_to_document().to_blocks()

    def parse_to_document(self) -> ParsedDocument:
        with open(self.file_path, encoding="utf-8") as file:
            return self.parse_markdown(file.read())

    def parse_markdown(self, markdown: str) -> ParsedDocument:
        text = markdown.strip()
        if not text:
            return ParsedDocument(metadata={"source": self.source})
        return ParsedDocument(
            blocks=[self._text_block(text)],
            metadata=self._document_metadata(text),
        )

    def _text_block(self, text: str) -> ParsedBlock:
        return ParsedBlock(
            text=text,
            metadata=self._block_metadata(text),
            split_policy=SPLIT_POLICY_MARKDOWN_HEADINGS,
        )

    def _block_metadata(self, text: str) -> Dict[str, object]:
        title = _first_markdown_heading(text.splitlines())
        headers = [title] if title else []
        return {
            "source": self.source,
            "headers": headers,
            "block_index": 0,
            "loader": "structured_markdown",
            "block_id": "block_001",
            "title": title,
        }

    def _document_metadata(self, text: str) -> Dict[str, object]:
        return {
            "source": self.source,
            STRUCTURED_MARKDOWN_ARTIFACTS_KEY: {
                "document_markdown": text,
                "tables": [],
            },
        }


def _first_markdown_heading(lines: List[str]) -> Optional[str]:
    in_fence = False
    for line in lines:
        if _is_fence(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = _heading_text(line)
        if heading:
            return heading
    return None


def _is_fence(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _heading_text(line: str) -> Optional[str]:
    match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line.strip())
    if not match:
        return None
    return match.group(2).strip()
