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
            blocks=self._text_blocks(text),
            metadata=self._document_metadata(text),
        )

    def _text_blocks(self, text: str) -> List[ParsedBlock]:
        sections = _markdown_sections(text)
        return [self._text_block(section_text, headers, index) for index, section_text, headers in sections]

    def _text_block(self, text: str, headers: List[str], index: int) -> ParsedBlock:
        return ParsedBlock(
            text=text,
            metadata=self._block_metadata(headers, index),
            split_policy=SPLIT_POLICY_MARKDOWN_HEADINGS,
        )

    def _block_metadata(self, headers: List[str], index: int) -> Dict[str, object]:
        title = headers[-1] if headers else None
        return {
            "source": self.source,
            "headers": list(headers),
            "block_index": index,
            "loader": "structured_markdown",
            "block_id": f"block_{index + 1:03d}",
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


def _markdown_sections(text: str) -> List[tuple]:
    sections = []
    current_lines: List[str] = []
    current_headers: List[str] = []
    headers: List[str] = []
    in_fence = False
    for line in text.splitlines():
        if _is_fence(line):
            in_fence = not in_fence
        heading = None if in_fence else _heading(line)
        if heading:
            _append_section(sections, current_lines, current_headers)
            headers = _replace_header(headers, heading[0], heading[1])
            current_headers = list(headers)
            current_lines = [line]
            continue
        current_lines.append(line)
    _append_section(sections, current_lines, current_headers)
    return [(index, section, headers) for index, section, headers in sections if section]


def _append_section(sections: List[tuple], lines: List[str], headers: List[str]) -> None:
    section = "\n".join(lines).strip()
    if section:
        sections.append((len(sections), section, list(headers)))


def _replace_header(headers: List[str], level: int, text: str) -> List[str]:
    index = max(level - 1, 0)
    next_headers = list(headers[:index])
    while len(next_headers) < index:
        next_headers.append("")
    next_headers.append(text)
    return next_headers


def _heading(line: str) -> Optional[tuple]:
    match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line.strip())
    if not match:
        return None
    return len(match.group(1)), match.group(2).strip()


def _is_fence(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _heading_text(line: str) -> Optional[str]:
    match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line.strip())
    if not match:
        return None
    return match.group(2).strip()
