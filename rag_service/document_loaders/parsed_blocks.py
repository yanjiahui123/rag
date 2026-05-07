from __future__ import annotations

import copy
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

BLOCK_TYPE_TEXT = "text"
BLOCK_TYPE_TABLE = "table"
DISPLAY_METADATA_KEY = "display"
SPLIT_POLICY_TEXT = "text"
SPLIT_POLICY_TABLE_ROWS = "table_rows"
SPLIT_POLICY_NO_SPLIT = "no_split"
SPLIT_POLICY_MARKDOWN_HEADINGS = "markdown_headings"


class DisplayPayload(BaseModel):
    type: str
    format: str
    content: Optional[str] = None
    ref: Optional[str] = None

    def to_metadata(self) -> Dict[str, str]:
        payload = {"type": self.type, "format": self.format}
        if self.content is not None:
            payload["content"] = self.content
        if self.ref is not None:
            payload["ref"] = self.ref
        return payload


class ParsedBlock(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    block_type: str = BLOCK_TYPE_TEXT
    split_policy: str = SPLIT_POLICY_TEXT

    def __init__(self, **data):
        super().__init__(**data)
        metadata = copy.deepcopy(self.metadata)
        metadata.setdefault("block_type", self.block_type)
        metadata.setdefault("split_policy", self.split_policy)
        object.__setattr__(self, "metadata", metadata)


def documents_to_parsed_blocks(documents: Iterable[Any], split_policy: str = SPLIT_POLICY_TEXT) -> List[ParsedBlock]:
    blocks = []
    for document in documents:
        text = getattr(document, "page_content", "")
        if not text:
            continue
        metadata = copy.deepcopy(getattr(document, "metadata", {}) or {})
        block_type = metadata.get("block_type", BLOCK_TYPE_TEXT)
        block_split_policy = metadata.get("split_policy", split_policy)
        blocks.append(
            ParsedBlock(
                text=text,
                metadata=metadata,
                block_type=block_type,
                split_policy=block_split_policy,
            )
        )
    return blocks


def blocks_to_documents(
    blocks: Iterable[ParsedBlock],
    document_factory: Callable[..., Any],
    text_splitter: Optional[Any] = None,
) -> List[Any]:
    documents = []
    for block in blocks:
        if not block.text:
            continue
        for split in _split_block_text(block, text_splitter):
            if not split:
                continue
            documents.append(document_factory(page_content=split, metadata=copy.deepcopy(block.metadata)))
    return documents


def _split_block_text(block: ParsedBlock, text_splitter: Optional[Any]) -> List[str]:
    if block.split_policy == SPLIT_POLICY_NO_SPLIT:
        return [block.text]
    if block.split_policy == SPLIT_POLICY_MARKDOWN_HEADINGS:
        return _split_markdown_text(block.text, text_splitter)
    if text_splitter is None:
        return [block.text]
    return text_splitter.split_text(block.text)


def _split_markdown_text(text: str, text_splitter: Optional[Any]) -> List[str]:
    sections = _split_markdown_by_deepest_heading(text)
    if text_splitter is None:
        return sections
    splits = []
    for section in sections:
        splits.extend(text_splitter.split_text(section))
    return splits


def _split_markdown_by_deepest_heading(text: str) -> List[str]:
    lines = text.splitlines()
    split_level = _deepest_heading_level(lines)
    if split_level is None:
        return [text]
    sections = []
    current = []
    for line in lines:
        if current and _markdown_heading_level(line) == split_level:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [section for section in sections if section]


def _deepest_heading_level(lines: List[str]) -> Optional[int]:
    levels = [_markdown_heading_level(line) for line in lines]
    levels = [level for level in levels if level is not None]
    return max(levels) if levels else None


def _markdown_heading_level(line: str) -> Optional[int]:
    match = re.match(r"^(#{1,6})\s+\S", line.strip())
    return len(match.group(1)) if match else None
