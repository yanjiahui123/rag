from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

from rag_service.document_loaders.image_markdown import ImageMarkdown, upload_image_bytes
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

DATA_IMAGE_PATTERN = re.compile(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.*)$", re.IGNORECASE | re.DOTALL)


class StructuredHtmlLoader(StructuredDocumentLoader):
    def __init__(self, file_path: str, source: Optional[str] = None):
        super().__init__(file_path, source)
        self.table_parser = HtmlTableParser()
        self.image_object_keys: List[str] = []

    def parse_blocks(self) -> List[ParsedBlock]:
        return self.parse_to_document().to_blocks()

    def parse_to_document(self) -> ParsedDocument:
        with open(self.file_path, encoding="utf-8") as file:
            return self.parse_html(file.read())

    def parse_html(self, html: str) -> ParsedDocument:
        self.image_object_keys = []
        state = _ParseState(source=self.source)
        state.image_object_keys = self.image_object_keys
        base_dir = Path(self.file_path).parent if self.file_path else None
        for event_type, payload in _content_events(_content_root(parse_html(html)), base_dir, state.image_object_keys):
            self._handle_event(state, event_type, payload)
        self._append_text_block(state)
        return ParsedDocument(blocks=state.blocks, metadata=self._metadata(state.blocks, state.tables, state.image_object_keys))

    def _handle_event(self, state, event_type: str, payload: Any) -> None:
        if event_type == "heading":
            level, text = payload
            self._append_heading(state, level, text)
        elif event_type == "table":
            self._append_table_block(state, payload)
        elif event_type == "text":
            state.text_parts.append(payload)

    def _append_heading(self, state, level: int, text: str) -> None:
        self._append_text_block(state)
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

    def _metadata(self, blocks: List[ParsedBlock], tables: List[TableBlock], image_object_keys: List[str]) -> Dict[str, Any]:
        return {
            "source": self.source,
            STRUCTURED_HTML_ARTIFACTS_KEY: {
                "document_markdown": "\n\n".join(block.text for block in blocks),
                "tables": [self._table_artifact(table) for table in tables],
                "image_object_keys": list(dict.fromkeys(image_object_keys)),
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
        self.image_object_keys: List[str] = []


def _content_root(root: HtmlNode) -> HtmlNode:
    return find_first(root, "body") or root


def _content_events(
    node: HtmlNode,
    base_dir: Optional[Path] = None,
    image_object_keys: Optional[List[str]] = None,
) -> Iterable[Tuple[str, Any]]:
    image_object_keys = image_object_keys if image_object_keys is not None else []
    for child in node.children:
        if isinstance(child, str):
            text = _clean_text(child)
            if text:
                yield "text", text
        elif child.tag in {"script", "style", "caption"}:
            continue
        elif child.tag == "table":
            yield "table", child
        elif child.tag == "img":
            text = _image_markdown(child, base_dir, image_object_keys)
            if text:
                yield "text", text
        elif child.tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            yield "heading", (int(child.tag[1]), child.text())
        elif child.tag in {"p", "li"}:
            text = _node_markdown_text(child, base_dir, image_object_keys)
            if text:
                yield "text", text
        else:
            yield from _content_events(child, base_dir, image_object_keys)


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


def _node_markdown_text(node: HtmlNode, base_dir: Optional[Path], image_object_keys: List[str]) -> str:
    parts = []
    for child in node.children:
        if isinstance(child, str):
            text = _clean_text(child)
            if text:
                parts.append(text)
            continue
        if child.tag in {"script", "style", "caption"}:
            continue
        if child.tag == "img":
            image_text = _image_markdown(child, base_dir, image_object_keys)
            if image_text:
                parts.append(image_text)
            continue
        text = _node_markdown_text(child, base_dir, image_object_keys)
        if text:
            parts.append(text)
    return _join_markdown_parts(parts)


def _image_markdown(node: HtmlNode, base_dir: Optional[Path], image_object_keys: List[str]) -> str:
    src = (node.attrs.get("src") or "").strip()
    if not src:
        return ""
    if _is_remote_image_src(src):
        return f"![]({src})"
    if _is_data_image_src(src):
        return _data_image_markdown(src, image_object_keys)
    return _local_image_markdown(src, base_dir, image_object_keys)


def _is_remote_image_src(src: str) -> bool:
    return src.startswith(("http://", "https://", "//"))


def _is_data_image_src(src: str) -> bool:
    return bool(DATA_IMAGE_PATTERN.match(src))


def _data_image_markdown(src: str, image_object_keys: List[str]) -> str:
    match = DATA_IMAGE_PATTERN.match(src)
    if not match:
        return ""
    extension = "jpg" if match.group(1).lower() == "jpeg" else match.group(1).lower()
    try:
        content = base64.b64decode(match.group(2), validate=True)
    except Exception:
        return ""
    return _uploaded_image_markdown(_upload_html_image_bytes(content, extension), image_object_keys)


def _local_image_markdown(src: str, base_dir: Optional[Path], image_object_keys: List[str]) -> str:
    if base_dir is None:
        return ""
    image_path = _safe_local_image_path(src, base_dir)
    if image_path is None or not image_path.is_file():
        return ""
    uploaded = _upload_html_image_bytes(image_path.read_bytes(), image_path.suffix.lstrip(".") or "png")
    return _uploaded_image_markdown(uploaded, image_object_keys)


def _safe_local_image_path(src: str, base_dir: Path) -> Optional[Path]:
    parsed = urlsplit(src)
    if parsed.scheme or parsed.netloc:
        return None
    relative_path = unquote(parsed.path or "")
    if not relative_path:
        return None
    base_dir = base_dir.resolve()
    image_path = (base_dir / relative_path).resolve()
    try:
        image_path.relative_to(base_dir)
    except ValueError:
        return None
    return image_path


def _upload_html_image_bytes(content: bytes, extension: str) -> ImageMarkdown:
    return upload_image_bytes(content, extension)


def _uploaded_image_markdown(image: ImageMarkdown, image_object_keys: List[str]) -> str:
    if image.object_key:
        image_object_keys.append(image.object_key)
    return image.markdown


def _join_markdown_parts(parts: List[str]) -> str:
    result = ""
    for part in parts:
        if not part:
            continue
        if part.startswith("![]("):
            if result and not result.endswith("\n"):
                result += "\n"
            result += part
            continue
        if result and not result.endswith(("\n", " ")):
            result += " "
        result += part
    return result.strip()


def _block_id(index: int) -> str:
    return f"block_{index + 1:03d}"


def _table_id(index: int) -> str:
    return f"table_{index + 1:03d}"
