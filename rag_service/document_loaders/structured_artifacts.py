from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

STRUCTURED_DOCX_ARTIFACTS_KEY = "_structured_docx_artifacts"
STRUCTURED_DOCX_METADATA_KEY = "structured_docx"
STRUCTURED_EXCEL_ARTIFACTS_KEY = "_structured_excel_artifacts"
STRUCTURED_EXCEL_METADATA_KEY = "structured_excel"
STRUCTURED_HTML_ARTIFACTS_KEY = "_structured_html_artifacts"
STRUCTURED_HTML_METADATA_KEY = "structured_html"
STRUCTURED_MARKDOWN_ARTIFACTS_KEY = "_structured_markdown_artifacts"
STRUCTURED_MARKDOWN_METADATA_KEY = "structured_markdown"
PARSED_MARKDOWN_METADATA_KEY = "parsed_markdown"

_SAFE_ARTIFACT_TYPES = (
    "structured_docx",
    "structured_excel",
    "structured_html",
    "structured_markdown",
    "parsed_markdown",
)


def build_structured_docx_artifact_prefix(kb_sn: str, asset_name: str, doc_id: str) -> str:
    return _build_document_artifact_prefix(doc_id, "structured_docx")


def build_structured_excel_artifact_prefix(kb_sn: str, asset_name: str, doc_id: str) -> str:
    return _build_document_artifact_prefix(doc_id, "structured_excel")


def build_structured_html_artifact_prefix(kb_sn: str, asset_name: str, doc_id: str) -> str:
    return _build_document_artifact_prefix(doc_id, "structured_html")


def build_structured_markdown_artifact_prefix(kb_sn: str, asset_name: str, doc_id: str) -> str:
    return _build_document_artifact_prefix(doc_id, "structured_markdown")


def build_parsed_markdown_artifact_prefix(kb_sn: str, asset_name: str, doc_id: str) -> str:
    return _build_document_artifact_prefix(doc_id, "parsed_markdown")


def _build_document_artifact_prefix(doc_id: str, artifact_type: str) -> str:
    return f"{doc_id}/{artifact_type}/"


def is_safe_structured_artifact_prefix(prefix: Optional[str]) -> bool:
    if not prefix:
        return False
    return bool(re.match(rf"^[^/]+/({'|'.join(_SAFE_ARTIFACT_TYPES)})/$", prefix))


def extract_structured_docx_artifact_prefix(extended_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    return _extract_artifact_prefix(extended_metadata, STRUCTURED_DOCX_METADATA_KEY)


def extract_structured_excel_artifact_prefix(extended_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    return _extract_artifact_prefix(extended_metadata, STRUCTURED_EXCEL_METADATA_KEY)


def extract_structured_html_artifact_prefix(extended_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    return _extract_artifact_prefix(extended_metadata, STRUCTURED_HTML_METADATA_KEY)


def extract_structured_markdown_artifact_prefix(extended_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    return _extract_artifact_prefix(extended_metadata, STRUCTURED_MARKDOWN_METADATA_KEY)


def extract_parsed_markdown_artifact_prefix(extended_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    return _extract_artifact_prefix(extended_metadata, PARSED_MARKDOWN_METADATA_KEY)


def _extract_artifact_prefix(extended_metadata: Optional[Dict[str, Any]], metadata_key: str) -> Optional[str]:
    if not extended_metadata:
        return None
    metadata = extended_metadata.get(metadata_key) or {}
    return metadata.get("artifact_prefix")


def merge_structured_docx_metadata(
    extended_metadata: Optional[Dict[str, Any]],
    artifact_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return merge_structured_metadata(extended_metadata, STRUCTURED_DOCX_METADATA_KEY, artifact_summary)


def merge_structured_excel_metadata(
    extended_metadata: Optional[Dict[str, Any]],
    artifact_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return merge_structured_metadata(extended_metadata, STRUCTURED_EXCEL_METADATA_KEY, artifact_summary)


def merge_structured_html_metadata(
    extended_metadata: Optional[Dict[str, Any]],
    artifact_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return merge_structured_metadata(extended_metadata, STRUCTURED_HTML_METADATA_KEY, artifact_summary)


def merge_structured_markdown_metadata(
    extended_metadata: Optional[Dict[str, Any]],
    artifact_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return merge_structured_metadata(extended_metadata, STRUCTURED_MARKDOWN_METADATA_KEY, artifact_summary)


def merge_structured_metadata(
    extended_metadata: Optional[Dict[str, Any]],
    metadata_key: str,
    artifact_summary: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = dict(extended_metadata or {})
    metadata[metadata_key] = dict(artifact_summary)
    return metadata


def persist_structured_docx_artifacts(
    parsed_document: Any,
    kb_sn: str,
    asset_name: str,
    doc_id: str,
    upload_content: Callable[[str, str], str],
) -> Dict[str, Any]:
    return _persist_structured_artifacts(
        parsed_document,
        build_structured_docx_artifact_prefix(kb_sn, asset_name, doc_id),
        STRUCTURED_DOCX_ARTIFACTS_KEY,
        STRUCTURED_DOCX_METADATA_KEY,
        upload_content,
    )


def persist_structured_excel_artifacts(
    parsed_document: Any,
    kb_sn: str,
    asset_name: str,
    doc_id: str,
    upload_content: Callable[[str, str], str],
) -> Dict[str, Any]:
    return _persist_structured_artifacts(
        parsed_document,
        build_structured_excel_artifact_prefix(kb_sn, asset_name, doc_id),
        STRUCTURED_EXCEL_ARTIFACTS_KEY,
        STRUCTURED_EXCEL_METADATA_KEY,
        upload_content,
    )


def persist_structured_html_artifacts(
    parsed_document: Any,
    kb_sn: str,
    asset_name: str,
    doc_id: str,
    upload_content: Callable[[str, str], str],
) -> Dict[str, Any]:
    return _persist_structured_artifacts(
        parsed_document,
        build_structured_html_artifact_prefix(kb_sn, asset_name, doc_id),
        STRUCTURED_HTML_ARTIFACTS_KEY,
        STRUCTURED_HTML_METADATA_KEY,
        upload_content,
    )


def persist_structured_markdown_artifacts(
    parsed_document: Any,
    kb_sn: str,
    asset_name: str,
    doc_id: str,
    upload_content: Callable[[str, str], str],
) -> Dict[str, Any]:
    return _persist_structured_artifacts(
        parsed_document,
        build_structured_markdown_artifact_prefix(kb_sn, asset_name, doc_id),
        STRUCTURED_MARKDOWN_ARTIFACTS_KEY,
        STRUCTURED_MARKDOWN_METADATA_KEY,
        upload_content,
    )


def persist_parsed_markdown_artifact(
    parsed_document: Any,
    kb_sn: str,
    asset_name: str,
    doc_id: str,
    upload_content: Callable[[str, str], str],
) -> Dict[str, Any]:
    document_markdown = _parsed_document_markdown(parsed_document)
    if not document_markdown:
        return {}
    prefix = build_parsed_markdown_artifact_prefix(kb_sn, asset_name, doc_id)
    document_key = prefix + "document.md"
    upload_content(document_key, document_markdown)
    summary = _parsed_markdown_summary(prefix, document_key, parsed_document)
    parsed_document.metadata[PARSED_MARKDOWN_METADATA_KEY] = summary
    return summary


def _persist_structured_artifacts(
    parsed_document: Any,
    prefix: str,
    artifacts_key: str,
    metadata_key: str,
    upload_content: Callable[[str, str], str],
) -> Dict[str, Any]:
    artifacts = parsed_document.metadata.get(artifacts_key)
    if not artifacts:
        return {}
    document_key = _upload_document_markdown(prefix, artifacts, upload_content)
    _upload_sections(prefix, parsed_document, upload_content)
    table_refs = _upload_table_artifacts(prefix, artifacts.get("tables", []), upload_content)
    _apply_table_refs(parsed_document, table_refs)
    manifest_key = _upload_manifest(prefix, parsed_document, upload_content)
    parsed_document.metadata.pop(artifacts_key, None)
    summary = _artifact_summary(prefix, document_key, manifest_key, parsed_document, table_refs)
    parsed_document.metadata[metadata_key] = summary
    return summary


def _upload_document_markdown(prefix: str, artifacts: Dict[str, Any], upload_content):
    document_key = prefix + "document.md"
    upload_content(document_key, artifacts.get("document_markdown", ""))
    return document_key


def _upload_manifest(prefix: str, parsed_document: Any, upload_content) -> str:
    manifest_key = prefix + "manifest.json"
    upload_content(manifest_key, _json_text(_manifest_from_blocks(parsed_document.blocks)))
    return manifest_key


def _upload_table_artifacts(prefix: str, tables, upload_content) -> Dict[str, Dict[str, str]]:
    refs = {}
    for table in tables:
        table_id = table["table_id"]
        table_prefix = prefix + "tables/" + table_id
        table_refs = _table_refs(table_prefix)
        upload_content(table_refs["display_ref"], table.get("html", ""))
        upload_content(table_refs["table_json_ref"], _json_text(table.get("json", {})))
        upload_content(table_refs["llm_table_ref"], table.get("llm_markdown", ""))
        refs[table_id] = table_refs
    return refs


def _upload_sections(prefix: str, parsed_document: Any, upload_content) -> None:
    for section in _sections_from_blocks(getattr(parsed_document, "blocks", [])):
        section_ref = prefix + "sections/" + _safe_ref_name(section["section_id"]) + ".md"
        for block in section["blocks"]:
            block.metadata["section_id"] = section["section_id"]
            block.metadata["section_ref"] = section_ref
        upload_content(section_ref, _section_markdown(section["blocks"]))


def _sections_from_blocks(blocks) -> List[Dict[str, Any]]:
    sections = []
    section_by_key: Dict[str, Dict[str, Any]] = {}
    for block in blocks:
        key = _section_key(block)
        if key not in section_by_key:
            section = _new_section(block, len(sections) + 1)
            section_by_key[key] = section
            sections.append(section)
        section_by_key[key]["blocks"].append(block)
    return sections


def _new_section(block: Any, index: int) -> Dict[str, Any]:
    return {
        "section_id": block.metadata.get("section_id") or f"section_{index:03d}",
        "title": _section_title(block),
        "headers": list(block.metadata.get("headers") or []),
        "blocks": [],
    }


def _section_key(block: Any) -> str:
    if block.metadata.get("section_id"):
        return str(block.metadata["section_id"])
    headers = block.metadata.get("headers") or []
    if headers:
        return " / ".join(str(header) for header in headers if header)
    return str(block.metadata.get("title") or "document")


def _section_title(block: Any) -> str:
    headers = block.metadata.get("headers") or []
    if headers:
        return str(headers[-1])
    return str(block.metadata.get("title") or "")


def _section_markdown(blocks) -> str:
    return "\n\n".join(block.text for block in blocks if block.text).strip()


def _safe_ref_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "section"


def _table_refs(table_prefix: str) -> Dict[str, str]:
    return {
        "display_ref": table_prefix + ".html",
        "table_json_ref": table_prefix + ".json",
        "llm_table_ref": table_prefix + ".llm.md",
    }


def _apply_table_refs(parsed_document: Any, table_refs: Dict[str, Dict[str, str]]) -> None:
    for block in parsed_document.blocks:
        table_id = block.metadata.get("table_id")
        if table_id in table_refs:
            block.metadata.update(table_refs[table_id])


def _manifest_from_blocks(blocks) -> Dict[str, Any]:
    return {
        "blocks": [
            {
                "block_id": block.metadata.get("block_id"),
                "type": block.metadata.get("block_type", "text"),
                "title": block.metadata.get("title", ""),
                "headers": block.metadata.get("headers", []),
                "section_id": block.metadata.get("section_id"),
                "section_ref": block.metadata.get("section_ref"),
                "table_id": block.metadata.get("table_id"),
                "display_ref": block.metadata.get("display_ref"),
                "table_json_ref": block.metadata.get("table_json_ref"),
                "llm_table_ref": block.metadata.get("llm_table_ref"),
            }
            for block in blocks
        ],
        "sections": _manifest_sections(blocks),
    }


def _manifest_sections(blocks) -> List[Dict[str, Any]]:
    sections = []
    seen = set()
    for block in blocks:
        section_id = block.metadata.get("section_id")
        if not section_id or section_id in seen:
            continue
        seen.add(section_id)
        sections.append(_manifest_section(block, blocks))
    return sections


def _manifest_section(first_block: Any, blocks) -> Dict[str, Any]:
    section_id = first_block.metadata.get("section_id")
    section_blocks = [block for block in blocks if block.metadata.get("section_id") == section_id]
    return {
        "section_id": section_id,
        "title": _section_title(first_block),
        "headers": first_block.metadata.get("headers", []),
        "block_ids": [block.metadata.get("block_id") for block in section_blocks if block.metadata.get("block_id")],
        "section_ref": first_block.metadata.get("section_ref"),
    }


def _artifact_summary(prefix, document_key, manifest_key, parsed_document, table_refs):
    return {
        "artifact_prefix": prefix,
        "document_markdown_key": document_key,
        "manifest_key": manifest_key,
        "table_count": len(table_refs),
        "block_count": len(parsed_document.blocks),
    }


def _parsed_document_markdown(parsed_document: Any) -> str:
    if getattr(parsed_document, "text", ""):
        return parsed_document.text.strip()
    block_texts = [block.text for block in getattr(parsed_document, "blocks", []) if block.text]
    return "\n\n".join(block_texts).strip()


def _parsed_markdown_summary(prefix: str, document_key: str, parsed_document: Any) -> Dict[str, Any]:
    return {
        "artifact_prefix": prefix,
        "document_markdown_key": document_key,
        "block_count": len(getattr(parsed_document, "blocks", [])),
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
