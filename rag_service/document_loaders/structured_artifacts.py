from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional

STRUCTURED_DOCX_ARTIFACTS_KEY = "_structured_docx_artifacts"
STRUCTURED_DOCX_METADATA_KEY = "structured_docx"
STRUCTURED_EXCEL_ARTIFACTS_KEY = "_structured_excel_artifacts"
STRUCTURED_EXCEL_METADATA_KEY = "structured_excel"


def build_structured_docx_artifact_prefix(kb_sn: str, asset_name: str, doc_id: str) -> str:
    return _build_document_artifact_prefix(doc_id, "structured_docx")


def build_structured_excel_artifact_prefix(kb_sn: str, asset_name: str, doc_id: str) -> str:
    return _build_document_artifact_prefix(doc_id, "structured_excel")


def _build_document_artifact_prefix(doc_id: str, artifact_type: str) -> str:
    return f"{doc_id}/{artifact_type}/"


def is_safe_structured_artifact_prefix(prefix: Optional[str]) -> bool:
    if not prefix:
        return False
    return bool(re.match(r"^[^/]+/(structured_docx|structured_excel)/$", prefix))


def extract_structured_docx_artifact_prefix(extended_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    return _extract_artifact_prefix(extended_metadata, STRUCTURED_DOCX_METADATA_KEY)


def extract_structured_excel_artifact_prefix(extended_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    return _extract_artifact_prefix(extended_metadata, STRUCTURED_EXCEL_METADATA_KEY)


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
                "table_id": block.metadata.get("table_id"),
                "display_ref": block.metadata.get("display_ref"),
                "table_json_ref": block.metadata.get("table_json_ref"),
                "llm_table_ref": block.metadata.get("llm_table_ref"),
            }
            for block in blocks
        ]
    }


def _artifact_summary(prefix, document_key, manifest_key, parsed_document, table_refs):
    return {
        "artifact_prefix": prefix,
        "document_markdown_key": document_key,
        "manifest_key": manifest_key,
        "table_count": len(table_refs),
        "block_count": len(parsed_document.blocks),
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
