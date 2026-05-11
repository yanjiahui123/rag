from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel

from rag_service.agent_retrieval.handles import AgentRetrievalHandleCodec
from rag_service.agent_retrieval.models import (
    DocumentOutlineRequest,
    DocumentOutlineResponse,
    DocumentSummary,
    OriginalTextBlock,
    OriginalTextRequest,
    OriginalTextResponse,
    OutlineSection,
    OutlineTable,
    SearchSlice,
    SearchSlicesRequest,
    SearchSlicesResponse,
    SectionRequest,
    SectionResponse,
    SliceActions,
    SliceHandles,
    SliceLocation,
    TableRequest,
    TableResponse,
)

MAX_SEARCH_TOP_K = 50
DEFAULT_SEARCH_TOP_K = 20
DEFAULT_MAX_CHARS = 12000

ARTIFACT_METADATA_KEYS = (
    "structured_excel",
    "structured_html",
    "structured_docx",
    "structured_markdown",
    "parsed_markdown",
)

T = TypeVar("T", bound=BaseModel)
Retriever = Callable[[SearchSlicesRequest, str, Any], Iterable[Any]]
ArtifactTextResolver = Callable[[str], Optional[str]]


class AgentRetrievalService:
    def __init__(
        self,
        retrieve_documents: Optional[Retriever] = None,
        artifact_text_resolver: Optional[ArtifactTextResolver] = None,
        handle_secret: Optional[str] = None,
        handle_codec: Optional[AgentRetrievalHandleCodec] = None,
    ):
        self.retrieve_documents = retrieve_documents or _default_retrieve_documents
        self.artifact_text_resolver = artifact_text_resolver or _default_artifact_text_resolver
        self.handle_codec = handle_codec or AgentRetrievalHandleCodec(secret=handle_secret)

    def search_slices(
        self,
        request: SearchSlicesRequest,
        uid: str,
        session: Any = None,
    ) -> SearchSlicesResponse:
        request = _coerce_model(SearchSlicesRequest, request)
        top_k = _bounded_top_k(request.top_k)
        documents = list(self.retrieve_documents(request, uid, session) or [])
        documents = sorted(documents, key=_document_score, reverse=True)[:top_k]
        return SearchSlicesResponse(
            query=request.query,
            kb_sn_list=_request_kb_sn_list(request),
            slices=[
                self._search_slice(document, rank=rank, uid=uid)
                for rank, document in enumerate(documents, start=1)
            ],
        )

    def get_document_outline(
        self,
        request: DocumentOutlineRequest,
        uid: str,
    ) -> DocumentOutlineResponse:
        request = _coerce_model(DocumentOutlineRequest, request)
        payload = self._decode_expected(request.document_handle, uid, "document")
        manifest = self._load_manifest(payload.get("manifest_key"))
        sections = [
            self._outline_section(section, payload, uid)
            for section in manifest.get("sections", [])
            if isinstance(section, dict) and section.get("section_id")
        ]
        tables = [
            self._outline_table(table_id, blocks, uid)
            for table_id, blocks in _table_blocks_by_id(manifest).items()
        ]
        return DocumentOutlineResponse(
            doc_id=_optional_string(payload.get("doc_id")),
            title=_optional_string(payload.get("title")),
            sections=sections,
            tables=tables,
        )

    def get_section(
        self,
        request: SectionRequest,
        uid: str,
    ) -> SectionResponse:
        request = _coerce_model(SectionRequest, request)
        payload = self._decode_expected(request.section_handle, uid, "section")
        section_ref = payload.get("section_ref")
        if not section_ref and payload.get("manifest_key"):
            section_ref = _section_ref_from_manifest(self._load_manifest(payload.get("manifest_key")), payload.get("section_id"))
        text = self._read_text(section_ref)
        text, truncated = _limit_text(text, request.max_chars)
        return SectionResponse(
            section_id=_optional_string(payload.get("section_id")),
            title=_optional_string(payload.get("title")),
            headers=_clean_string_list(payload.get("headers")),
            mode="full_section" if section_ref else "missing_section",
            text=text,
            truncated=truncated,
        )

    def get_table(
        self,
        request: TableRequest,
        uid: str,
    ) -> TableResponse:
        request = _coerce_model(TableRequest, request)
        payload = self._decode_expected(request.table_handle, uid, "table")
        content = self._table_content(payload, request.mode)
        content, truncated = _limit_text(content, request.max_chars)
        return TableResponse(
            table_id=_optional_string(payload.get("table_id")),
            title=_optional_string(payload.get("title")),
            mode=request.mode,
            row_count=_optional_int(payload.get("row_count")),
            content=content,
            truncated=truncated,
        )

    def get_original_text(
        self,
        request: OriginalTextRequest,
        uid: str,
    ) -> OriginalTextResponse:
        request = _coerce_model(OriginalTextRequest, request)
        payload = self._decode_expected(request.document_handle, uid, "document")
        manifest = self._load_manifest(payload.get("manifest_key"))
        blocks = _manifest_blocks(manifest)
        selected = _neighbor_blocks(blocks, request.center_block_id, request.before, request.after)
        if not selected:
            fallback_text = self._read_text(payload.get("document_markdown_key"))
            text, truncated = _limit_text(fallback_text, request.max_chars)
            selected_blocks = [OriginalTextBlock(block_id=None, block_index=None, text=text)] if text else []
            return OriginalTextResponse(
                doc_id=_optional_string(payload.get("doc_id")),
                mode="document_markdown",
                blocks=selected_blocks,
                truncated=truncated,
            )
        response_blocks, truncated = _bounded_original_blocks(selected, request.max_chars)
        return OriginalTextResponse(
            doc_id=_optional_string(payload.get("doc_id")),
            mode="neighbor_blocks",
            blocks=response_blocks,
            truncated=truncated,
        )

    def _search_slice(self, document: Any, rank: int, uid: str) -> SearchSlice:
        metadata = _extended_metadata(document)
        source = _document_source(document)
        doc = _document_summary(document, metadata, source)
        location = _slice_location(metadata)
        document_payload = _document_payload(doc, metadata)
        section_payload = _section_payload(doc, metadata)
        table_payload = _table_payload(doc, metadata)
        handles = SliceHandles(
            document_handle=self._encode(document_payload, uid) if document_payload else None,
            section_handle=self._encode(section_payload, uid) if section_payload else None,
            table_handle=self._encode(table_payload, uid) if table_payload else None,
        )
        actions = SliceActions(
            can_get_section=handles.section_handle is not None,
            can_get_table=handles.table_handle is not None,
            can_get_original_text=handles.document_handle is not None,
        )
        return SearchSlice(
            slice_id=_slice_id(doc, location, rank),
            rank=rank,
            text=getattr(document, "text", "") or "",
            score=_document_score(document),
            doc=doc,
            location=location,
            actions=actions,
            handles=handles,
            es_index=_optional_string(getattr(document, "es_index", None)),
            es_doc_id=_optional_string(getattr(document, "es_doc_id", None)),
        )

    def _outline_section(self, section: Dict[str, Any], document_payload: Dict[str, Any], uid: str) -> OutlineSection:
        payload = {
            "kind": "section",
            "doc_id": document_payload.get("doc_id"),
            "title": section.get("title"),
            "section_id": section.get("section_id"),
            "headers": _clean_string_list(section.get("headers")),
            "section_ref": section.get("section_ref"),
            "manifest_key": document_payload.get("manifest_key"),
        }
        return OutlineSection(
            section_id=str(section.get("section_id")),
            title=_optional_string(section.get("title")),
            headers=_clean_string_list(section.get("headers")),
            block_count=len(section.get("block_ids") or []),
            section_handle=self._encode(payload, uid),
        )

    def _outline_table(self, table_id: str, blocks: List[Dict[str, Any]], uid: str) -> OutlineTable:
        block = blocks[0]
        payload = _table_payload_from_block(block)
        return OutlineTable(
            table_id=table_id,
            title=_optional_string(block.get("title")),
            row_count=_optional_int(block.get("row_count")),
            table_handle=self._encode(payload, uid) if payload else None,
        )

    def _table_content(self, payload: Dict[str, Any], mode: str) -> str:
        if mode == "summary":
            return _table_summary(payload)
        ref = {
            "llm_text": payload.get("llm_table_ref"),
            "json": payload.get("table_json_ref"),
            "html": payload.get("display_ref"),
        }.get(mode)
        return self._read_text(ref)

    def _load_manifest(self, manifest_key: Optional[str]) -> Dict[str, Any]:
        text = self._read_text(manifest_key)
        if not text:
            return {}
        try:
            value = json.loads(text)
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def _read_text(self, ref: Optional[str]) -> str:
        if not ref:
            return ""
        text = self.artifact_text_resolver(ref)
        return text or ""

    def _encode(self, payload: Dict[str, Any], uid: str) -> str:
        return self.handle_codec.encode(payload, uid)

    def _decode_expected(self, handle: str, uid: str, kind: str) -> Dict[str, Any]:
        payload = self.handle_codec.decode(handle, expected_uid=uid)
        if payload.get("kind") != kind:
            raise PermissionError("retrieval handle has the wrong kind")
        return payload


def _default_retrieve_documents(request: SearchSlicesRequest, uid: str, session: Any = None) -> Iterable[Any]:
    if session is None:
        raise RuntimeError("agent retrieval search requires a database session when no retriever is injected")
    from rag_service.rag_app.service import knowledge_base_service

    kb_sn_list = _request_kb_sn_list(request)
    knowledge_bases = knowledge_base_service.permission_judge(session, kb_sn_list, uid)
    query_request = SimpleNamespace(
        question=request.query,
        kb_sn=None,
        kb_sn_list=kb_sn_list,
        uid=uid,
        top_k=_bounded_top_k(request.top_k),
        request_id=None,
        user_custom_retrieve_config=None,
        historical_questions=[],
    )
    documents = []
    manager = knowledge_base_service.get_vector_store_manager()
    for knowledge_base in knowledge_bases:
        retrieve_config = knowledge_base_service.get_retrieve_param_by_kb_config_and_request(knowledge_base, query_request)
        documents.extend(
            manager.retrieve(
                query_request.question,
                retrieve_config.top_k,
                knowledge_base_service.get_embedding_model_and_vector_stores(session, knowledge_base.sn),
                retrieve_config.document_score_threshold,
                collect_info=False,
                analyzer=knowledge_base.analyzer,
                query_strategy=retrieve_config.query_strategy,
                request_id=None,
                background_tasks=None,
            )
        )
    return sorted(documents, key=_document_score, reverse=True)[: query_request.top_k]


def _default_artifact_text_resolver(object_key: str) -> Optional[str]:
    from rag_service.utils.his_util.obs_util import download_file_as_bytes

    if not object_key:
        return None
    return download_file_as_bytes(object_key).decode("utf-8")


def _request_kb_sn_list(request: SearchSlicesRequest) -> List[str]:
    if request.kb_sn_list:
        return list(request.kb_sn_list)
    return [request.kb_sn] if request.kb_sn else []


def _bounded_top_k(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_SEARCH_TOP_K
    return min(max(parsed, 1), MAX_SEARCH_TOP_K)


def _coerce_model(model_type: Type[T], value: Any) -> T:
    if isinstance(value, model_type):
        return value
    if hasattr(model_type, "model_validate"):
        return model_type.model_validate(value)
    return model_type(**value)


def _extended_metadata(document: Any) -> Dict[str, Any]:
    metadata = getattr(document, "metadata", None)
    extended_metadata = getattr(metadata, "extended_metadata", None)
    if isinstance(extended_metadata, dict):
        return dict(extended_metadata)
    if isinstance(metadata, dict) and isinstance(metadata.get("extended_metadata"), dict):
        return dict(metadata["extended_metadata"])
    return {}


def _document_source(document: Any) -> str:
    metadata = getattr(document, "metadata", None)
    if isinstance(metadata, dict):
        return str(metadata.get("source") or "")
    return str(getattr(metadata, "source", "") or "")


def _document_score(document: Any) -> float:
    try:
        return float(getattr(document, "score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _document_summary(document: Any, metadata: Dict[str, Any], source: str) -> DocumentSummary:
    title = _optional_string(metadata.get("title")) or _retrieve_title(document) or source
    return DocumentSummary(
        doc_id=_optional_string(metadata.get("doc_id")),
        title=title,
        source=source,
        asset_name=_optional_string(metadata.get("asset_name")),
        kb_sn=_optional_string(metadata.get("kb_sn")),
    )


def _retrieve_title(document: Any) -> Optional[str]:
    retrieve_metadata = getattr(getattr(document, "metadata", None), "retrieve_metadata", None)
    return _optional_string(getattr(retrieve_metadata, "title", None))


def _slice_location(metadata: Dict[str, Any]) -> SliceLocation:
    headers = _clean_string_list(metadata.get("headers"))
    section_path = _optional_string(metadata.get("section_path")) or (" > ".join(headers) if headers else None)
    return SliceLocation(
        block_id=_optional_string(metadata.get("block_id")),
        block_index=_optional_int(metadata.get("block_index")),
        block_type=_optional_string(metadata.get("block_type")),
        section_id=_optional_string(metadata.get("section_id")),
        section_title=_optional_string(metadata.get("section_title")) or _optional_string(metadata.get("title")),
        section_path=section_path,
        table_id=_optional_string(metadata.get("table_id")),
        row_range=_row_range_list(metadata.get("row_range")),
    )


def _document_payload(doc: DocumentSummary, metadata: Dict[str, Any]) -> Dict[str, Any]:
    artifact = _document_artifact(metadata)
    payload = {
        "kind": "document",
        "doc_id": doc.doc_id,
        "title": doc.title,
        "source": doc.source,
        "asset_name": doc.asset_name,
        "kb_sn": doc.kb_sn,
        "manifest_key": artifact.get("manifest_key") or artifact.get("manifest"),
        "document_markdown_key": artifact.get("document_markdown_key") or artifact.get("document_markdown"),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _section_payload(doc: DocumentSummary, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    section_ref = _optional_string(metadata.get("section_ref"))
    section_id = _optional_string(metadata.get("section_id"))
    if not section_ref and not section_id:
        return None
    artifact = _document_artifact(metadata)
    payload = {
        "kind": "section",
        "doc_id": doc.doc_id,
        "title": metadata.get("section_title") or metadata.get("title"),
        "section_id": section_id,
        "headers": _clean_string_list(metadata.get("headers")),
        "section_ref": section_ref,
        "manifest_key": artifact.get("manifest_key") or artifact.get("manifest"),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _table_payload(doc: DocumentSummary, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not metadata.get("table_id"):
        return None
    table = metadata.get("table") if isinstance(metadata.get("table"), dict) else {}
    payload = {
        "kind": "table",
        "doc_id": doc.doc_id,
        "table_id": _optional_string(metadata.get("table_id")),
        "title": metadata.get("title") or table.get("title"),
        "row_count": metadata.get("row_count") or table.get("row_count"),
        "display_ref": metadata.get("display_ref"),
        "table_json_ref": metadata.get("table_json_ref"),
        "llm_table_ref": metadata.get("llm_table_ref"),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _document_artifact(metadata: Dict[str, Any]) -> Dict[str, Any]:
    for key in ARTIFACT_METADATA_KEYS:
        value = metadata.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _slice_id(doc: DocumentSummary, location: SliceLocation, rank: int) -> str:
    doc_id = doc.doc_id or doc.source or "document"
    item_id = location.block_id or location.table_id or location.section_id or f"slice_{rank}"
    return f"{doc_id}:{item_id}"


def _manifest_blocks(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks = manifest.get("blocks")
    return [dict(block) for block in blocks if isinstance(block, dict)] if isinstance(blocks, list) else []


def _table_blocks_by_id(manifest: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for block in _manifest_blocks(manifest):
        table_id = _optional_string(block.get("table_id"))
        if table_id:
            grouped.setdefault(table_id, []).append(block)
    return grouped


def _table_payload_from_block(block: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "kind": "table",
        "table_id": block.get("table_id"),
        "title": block.get("title"),
        "row_count": block.get("row_count"),
        "display_ref": block.get("display_ref"),
        "table_json_ref": block.get("table_json_ref"),
        "llm_table_ref": block.get("llm_table_ref"),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _section_ref_from_manifest(manifest: Dict[str, Any], section_id: Optional[str]) -> Optional[str]:
    for section in manifest.get("sections") or []:
        if isinstance(section, dict) and section.get("section_id") == section_id:
            return _optional_string(section.get("section_ref"))
    return None


def _neighbor_blocks(
    blocks: Sequence[Dict[str, Any]],
    center_block_id: Optional[str],
    before: int,
    after: int,
) -> List[Dict[str, Any]]:
    if not blocks:
        return []
    if center_block_id:
        center_index = next((index for index, block in enumerate(blocks) if block.get("block_id") == center_block_id), 0)
    else:
        center_index = 0
    start = max(center_index - max(int(before or 0), 0), 0)
    end = min(center_index + max(int(after or 0), 0) + 1, len(blocks))
    return [dict(block) for block in blocks[start:end]]


def _bounded_original_blocks(blocks: Sequence[Dict[str, Any]], max_chars: int) -> tuple[List[OriginalTextBlock], bool]:
    budget = max(int(max_chars or DEFAULT_MAX_CHARS), 0)
    used = 0
    response_blocks: List[OriginalTextBlock] = []
    truncated = False
    for index, block in enumerate(blocks):
        text = _optional_string(block.get("text")) or _optional_string(block.get("content")) or ""
        remaining = budget - used if budget else len(text)
        if budget and remaining <= 0:
            truncated = True
            break
        if budget and len(text) > remaining:
            text = text[:remaining]
            truncated = True
        used += len(text)
        response_blocks.append(
            OriginalTextBlock(
                block_id=_optional_string(block.get("block_id")),
                block_index=_optional_int(block.get("block_index")) if block.get("block_index") is not None else index,
                text=text,
            )
        )
    return response_blocks, truncated


def _table_summary(payload: Dict[str, Any]) -> str:
    parts = [f"table_id: {payload.get('table_id')}"]
    if payload.get("title"):
        parts.append(f"title: {payload.get('title')}")
    if payload.get("row_count") is not None:
        parts.append(f"row_count: {payload.get('row_count')}")
    return "\n".join(parts)


def _limit_text(text: str, max_chars: int) -> tuple[str, bool]:
    limit = max(int(max_chars or 0), 0)
    if limit and len(text) > limit:
        return text[:limit], True
    return text, False


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _row_range_list(value: Any) -> Optional[List[int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError):
        return None


def _clean_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]

