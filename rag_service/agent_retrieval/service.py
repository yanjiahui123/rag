from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel

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
DOCUMENT_ARTIFACT_FILES = ("manifest.json", "document.md")
TABLE_ARTIFACT_SUFFIXES = (".llm.md", ".json", ".html")

T = TypeVar("T", bound=BaseModel)
Retriever = Callable[[SearchSlicesRequest, str, Any], Iterable[Any]]
ArtifactTextResolver = Callable[[str], Optional[str]]
SearchLogWriter = Callable[[Dict[str, Any], Any], None]
RequestIdFactory = Callable[[], str]
logger = logging.getLogger(__name__)


class RetrievalOutcome(BaseModel):
    documents: List[Any]
    diagnostics: Dict[str, Any]


class AgentRetrievalService:
    def __init__(
        self,
        retrieve_documents: Optional[Retriever] = None,
        artifact_text_resolver: Optional[ArtifactTextResolver] = None,
        search_log_writer: Optional[SearchLogWriter] = None,
        request_id_factory: Optional[RequestIdFactory] = None,
    ):
        self.uses_default_retriever = retrieve_documents is None
        self.retrieve_documents = retrieve_documents or _default_retrieve_documents
        self.artifact_text_resolver = artifact_text_resolver or _default_artifact_text_resolver
        self.search_log_writer = search_log_writer or _default_search_log_writer
        self.request_id_factory = request_id_factory or (lambda: uuid.uuid4().hex)

    def search_slices(
        self,
        request: SearchSlicesRequest,
        uid: str,
        session: Any = None,
        request_id: Optional[str] = None,
    ) -> SearchSlicesResponse:
        request = _coerce_model(SearchSlicesRequest, request)
        request_id = request_id or self.request_id_factory()
        top_k = _bounded_top_k(request.top_k)
        started_at = _utcnow()
        retrieve_started_at = _utcnow()
        diagnostics = _base_search_diagnostics(request, top_k)
        try:
            documents = self._search_documents(request, uid, session, top_k, diagnostics)
            response = self._search_response(request_id, request, documents, uid)
        except Exception as exc:
            self._record_search_result(
                request_id, uid, request, None, diagnostics, started_at, retrieve_started_at, session, str(exc)
            )
            raise
        diagnostics["returned_slice_count"] = len(response.slices)
        self._record_search_result(
            request_id, uid, request, response, diagnostics, started_at, retrieve_started_at, session
        )
        return response

    def _search_documents(
        self,
        request: SearchSlicesRequest,
        uid: str,
        session: Any,
        top_k: int,
        diagnostics: Dict[str, Any],
    ) -> List[Any]:
        if self.uses_default_retriever:
            outcome = _default_retrieve_outcome(request, uid, session)
            diagnostics.update(outcome.diagnostics)
            return outcome.documents
        candidates = list(self.retrieve_documents(request, uid, session) or [])
        diagnostics.update(
            {"candidate_count_before_dedup": len(candidates), "candidate_count_after_dedup": len(candidates)}
        )
        return sorted(candidates, key=_document_score, reverse=True)[:top_k]

    def _search_response(
        self, request_id: str, request: SearchSlicesRequest, documents: List[Any], uid: str
    ) -> SearchSlicesResponse:
        return SearchSlicesResponse(
            request_id=request_id,
            query=request.query,
            kb_sn_list=_request_kb_sn_list(request),
            slices=[self._search_slice(document, rank=rank, uid=uid) for rank, document in enumerate(documents, start=1)],
        )

    def _record_search_result(
        self,
        request_id: str,
        uid: str,
        request: SearchSlicesRequest,
        response: Optional[SearchSlicesResponse],
        diagnostics: Dict[str, Any],
        started_at: datetime,
        retrieve_started_at: datetime,
        session: Any,
        error_reason: Optional[str] = None,
    ) -> None:
        self._write_search_log(
            _search_log_record(
                request_id, uid, request, response, diagnostics, started_at, retrieve_started_at, _utcnow(), error_reason
            ),
            session,
        )

    def _write_search_log(self, record: Dict[str, Any], session: Any) -> None:
        try:
            self.search_log_writer(record, session)
        except Exception as exc:
            logger.warning("agent search_slices log persistence failed: %s", exc)

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
            document_handle=_obs_handle(document_payload),
            section_handle=_obs_handle(section_payload),
            table_handle=_obs_handle(table_payload),
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
            section_handle=_obs_handle(payload),
        )

    def _outline_table(self, table_id: str, blocks: List[Dict[str, Any]], uid: str) -> OutlineTable:
        block = blocks[0]
        payload = _table_payload_from_block(block)
        return OutlineTable(
            table_id=table_id,
            title=_optional_string(block.get("title")),
            row_count=_optional_int(block.get("row_count")),
            table_handle=_obs_handle(payload),
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

    def _decode_expected(self, handle: str, uid: str, kind: str) -> Dict[str, Any]:
        payload = _obs_payload(handle, kind)
        if payload is None:
            raise PermissionError("invalid retrieval handle")
        if payload.get("kind") != kind:
            raise PermissionError("retrieval handle has the wrong kind")
        return payload


def _default_retrieve_documents(request: SearchSlicesRequest, uid: str, session: Any = None) -> Iterable[Any]:
    return _default_retrieve_outcome(request, uid, session).documents


def _default_retrieve_outcome(request: SearchSlicesRequest, uid: str, session: Any = None) -> RetrievalOutcome:
    if session is None:
        raise RuntimeError("agent retrieval search requires a database session when no retriever is injected")
    from rag_service.rag_app.service import knowledge_base_service

    kb_sn_list = _request_kb_sn_list(request)
    knowledge_bases = knowledge_base_service.permission_judge(session, kb_sn_list, uid)
    final_top_k = _bounded_top_k(request.top_k)
    candidate_top_k = _candidate_top_k(request)
    diagnostics = _base_search_diagnostics(request, final_top_k)
    query_request = _agent_query_request(request, uid, kb_sn_list, candidate_top_k)
    if not knowledge_bases:
        return _empty_retrieval_outcome(diagnostics)
    retrieve_config = _agent_retrieve_config(knowledge_base_service, knowledge_bases, query_request)
    documents = _agent_backend_documents(
        knowledge_base_service, request, session, knowledge_bases, query_request, retrieve_config, diagnostics
    )
    return _finalize_retrieval_outcome(
        knowledge_base_service,
        request,
        knowledge_bases,
        query_request,
        retrieve_config,
        documents,
        diagnostics,
    )


def _agent_query_request(
    request: SearchSlicesRequest, uid: str, kb_sn_list: List[str], candidate_top_k: int
) -> SimpleNamespace:
    return SimpleNamespace(
        question=request.query,
        kb_sn=None,
        kb_sn_list=kb_sn_list,
        uid=uid,
        top_k=candidate_top_k,
        request_id=None,
        user_custom_retrieve_config=None,
        historical_questions=[],
    )


def _empty_retrieval_outcome(diagnostics: Dict[str, Any]) -> RetrievalOutcome:
    diagnostics.update(
        {"libing_analyzer_group_count": 0, "candidate_count_before_dedup": 0, "candidate_count_after_dedup": 0}
    )
    return RetrievalOutcome(documents=[], diagnostics=diagnostics)


def _agent_retrieve_config(knowledge_base_service: Any, knowledge_bases: List[Any], query_request: Any) -> Any:
    if len(knowledge_bases) == 1:
        return knowledge_base_service.get_retrieve_param_by_kb_config_and_request(knowledge_bases[0], query_request)
    return knowledge_base_service.get_multi_kb_retrieve_param(knowledge_bases, query_request)


def _agent_backend_documents(
    knowledge_base_service: Any,
    request: SearchSlicesRequest,
    session: Any,
    knowledge_bases: List[Any],
    query_request: Any,
    retrieve_config: Any,
    diagnostics: Dict[str, Any],
) -> List[Any]:
    if request.retrieval_backend == "ipd":
        return _agent_ipd_documents(knowledge_base_service, knowledge_bases, query_request, retrieve_config, diagnostics)
    return _agent_libing_documents(knowledge_base_service, session, knowledge_bases, query_request, retrieve_config, diagnostics)


def _agent_ipd_documents(
    knowledge_base_service: Any,
    knowledge_bases: List[Any],
    query_request: Any,
    retrieve_config: Any,
    diagnostics: Dict[str, Any],
) -> List[Any]:
    mapped_kbs = [knowledge_base for knowledge_base in knowledge_bases if getattr(knowledge_base, "ipd_rag_kb_id", None)]
    diagnostics["ipd_mapped_kb_sn_list"] = [knowledge_base.sn for knowledge_base in mapped_kbs]
    diagnostics["ipd_skipped_unmapped_kb_sn_list"] = [
        knowledge_base.sn for knowledge_base in knowledge_bases if knowledge_base not in mapped_kbs
    ]
    if not mapped_kbs:
        return []
    return list(
        knowledge_base_service.retrieve_documents_from_ipd_rag(
            query_request.question, query_request.top_k, retrieve_config.query_strategy, mapped_kbs
        )
    )


def _agent_libing_documents(
    knowledge_base_service: Any,
    session: Any,
    knowledge_bases: List[Any],
    query_request: Any,
    retrieve_config: Any,
    diagnostics: Dict[str, Any],
) -> List[Any]:
    manager = knowledge_base_service.get_vector_store_manager()
    diagnostics["libing_manager_top_k"] = retrieve_config.top_k
    if len(knowledge_bases) == 1:
        diagnostics["libing_analyzer_group_count"] = 1
        stores = knowledge_base_service.get_embedding_model_and_vector_stores(session, knowledge_bases[0].sn)
        _record_libing_store_diagnostics(diagnostics, stores)
        return _agent_libing_group_documents(
            manager, query_request, retrieve_config, stores, knowledge_bases[0].analyzer
        )
    grouped_stores = knowledge_base_service.get_grouped_vector_stores_by_knowledge_base_and_asset(
        session, {knowledge_base.sn: [] for knowledge_base in knowledge_bases}
    )
    diagnostics["libing_analyzer_group_count"] = len(grouped_stores)
    documents = []
    for analyzer, stores in grouped_stores.items():
        _record_libing_store_diagnostics(diagnostics, stores)
        documents.extend(_agent_libing_group_documents(manager, query_request, retrieve_config, stores, analyzer))
    return documents


def _record_libing_store_diagnostics(diagnostics: Dict[str, Any], stores: Any) -> None:
    if not hasattr(stores, "items"):
        return
    for embedding_model, search_info in stores.items():
        model_name = str(getattr(embedding_model, "value", embedding_model))
        diagnostics["libing_embedding_model_group_count"] += 1
        if model_name not in diagnostics["libing_embedding_model_list"]:
            diagnostics["libing_embedding_model_list"].append(model_name)
        indexes = (
            search_info.get("vs_indexes")
            if isinstance(search_info, dict)
            else getattr(search_info, "vs_indexes", None)
        )
        if indexes:
            diagnostics["libing_vector_store_index_count"] += len(indexes)


def _agent_libing_group_documents(
    manager: Any, query_request: Any, retrieve_config: Any, stores: Any, analyzer: Any
) -> List[Any]:
    return list(
        manager.retrieve(
            query_request.question,
            retrieve_config.top_k,
            stores,
            retrieve_config.document_score_threshold,
            collect_info=False,
            analyzer=analyzer,
            query_strategy=retrieve_config.query_strategy,
            request_id=None,
            background_tasks=None,
        )
    )


def _finalize_retrieval_outcome(
    knowledge_base_service: Any,
    request: SearchSlicesRequest,
    knowledge_bases: List[Any],
    query_request: Any,
    retrieve_config: Any,
    documents: List[Any],
    diagnostics: Dict[str, Any],
) -> RetrievalOutcome:
    diagnostics["candidate_count_before_dedup"] = len(documents)
    should_deduplicate = request.enable_rerank or request.retrieval_backend == "ipd" or len(knowledge_bases) > 1
    diagnostics["deduplication_applied"] = should_deduplicate
    unique_documents = _deduplicate_documents(documents) if should_deduplicate else list(documents)
    diagnostics["candidate_count_after_dedup"] = len(unique_documents)
    if request.enable_rerank and unique_documents:
        results, degraded = _rerank_documents(
            knowledge_base_service, query_request.question, unique_documents, _bounded_top_k(request.top_k), retrieve_config
        )
        diagnostics["rerank_degraded"] = degraded
        return RetrievalOutcome(documents=results, diagnostics=diagnostics)
    results = sorted(unique_documents, key=_document_score, reverse=True)[: _bounded_top_k(request.top_k)]
    return RetrievalOutcome(documents=results, diagnostics=diagnostics)


def _default_artifact_text_resolver(object_key: str) -> Optional[str]:
    from rag_service.utils.his_util.obs_util import download_file_as_bytes

    if not object_key:
        return None
    return download_file_as_bytes(object_key).decode("utf-8")


def _base_search_diagnostics(request: SearchSlicesRequest, final_top_k: int) -> Dict[str, Any]:
    return {
        "retrieval_backend": request.retrieval_backend,
        "enable_rerank": request.enable_rerank,
        "requested_top_k": request.top_k,
        "final_top_k": final_top_k,
        "candidate_top_k": _candidate_top_k(request),
        "kb_sn_list": _request_kb_sn_list(request),
        "rerank_requested": request.enable_rerank,
        "rerank_degraded": False,
        "ipd_mapped_kb_sn_list": [],
        "ipd_skipped_unmapped_kb_sn_list": [],
        "libing_embedding_model_group_count": 0,
        "libing_embedding_model_list": [],
        "libing_vector_store_index_count": 0,
        "libing_manager_top_k": None,
        "query_preprocessing": "raw_request_query",
        "rerank_control": "enable_rerank",
    }


def _candidate_top_k(request: SearchSlicesRequest) -> int:
    return _bounded_top_k(request.top_k)


def _deduplicate_documents(documents: Iterable[Any]) -> List[Any]:
    seen = set()
    unique_documents = []
    for document in documents:
        text = getattr(document, "text", "")
        if text not in seen:
            unique_documents.append(document)
            seen.add(text)
    return unique_documents


def _rerank_documents(
    knowledge_base_service: Any,
    question: str,
    documents: List[Any],
    top_k: int,
    retrieve_config: Any,
) -> tuple[List[Any], bool]:
    pairs = [(question, knowledge_base_service.get_rerank_format(document)) for document in documents]
    try:
        scores = knowledge_base_service.rerank_embedding(pairs, retrieve_config.rerank_model)
    except Exception as exc:
        logger.warning("agent search_slices rerank degraded: %s", exc)
        return sorted(documents, key=_document_score, reverse=True)[:top_k], True
    ranked_documents = []
    for score, document in sorted(zip(scores, documents), key=lambda item: item[0], reverse=True):
        if score > retrieve_config.document_score_threshold and len(ranked_documents) < top_k:
            document.score = score
            ranked_documents.append(document)
    return ranked_documents, False


def _search_log_record(
    request_id: str,
    uid: str,
    request: SearchSlicesRequest,
    response: Optional[SearchSlicesResponse],
    diagnostics: Dict[str, Any],
    started_at: datetime,
    retrieve_started_at: datetime,
    ended_at: datetime,
    error_reason: Optional[str] = None,
) -> Dict[str, Any]:
    retrieve_result = []
    if response is not None:
        retrieve_result = [_dump_model(slice_) for slice_ in response.slices]
    return {
        "request_id": request_id,
        "user_id": uid,
        "method_name": "agent.retrieval.search_slices",
        "kb_sn": request.kb_sn,
        "question": request.query,
        "request_start_time": started_at,
        "request_end_time": ended_at,
        "request_time_use": (ended_at - started_at).total_seconds(),
        "retrieve_start_time": retrieve_started_at,
        "retrieve_end_time": ended_at,
        "retrieve_time_use": (ended_at - retrieve_started_at).total_seconds(),
        "retrieve_result": json.dumps(retrieve_result, ensure_ascii=False),
        "error_reason": error_reason,
        "extra_info": dict(diagnostics),
    }


def _default_search_log_writer(record: Dict[str, Any], session: Any) -> None:
    if session is None:
        return
    from rag_service.models.database.models import RequestResponseLog

    unused_response_fields = {
        "aigc_record_id": None,
        "load_non_stream_llm_start_time": None,
        "load_non_stream_llm_end_time": None,
        "load_non_stream_llm_time_use": None,
        "load_non_stream_llm_prompt": None,
        "load_non_stream_llm_result": None,
        "load_stream_llm_start_time": None,
        "load_stream_llm_end_time": None,
        "load_stream_llm_first_token_time": None,
        "load_stream_llm_first_token_time_use": None,
        "load_stream_llm_time_use": None,
        "load_stream_llm_prompt": None,
        "load_stream_llm_result": None,
        "answer_user_want": None,
        "answer_source": None,
        "acceptance": None,
        "score": None,
        "rewrite_question": None,
        "question_id": None,
    }
    row = RequestResponseLog(**{**unused_response_fields, **record})
    if hasattr(session, "merge"):
        session.merge(row)
    else:
        session.add(row)
    session.commit()


def _dump_model(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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


def _obs_handle(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not payload:
        return None
    kind = payload.get("kind")
    if kind == "document":
        return _optional_string(payload.get("manifest_key")) or _optional_string(payload.get("document_markdown_key"))
    if kind == "section":
        return _optional_string(payload.get("section_ref"))
    if kind == "table":
        return (
            _optional_string(payload.get("llm_table_ref"))
            or _optional_string(payload.get("table_json_ref"))
            or _optional_string(payload.get("display_ref"))
        )
    return None


def _obs_payload(handle: str, kind: str) -> Optional[Dict[str, Any]]:
    object_key = (handle or "").strip()
    parts = _structured_artifact_key_parts(object_key)
    if parts is None:
        return None
    if kind == "document":
        return _document_obs_payload(object_key, parts)
    if kind == "section":
        return _section_obs_payload(object_key, parts)
    if kind == "table":
        return _table_obs_payload(object_key, parts)
    return None


def _structured_artifact_key_parts(object_key: str) -> Optional[List[str]]:
    if not object_key or "\\" in object_key:
        return None
    parts = object_key.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    if _artifact_type_index(parts) is None:
        return None
    return parts


def _document_obs_payload(object_key: str, parts: List[str]) -> Optional[Dict[str, Any]]:
    artifact_index = _artifact_type_index(parts)
    if artifact_index is None:
        return None
    file_index = artifact_index + 1
    if len(parts) != file_index + 1 or parts[file_index] not in DOCUMENT_ARTIFACT_FILES:
        return None
    prefix = _document_artifact_prefix(object_key)
    payload = _artifact_payload_base(parts, artifact_index)
    payload.update({
        "kind": "document",
        "manifest_key": prefix + "manifest.json",
        "document_markdown_key": prefix + "document.md",
    })
    return payload


def _section_obs_payload(object_key: str, parts: List[str]) -> Optional[Dict[str, Any]]:
    artifact_index = _artifact_type_index(parts)
    if artifact_index is None:
        return None
    section_dir_index = artifact_index + 1
    section_file_index = artifact_index + 2
    if (
        len(parts) != section_file_index + 1
        or parts[section_dir_index] != "sections"
        or not parts[section_file_index].endswith(".md")
    ):
        return None
    payload = _artifact_payload_base(parts, artifact_index)
    payload.update({
        "kind": "section",
        "section_id": parts[section_file_index][:-3],
        "section_ref": object_key,
    })
    return payload


def _table_obs_payload(object_key: str, parts: List[str]) -> Optional[Dict[str, Any]]:
    artifact_index = _artifact_type_index(parts)
    if artifact_index is None:
        return None
    table_dir_index = artifact_index + 1
    table_file_index = artifact_index + 2
    if len(parts) != table_file_index + 1 or parts[table_dir_index] != "tables":
        return None
    stem = _table_ref_stem(object_key)
    if stem is None:
        return None
    table_id = stem.rsplit("/", 1)[-1]
    payload = _artifact_payload_base(parts, artifact_index)
    payload.update({
        "kind": "table",
        "table_id": table_id,
        "display_ref": stem + ".html",
        "table_json_ref": stem + ".json",
        "llm_table_ref": stem + ".llm.md",
    })
    return payload


def _artifact_type_index(parts: List[str]) -> Optional[int]:
    if len(parts) >= 3 and parts[1] in ARTIFACT_METADATA_KEYS:
        return 1
    if len(parts) >= 4 and parts[2] in ARTIFACT_METADATA_KEYS:
        return 2
    if len(parts) >= 5 and parts[2] == "artifacts" and parts[3] in ARTIFACT_METADATA_KEYS:
        return 3
    return None


def _artifact_payload_base(parts: List[str], artifact_index: int) -> Dict[str, Any]:
    if artifact_index == 3:
        return {
            "knowledge_base_asset_id": parts[0],
            "doc_id": parts[1],
        }
    payload = {"doc_id": parts[artifact_index - 1]}
    if artifact_index == 2:
        payload["knowledge_base_asset_id"] = parts[0]
    return payload


def _document_artifact_prefix(object_key: str) -> str:
    return object_key.rsplit("/", 1)[0] + "/"


def _table_ref_stem(object_key: str) -> Optional[str]:
    for suffix in TABLE_ARTIFACT_SUFFIXES:
        if object_key.endswith(suffix):
            return object_key[: -len(suffix)]
    return None


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
