from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

ARTIFACT_METADATA_KEYS = (
    "structured_excel",
    "structured_html",
    "structured_docx",
    "structured_markdown",
    "parsed_markdown",
)

REF_FIELD_MAP = {
    "display_ref": "display",
    "table_json_ref": "table_json",
    "llm_table_ref": "llm_table",
    "section_ref": "section",
}

SECTION_MERGE_MIN_BLOCKS = 2
SECTION_FULL_EXPAND_RATIO = 0.7
MAX_INLINE_SECTION_CHARS = 12000
LOGGER = logging.getLogger(__name__)


class EvidencePackageOptions(BaseModel):
    package_top_k: int = 5
    max_evidence_per_package: int = 6
    artifact_mode: str = "key"
    enable_table_expansion: bool = True
    table_expand_ratio_threshold: float = 0.7
    max_full_table_rows: int = 500
    max_inline_table_chars: int = 40000


class DocumentArtifact(BaseModel):
    artifact_type: str
    artifact_prefix: str
    document_markdown: Optional[str] = None
    manifest: Optional[str] = None
    table_count: int = 0
    block_count: int = 0
    raw: Dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    text: str
    score: float
    evidence_type: Optional[str] = None
    mode: Optional[str] = None
    block_id: Optional[str] = None
    block_index: Optional[int] = None
    block_type: Optional[str] = None
    table_id: Optional[str] = None
    section_id: Optional[str] = None
    row_range: Optional[Tuple[int, int]] = None
    row_count: Optional[int] = None
    hit_rows: Optional[int] = None
    hit_blocks: Optional[int] = None
    hit_ratio: Optional[float] = None
    hit_row_ranges: List[List[int]] = Field(default_factory=list)
    merged_block_ids: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    table: Optional[Dict[str, Any]] = Field(default=None, exclude=True)
    refs: Dict[str, str] = Field(default_factory=dict, exclude=True)
    artifact_keys: Dict[str, str] = Field(default_factory=dict, exclude=True)
    es_index: Optional[str] = None
    es_doc_id: Optional[str] = None


class TableArtifact(BaseModel):
    type: str = "table"
    mode: str
    table_id: str
    title: Optional[str] = None
    row_count: int = 0
    hit_rows: int = 0
    hit_row_ranges: List[List[int]] = Field(default_factory=list)
    hit_blocks: int = 0
    hit_ratio: float = 0.0
    score: float = 0.0
    expanded: bool = False
    table: Optional[Dict[str, Any]] = None
    refs: Dict[str, str] = Field(default_factory=dict)
    artifact_keys: Dict[str, str] = Field(default_factory=dict, exclude=True)


class DocumentSectionArtifact(BaseModel):
    type: str = "section"
    mode: str = "merged_section"
    section_id: str
    title: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    hit_blocks: int = 0
    score: float = 0.0
    merged_block_ids: List[str] = Field(default_factory=list)
    refs: Dict[str, str] = Field(default_factory=dict)
    artifact_keys: Dict[str, str] = Field(default_factory=dict, exclude=True)


class PackageArtifacts(BaseModel):
    document: Optional[DocumentArtifact] = None
    sections: List[DocumentSectionArtifact] = Field(default_factory=list)
    tables: List[TableArtifact] = Field(default_factory=list)


class DocumentEvidencePackage(BaseModel):
    kb_sn: Optional[str]
    asset_name: Optional[str]
    doc_id: Optional[str]
    source: str
    title: Optional[str]
    score: float
    evidence_count: int
    artifacts: PackageArtifacts
    evidence: List[EvidenceItem] = Field(default_factory=list)
    candidate_slices: List[EvidenceItem] = Field(default_factory=list, exclude=True)


class EvidencePackageResponse(BaseModel):
    query: str
    rewrite_query: str
    packages: List[DocumentEvidencePackage]

    def to_dict(self) -> Dict[str, Any]:
        return to_agent_evidence_response(self)


ArtifactUrlResolver = Callable[[str], str]
ArtifactTextResolver = Callable[[str], Optional[str]]
Retriever = Callable[[], Optional[Iterable[Any]]]


def run_parallel_retrievers(
    retrievers: Iterable[Retriever],
    raise_on_all_failed: bool = False,
) -> List[Any]:
    retriever_list = [retriever for retriever in retrievers if retriever is not None]
    if not retriever_list:
        return []
    if len(retriever_list) == 1:
        return _collect_retriever_result(retriever_list[0], raise_on_all_failed)

    results: List[Any] = []
    failure_count = 0
    with ThreadPoolExecutor(max_workers=len(retriever_list)) as pool:
        futures = {pool.submit(retriever): retriever for retriever in retriever_list}
        for future in as_completed(futures):
            retriever = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failure_count += 1
                _log_retriever_failure(retriever, exc)
                continue
            results.extend(_normalize_retriever_result(result))
    if raise_on_all_failed and failure_count == len(retriever_list):
        raise RuntimeError("all retrievers failed")
    return results


def _collect_retriever_result(retriever: Retriever, raise_on_all_failed: bool = False) -> List[Any]:
    try:
        return _normalize_retriever_result(retriever())
    except Exception as exc:
        _log_retriever_failure(retriever, exc)
        if raise_on_all_failed:
            raise RuntimeError("all retrievers failed") from exc
        return []


def _log_retriever_failure(retriever: Retriever, exc: Exception) -> None:
    LOGGER.warning(
        "evidence.retriever_failed source=%s name=%s error=%s",
        getattr(retriever, "source", None),
        getattr(retriever, "name", None),
        exc,
    )


def _normalize_retriever_result(result: Optional[Iterable[Any]]) -> List[Any]:
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return list(result)


def collect_evidence_candidate_documents(
    candidate_batches: Iterable[Tuple[str, Optional[Iterable[Any]]]],
    max_documents: Optional[int] = None,
) -> Tuple[List[Any], bool]:
    candidates_by_text: Dict[str, Any] = {}
    has_ipd_documents = False
    for source, batch_documents in candidate_batches:
        batch_documents = list(batch_documents or [])
        if source == "ipd" and batch_documents:
            has_ipd_documents = True
        _merge_candidate_documents(candidates_by_text, batch_documents)

    documents = sorted(candidates_by_text.values(), key=_document_score, reverse=True)
    if max_documents is not None:
        documents = documents[:max(max_documents, 0)]
    return documents, has_ipd_documents


def _merge_candidate_documents(candidates_by_text: Dict[str, Any], documents: Iterable[Any]) -> None:
    for document in documents:
        key = getattr(document, "text", "") or ""
        current = candidates_by_text.get(key)
        if current is None or _document_score(document) > _document_score(current):
            candidates_by_text[key] = document


def _document_score(document: Any) -> float:
    return float(getattr(document, "score", 0.0) or 0.0)


def expanded_candidate_top_k(
    package_top_k: int,
    max_evidence_per_package: int,
    candidate_multiplier: int,
    max_candidate_k: int = 100,
) -> int:
    if int(package_top_k or 0) <= 0:
        return 1
    package_top_k = max(int(package_top_k or 0), 1)
    max_evidence_per_package = max(int(max_evidence_per_package or 0), 1)
    candidate_multiplier = max(int(candidate_multiplier or 0), 1)
    max_candidate_k = max(int(max_candidate_k or 0), 1)
    return min(package_top_k * max_evidence_per_package * candidate_multiplier, max_candidate_k)


def build_evidence_packages(
    query: str,
    rewrite_query: str,
    documents: Iterable[Any],
    options: Optional[EvidencePackageOptions] = None,
    artifact_url_resolver: Optional[ArtifactUrlResolver] = None,
    artifact_text_resolver: Optional[ArtifactTextResolver] = None,
) -> EvidencePackageResponse:
    options = options or EvidencePackageOptions()
    document_list = list(documents)
    artifact_url_resolver = _prepared_artifact_url_resolver(document_list, options, artifact_url_resolver)
    artifact_text_resolver = _memoized_text_resolver(artifact_text_resolver)
    package_limit = max(options.package_top_k, 0)
    evidence_limit = max(options.max_evidence_per_package, 0)
    groups: Dict[Tuple[Optional[str], Optional[str], str], _PackageAccumulator] = {}

    for document in document_list:
        metadata = _extended_metadata(document)
        source = _document_source(document)
        group_key = _document_group_key(metadata, source)
        if group_key not in groups:
            groups[group_key] = _PackageAccumulator(
                kb_sn=metadata.get("kb_sn"),
                asset_name=metadata.get("asset_name"),
                doc_id=_optional_string(metadata.get("doc_id")),
                source=source,
                title=_document_title(document, metadata),
                document_artifact=_document_artifact(metadata, options, artifact_url_resolver),
            )
        groups[group_key].add(_evidence_item(document, metadata, options, artifact_url_resolver))

    packages = [
        group.to_package(evidence_limit, options, artifact_text_resolver)
        for group in groups.values()
    ]
    packages.sort(key=lambda package: package.score, reverse=True)
    if package_limit:
        packages = packages[:package_limit]
    else:
        packages = []
    LOGGER.debug(
        "evidence.package_build candidate_count=%s package_count=%s",
        len(document_list),
        len(packages),
    )
    return EvidencePackageResponse(query=query, rewrite_query=rewrite_query, packages=packages)


def to_answer_evidence_response(
    response: EvidencePackageResponse,
    artifact_url_builder: Optional[ArtifactUrlResolver] = None,
) -> Dict[str, Any]:
    llm_context = []
    display_sources = []
    context_index = 1

    for source_index, package in enumerate(response.packages, start=1):
        source_id = f"S{source_index}"
        source_items = []
        table_artifacts = {artifact.table_id: artifact for artifact in package.artifacts.tables}
        for item in package.evidence:
            context_id = f"C{context_index}"
            context_index += 1
            content_type = _answer_content_type(item)
            title = _answer_item_title(item, package, table_artifacts)
            llm_context.append(
                {
                    "context_id": context_id,
                    "source_id": source_id,
                    "title": title,
                    "content": item.text,
                    "content_type": content_type,
                    "score": item.score,
                }
            )
            source_items.append(
                _answer_display_item(
                    context_id,
                    item,
                    package,
                    table_artifacts,
                    artifact_url_builder,
                )
            )

        display_sources.append(_answer_display_source(package, source_id, source_items))

    return {
        "query": response.query,
        "rewrite_query": response.rewrite_query,
        "llm_context": llm_context,
        "display_sources": display_sources,
    }


def to_agent_evidence_response(response: EvidencePackageResponse) -> Dict[str, Any]:
    return {
        "query": response.query,
        "rewrite_query": response.rewrite_query,
        "packages": [_agent_package(package) for package in response.packages],
    }


def _answer_display_source(
    package: DocumentEvidencePackage,
    source_id: str,
    source_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "source_id": source_id,
        "title": package.title or package.source,
        "source": package.source,
        "score": package.score,
        "items": source_items,
    }


def _agent_package(package: DocumentEvidencePackage) -> Dict[str, Any]:
    package_dict = package.model_dump()
    package_dict["slices"] = [
        _agent_slice(package, item, rank)
        for rank, item in enumerate(package.candidate_slices, start=1)
    ]
    return package_dict


def _agent_slice(package: DocumentEvidencePackage, item: EvidenceItem, rank: int) -> Dict[str, Any]:
    return {
        "slice_id": _slice_id(package, item, rank),
        "rank": rank,
        "text": item.text,
        "score": item.score,
        "block_id": item.block_id,
        "block_index": item.block_index,
        "block_type": item.block_type,
        "section_id": item.section_id,
        "table_id": item.table_id,
        "row_range": item.row_range,
        "row_count": item.row_count,
        "title": item.title,
        "headers": item.headers,
        "refs": dict(item.refs),
        "artifact_keys": dict(item.artifact_keys),
        "es_index": item.es_index,
        "es_doc_id": item.es_doc_id,
    }


def _slice_id(package: DocumentEvidencePackage, item: EvidenceItem, rank: int) -> str:
    doc_id = package.doc_id or package.source or "document"
    item_id = item.block_id or item.table_id or item.section_id or f"slice_{rank}"
    return f"{doc_id}:{item_id}"


def _answer_content_type(item: EvidenceItem) -> str:
    if item.evidence_type == "table" or item.block_type == "table" or item.table_id:
        return "table"
    return "text"


def _answer_item_title(
    item: EvidenceItem,
    package: DocumentEvidencePackage,
    table_artifacts: Dict[str, TableArtifact],
) -> Optional[str]:
    if item.title:
        return item.title
    if item.table_id and item.table_id in table_artifacts:
        return table_artifacts[item.table_id].title
    return package.title


def _answer_display_item(
    context_id: str,
    item: EvidenceItem,
    package: DocumentEvidencePackage,
    table_artifacts: Dict[str, TableArtifact],
    artifact_url_builder: Optional[ArtifactUrlResolver],
) -> Dict[str, Any]:
    if _answer_content_type(item) == "table":
        return _answer_table_display_item(context_id, item, package, table_artifacts, artifact_url_builder)
    return {
        "type": "text",
        "context_id": context_id,
        "title": _answer_item_title(item, package, table_artifacts),
        "content": item.text,
    }


def _answer_table_display_item(
    context_id: str,
    item: EvidenceItem,
    package: DocumentEvidencePackage,
    table_artifacts: Dict[str, TableArtifact],
    artifact_url_builder: Optional[ArtifactUrlResolver],
) -> Dict[str, Any]:
    artifact = table_artifacts.get(item.table_id or "")
    display_item: Dict[str, Any] = {
        "type": "table",
        "context_id": context_id,
        "title": _answer_item_title(item, package, table_artifacts),
        "row_count": _table_display_row_count(item, artifact),
        "hit_row_ranges": _table_display_hit_row_ranges(item, artifact),
    }
    display_ref = _table_display_ref(item, artifact)
    if display_ref and artifact_url_builder:
        display_item["display"] = {
            "mode": "url",
            "url": artifact_url_builder(display_ref),
        }
    return display_item


def _table_display_ref(item: EvidenceItem, artifact: Optional[TableArtifact]) -> Optional[str]:
    if item.refs.get("display"):
        return item.refs["display"]
    if artifact and artifact.refs.get("display"):
        return artifact.refs["display"]
    return None


def _table_display_row_count(item: EvidenceItem, artifact: Optional[TableArtifact]) -> Optional[int]:
    if item.row_count is not None:
        return item.row_count
    if artifact:
        return artifact.row_count
    return None


def _table_display_hit_row_ranges(
    item: EvidenceItem,
    artifact: Optional[TableArtifact],
) -> List[List[int]]:
    if item.hit_row_ranges:
        return item.hit_row_ranges
    if artifact:
        return artifact.hit_row_ranges
    if item.row_range:
        return [[item.row_range[0], item.row_range[1]]]
    return []


def _prepared_artifact_url_resolver(
    documents: Iterable[Any],
    options: EvidencePackageOptions,
    artifact_url_resolver: Optional[ArtifactUrlResolver],
) -> Optional[ArtifactUrlResolver]:
    if artifact_url_resolver is None:
        return None
    if options.artifact_mode != "signed_url":
        return _memoized_resolver(artifact_url_resolver)

    refs = _artifact_refs_from_documents(documents)
    prefetched_urls = _prefetch_artifact_urls(refs, artifact_url_resolver)
    return _memoized_resolver(artifact_url_resolver, prefetched_urls)


def _prefetch_artifact_urls(
    refs: Iterable[str],
    artifact_url_resolver: ArtifactUrlResolver,
) -> Dict[str, str]:
    retrievers = [_artifact_url_retriever(ref, artifact_url_resolver) for ref in sorted(set(refs))]
    resolved_pairs = run_parallel_retrievers(retrievers)
    return {ref: url for ref, url in resolved_pairs if ref and url}


def _artifact_url_retriever(ref: str, artifact_url_resolver: ArtifactUrlResolver) -> Retriever:
    def retrieve() -> List[Tuple[str, str]]:
        return [(ref, artifact_url_resolver(ref))]

    return retrieve


def _artifact_refs_from_documents(documents: Iterable[Any]) -> List[str]:
    refs = []
    for document in documents:
        refs.extend(_artifact_refs_from_metadata(_extended_metadata(document)))
    return refs


def _artifact_refs_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    refs = []
    for metadata_key in REF_FIELD_MAP:
        refs.extend(_present_refs(metadata.get(metadata_key)))
    for metadata_key in ARTIFACT_METADATA_KEYS:
        raw = metadata.get(metadata_key)
        if isinstance(raw, dict):
            refs.extend(_present_refs(raw.get("document_markdown_key"), raw.get("manifest_key")))
    return refs


def _present_refs(*refs: Any) -> List[str]:
    return [str(ref) for ref in refs if ref]


def _memoized_resolver(
    artifact_url_resolver: Optional[ArtifactUrlResolver],
    initial_cache: Optional[Dict[str, str]] = None,
) -> Optional[ArtifactUrlResolver]:
    if artifact_url_resolver is None:
        return None
    cache: Dict[str, str] = dict(initial_cache or {})

    def resolve(ref: str) -> str:
        if ref not in cache:
            cache[ref] = artifact_url_resolver(ref)
        return cache[ref]

    return resolve


def _memoized_text_resolver(
    artifact_text_resolver: Optional[ArtifactTextResolver],
) -> Optional[ArtifactTextResolver]:
    if artifact_text_resolver is None:
        return None
    cache: Dict[str, Optional[str]] = {}

    def resolve(ref: str) -> Optional[str]:
        if ref not in cache:
            try:
                cache[ref] = artifact_text_resolver(ref)
            except Exception:
                cache[ref] = None
        return cache[ref]

    return resolve


class _PackageAccumulator:
    def __init__(
        self,
        kb_sn: Optional[str],
        asset_name: Optional[str],
        doc_id: Optional[str],
        source: str,
        title: Optional[str],
        document_artifact: Optional[DocumentArtifact],
    ):
        self.kb_sn = kb_sn
        self.asset_name = asset_name
        self.doc_id = doc_id
        self.source = source
        self.title = title
        self.document_artifact = document_artifact
        self._items_by_key: Dict[Tuple[str, ...], EvidenceItem] = {}

    def add(self, item: EvidenceItem) -> None:
        key = _evidence_key(item)
        current = self._items_by_key.get(key)
        if current is None or item.score > current.score:
            self._items_by_key[key] = item

    def to_package(
        self,
        evidence_limit: int,
        options: EvidencePackageOptions,
        artifact_text_resolver: Optional[ArtifactTextResolver],
    ) -> DocumentEvidencePackage:
        all_evidence = sorted(self._items_by_key.values(), key=lambda item: item.score, reverse=True)
        table_artifacts = _table_artifacts(all_evidence, options, artifact_text_resolver)
        section_artifacts = _section_artifacts(all_evidence, self.document_artifact, artifact_text_resolver)
        evidence = _package_evidence(all_evidence, table_artifacts, section_artifacts, evidence_limit)
        score = _package_score(evidence, len(self._items_by_key), self.document_artifact)
        return DocumentEvidencePackage(
            kb_sn=self.kb_sn,
            asset_name=self.asset_name,
            doc_id=self.doc_id,
            source=self.source,
            title=self.title,
            score=score,
            evidence_count=len(self._items_by_key),
            artifacts=PackageArtifacts(document=self.document_artifact, sections=section_artifacts, tables=table_artifacts),
            evidence=evidence,
            candidate_slices=all_evidence,
        )


def _evidence_item(
    document: Any,
    metadata: Dict[str, Any],
    options: EvidencePackageOptions,
    artifact_url_resolver: Optional[ArtifactUrlResolver],
) -> EvidenceItem:
    return EvidenceItem(
        text=getattr(document, "text", "") or "",
        score=float(getattr(document, "score", 0.0) or 0.0),
        block_id=_optional_string(metadata.get("block_id")),
        block_index=_optional_int(metadata.get("block_index")),
        block_type=_optional_string(metadata.get("block_type")),
        table_id=_optional_string(metadata.get("table_id")),
        section_id=_optional_string(metadata.get("section_id")),
        row_range=_row_range(metadata.get("row_range")),
        title=_optional_string(metadata.get("title")),
        headers=list(metadata.get("headers") or []),
        table=_optional_dict(metadata.get("table")),
        refs=_artifact_refs(metadata, options, artifact_url_resolver),
        artifact_keys=_artifact_keys(metadata),
        es_index=_optional_string(getattr(document, "es_index", None)),
        es_doc_id=_optional_string(getattr(document, "es_doc_id", None)),
    )


def _document_artifact(
    metadata: Dict[str, Any],
    options: EvidencePackageOptions,
    artifact_url_resolver: Optional[ArtifactUrlResolver],
) -> Optional[DocumentArtifact]:
    for metadata_key in ARTIFACT_METADATA_KEYS:
        raw = metadata.get(metadata_key)
        if not isinstance(raw, dict) or not raw.get("artifact_prefix"):
            continue
        return DocumentArtifact(
            artifact_type=metadata_key,
            artifact_prefix=raw["artifact_prefix"],
            document_markdown=_resolve_artifact_ref(raw.get("document_markdown_key"), options, artifact_url_resolver),
            manifest=_resolve_artifact_ref(raw.get("manifest_key"), options, artifact_url_resolver),
            table_count=int(raw.get("table_count") or 0),
            block_count=int(raw.get("block_count") or 0),
            raw=dict(raw),
        )
    return None


def _artifact_refs(
    metadata: Dict[str, Any],
    options: EvidencePackageOptions,
    artifact_url_resolver: Optional[ArtifactUrlResolver],
) -> Dict[str, str]:
    refs = {}
    for metadata_key, response_key in REF_FIELD_MAP.items():
        value = _resolve_artifact_ref(metadata.get(metadata_key), options, artifact_url_resolver)
        if value:
            refs[response_key] = value
    return refs


def _artifact_keys(metadata: Dict[str, Any]) -> Dict[str, str]:
    return {
        response_key: str(metadata[metadata_key])
        for metadata_key, response_key in REF_FIELD_MAP.items()
        if metadata.get(metadata_key)
    }


def _resolve_artifact_ref(
    ref: Optional[str],
    options: EvidencePackageOptions,
    artifact_url_resolver: Optional[ArtifactUrlResolver],
) -> Optional[str]:
    if not ref:
        return None
    if options.artifact_mode == "signed_url" and artifact_url_resolver:
        return artifact_url_resolver(ref)
    return ref


def _document_group_key(metadata: Dict[str, Any], source: str) -> Tuple[Optional[str], Optional[str], str]:
    doc_identifier = _optional_string(metadata.get("doc_id")) or source
    return metadata.get("kb_sn"), metadata.get("asset_name"), doc_identifier


def _evidence_key(item: EvidenceItem) -> Tuple[str, ...]:
    if item.table_id and item.block_id:
        return "block", item.block_id
    if item.table_id:
        return "table", item.table_id
    if item.block_id:
        return "block_text", item.block_id, item.text
    return "text", item.text


def _section_artifacts(
    items: List[EvidenceItem],
    document_artifact: Optional[DocumentArtifact],
    artifact_text_resolver: Optional[ArtifactTextResolver],
) -> List[DocumentSectionArtifact]:
    manifest = _document_manifest(document_artifact, artifact_text_resolver)
    artifacts = [
        artifact
        for section_id, section_items in _group_section_items(items).items()
        for artifact in [_section_artifact(section_id, section_items, manifest, artifact_text_resolver)]
        if artifact is not None
    ]
    return sorted(artifacts, key=lambda artifact: (artifact.hit_blocks, artifact.score), reverse=True)


def _document_manifest(
    document_artifact: Optional[DocumentArtifact],
    artifact_text_resolver: Optional[ArtifactTextResolver],
) -> Dict[str, Any]:
    manifest_ref = _document_manifest_ref(document_artifact)
    if not manifest_ref or artifact_text_resolver is None:
        return {}
    manifest_text = artifact_text_resolver(manifest_ref)
    if not manifest_text:
        return {}
    try:
        manifest = json.loads(manifest_text)
    except Exception:
        LOGGER.debug("evidence.manifest_load_failed manifest_ref=%s", manifest_ref)
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _document_manifest_ref(document_artifact: Optional[DocumentArtifact]) -> Optional[str]:
    if document_artifact is None:
        return None
    return _optional_string(document_artifact.raw.get("manifest_key")) or document_artifact.manifest


def _group_section_items(items: List[EvidenceItem]) -> Dict[str, List[EvidenceItem]]:
    grouped: Dict[str, List[EvidenceItem]] = {}
    for item in items:
        if item.table_id:
            continue
        section_id = _item_section_id(item)
        if not section_id:
            continue
        grouped.setdefault(section_id, []).append(item)
    return grouped


def _section_artifact(
    section_id: str,
    items: List[EvidenceItem],
    manifest: Dict[str, Any],
    artifact_text_resolver: Optional[ArtifactTextResolver],
) -> Optional[DocumentSectionArtifact]:
    ordered_items = _ordered_section_items(items)
    text = _merged_section_text(ordered_items)
    if len(ordered_items) < SECTION_MERGE_MIN_BLOCKS or not _within_section_budget(text):
        return None
    manifest_section = _section_manifest(section_id, ordered_items, manifest)
    total_blocks = len(_manifest_block_ids(manifest_section))
    coverage = _section_coverage(ordered_items, total_blocks)
    artifact_keys = _merged_artifact_keys(ordered_items)
    refs = _merged_refs(ordered_items)
    _apply_manifest_section_ref(manifest_section, artifact_keys, refs)
    inline_text = _full_section_text(artifact_keys, manifest_section, coverage, artifact_text_resolver)
    mode = "full_section" if inline_text else "merged_section"
    if inline_text:
        artifact_keys["_inline_text"] = inline_text
    _log_section_trace(section_id, ordered_items, total_blocks, coverage, mode)
    return DocumentSectionArtifact(
        mode=mode,
        section_id=section_id,
        title=_section_title(ordered_items),
        headers=_section_headers(ordered_items),
        hit_blocks=len(ordered_items),
        score=max(item.score for item in ordered_items),
        merged_block_ids=_unique_block_ids(ordered_items),
        refs=refs,
        artifact_keys=artifact_keys,
    )


def _section_manifest(
    section_id: str,
    items: List[EvidenceItem],
    manifest: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    sections = manifest.get("sections")
    if not isinstance(sections, list):
        return None
    exact = _section_manifest_by_id(section_id, items, sections)
    if exact:
        return exact
    return _section_manifest_by_headers(_section_headers(items), sections)


def _section_manifest_by_id(
    section_id: str,
    items: List[EvidenceItem],
    sections: List[Any],
) -> Optional[Dict[str, Any]]:
    section_ids = {section_id}
    section_ids.update(item.section_id for item in items if item.section_id)
    for section in sections:
        if isinstance(section, dict) and str(section.get("section_id")) in section_ids:
            return section
    return None


def _section_manifest_by_headers(headers: List[str], sections: List[Any]) -> Optional[Dict[str, Any]]:
    if not headers:
        return None
    for section in sections:
        if isinstance(section, dict) and _clean_headers(section.get("headers") or []) == headers:
            return section
    return None


def _manifest_block_ids(manifest_section: Optional[Dict[str, Any]]) -> List[str]:
    if not manifest_section:
        return []
    block_ids = manifest_section.get("block_ids")
    if not isinstance(block_ids, list):
        return []
    return [str(block_id) for block_id in block_ids if block_id]


def _section_coverage(items: List[EvidenceItem], total_blocks: int) -> Optional[float]:
    if total_blocks <= 0:
        return None
    return len(_unique_block_ids(items)) / total_blocks


def _apply_manifest_section_ref(
    manifest_section: Optional[Dict[str, Any]],
    artifact_keys: Dict[str, str],
    refs: Dict[str, str],
) -> None:
    section_ref = _manifest_section_ref(manifest_section)
    if not section_ref:
        return
    artifact_keys.setdefault("section", section_ref)
    refs.setdefault("section", section_ref)


def _manifest_section_ref(manifest_section: Optional[Dict[str, Any]]) -> Optional[str]:
    if not manifest_section:
        return None
    return _optional_string(manifest_section.get("section_ref"))


def _full_section_text(
    artifact_keys: Dict[str, str],
    manifest_section: Optional[Dict[str, Any]],
    coverage: Optional[float],
    artifact_text_resolver: Optional[ArtifactTextResolver],
) -> Optional[str]:
    if not _should_use_full_section(artifact_keys, manifest_section, coverage):
        return None
    return _inline_section_text(artifact_keys, artifact_text_resolver)


def _should_use_full_section(
    artifact_keys: Dict[str, str],
    manifest_section: Optional[Dict[str, Any]],
    coverage: Optional[float],
) -> bool:
    if not artifact_keys.get("section"):
        return False
    if manifest_section:
        return coverage is not None and coverage >= SECTION_FULL_EXPAND_RATIO
    return True


def _log_section_trace(
    section_id: str,
    items: List[EvidenceItem],
    total_blocks: int,
    coverage: Optional[float],
    mode: str,
) -> None:
    coverage_text = f"{coverage:.6f}" if coverage is not None else "unknown"
    LOGGER.debug(
        "evidence.section section_id=%s hit_blocks=%s total_blocks=%s coverage=%s mode=%s",
        section_id,
        len(_unique_block_ids(items)) or len(items),
        total_blocks,
        coverage_text,
        mode,
    )


def _item_section_id(item: EvidenceItem) -> Optional[str]:
    if item.section_id:
        return item.section_id
    headers = _clean_headers(item.headers)
    if headers:
        return " / ".join(headers)
    return item.title


def _ordered_section_items(items: List[EvidenceItem]) -> List[EvidenceItem]:
    return sorted(items, key=lambda item: (_block_order(item), item.block_id or "", -item.score))


def _merged_section_text(items: List[EvidenceItem]) -> str:
    return "\n\n".join(item.text for item in items if item.text)


def _within_section_budget(text: str) -> bool:
    return len(text) <= MAX_INLINE_SECTION_CHARS


def _inline_section_text(
    artifact_keys: Dict[str, str],
    artifact_text_resolver: Optional[ArtifactTextResolver],
) -> Optional[str]:
    ref = artifact_keys.get("section")
    if not ref or artifact_text_resolver is None:
        return None
    text = artifact_text_resolver(ref)
    if not text or len(text) > MAX_INLINE_SECTION_CHARS:
        return None
    return text


def _section_title(items: List[EvidenceItem]) -> Optional[str]:
    headers = _section_headers(items)
    if headers:
        return headers[-1]
    for item in items:
        if item.title:
            return item.title
    return None


def _section_headers(items: List[EvidenceItem]) -> List[str]:
    for item in items:
        headers = _clean_headers(item.headers)
        if headers:
            return headers
    return []


def _clean_headers(headers: List[Any]) -> List[str]:
    return [str(header) for header in headers if header]


def _unique_block_ids(items: List[EvidenceItem]) -> List[str]:
    block_ids = []
    seen = set()
    for item in items:
        if not item.block_id or item.block_id in seen:
            continue
        seen.add(item.block_id)
        block_ids.append(item.block_id)
    return block_ids


def _block_order(item: EvidenceItem) -> int:
    if item.block_index is not None:
        return item.block_index
    parsed_order = _trailing_int(item.block_id)
    if parsed_order is not None:
        return parsed_order
    return 1_000_000


def _trailing_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    suffix = value.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _table_artifacts(
    items: List[EvidenceItem],
    options: EvidencePackageOptions,
    artifact_text_resolver: Optional[ArtifactTextResolver],
) -> List[TableArtifact]:
    if not options.enable_table_expansion:
        return []
    artifacts = [
        artifact
        for artifact in (
            _table_artifact(table_id, table_items, options)
            for table_id, table_items in _group_table_items(items).items()
        )
        if artifact is not None
    ]
    artifacts = _resolve_expanded_table_artifacts(artifacts, artifact_text_resolver, options)
    return sorted(artifacts, key=lambda artifact: (artifact.expanded, artifact.hit_ratio, artifact.hit_rows), reverse=True)


def _group_table_items(items: List[EvidenceItem]) -> Dict[str, List[EvidenceItem]]:
    grouped: Dict[str, List[EvidenceItem]] = {}
    for item in items:
        if not item.table_id:
            continue
        grouped.setdefault(item.table_id, []).append(item)
    return grouped


def _table_artifact(
    table_id: str,
    items: List[EvidenceItem],
    options: EvidencePackageOptions,
) -> Optional[TableArtifact]:
    table = _first_table(items)
    row_count = _positive_int_from_value(table.get("row_count") if table else None, 0)
    hit_row_ranges = _hit_table_row_ranges(items, row_count) if row_count > 0 else []
    hit_rows = _count_row_ranges(hit_row_ranges)
    hit_ratio = round(hit_rows / row_count, 6) if row_count > 0 else 0.0
    expanded = hit_ratio >= _table_expand_ratio_threshold(options.table_expand_ratio_threshold)
    mode = _table_artifact_mode(row_count, options.max_full_table_rows, expanded)
    _log_table_trace(table_id, items, row_count, hit_rows, hit_ratio, mode)
    return TableArtifact(
        mode=mode,
        table_id=table_id,
        title=_optional_string(table.get("title") if table else None),
        row_count=row_count,
        hit_rows=hit_rows,
        hit_row_ranges=hit_row_ranges,
        hit_blocks=len(items),
        hit_ratio=hit_ratio,
        score=max(item.score for item in items),
        expanded=expanded,
        table=table,
        refs=_merged_refs(items),
        artifact_keys=_merged_artifact_keys(items),
    )


def _log_table_trace(
    table_id: str,
    items: List[EvidenceItem],
    row_count: int,
    hit_rows: int,
    hit_ratio: float,
    mode: str,
) -> None:
    LOGGER.debug(
        "evidence.table table_id=%s hit_blocks=%s hit_rows=%s row_count=%s hit_ratio=%.6f mode=%s",
        table_id,
        len(items),
        hit_rows,
        row_count,
        hit_ratio,
        mode,
    )


def _first_table(items: List[EvidenceItem]) -> Optional[Dict[str, Any]]:
    for item in items:
        if item.table:
            return dict(item.table)
    return None


def _hit_table_row_ranges(items: List[EvidenceItem], row_count: int) -> List[List[int]]:
    ranges = [_clamped_row_range(item.row_range, row_count) for item in items if item.row_range]
    ranges = [row_range for row_range in ranges if row_range is not None]
    if ranges:
        return _merge_row_ranges(ranges)
    if len(items) == 1 and items[0].block_type == "table":
        return [[0, row_count]]
    return []


def _clamped_row_range(row_range: Tuple[int, int], row_count: int) -> Optional[Tuple[int, int]]:
    start, end = row_range
    start = max(start, 0)
    end = min(end, row_count)
    if end <= start:
        return None
    return start, end


def _merge_row_ranges(ranges: List[Tuple[int, int]]) -> List[List[int]]:
    merged: List[List[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            continue
        merged.append([start, end])
    return merged


def _count_row_ranges(ranges: List[List[int]]) -> int:
    return sum(end - start for start, end in ranges)


def _table_artifact_mode(row_count: int, max_full_table_rows: int, expanded: bool) -> str:
    if not expanded:
        return "matched_table"
    max_rows = max(int(max_full_table_rows or 0), 0)
    if max_rows and row_count > max_rows:
        return "table_reference"
    return "full_table_candidate"


def _resolve_expanded_table_artifacts(
    artifacts: List[TableArtifact],
    artifact_text_resolver: Optional[ArtifactTextResolver],
    options: EvidencePackageOptions,
) -> List[TableArtifact]:
    for artifact in artifacts:
        if not artifact.expanded or artifact.mode != "full_table_candidate":
            continue
        artifact.mode = "full_table_inline" if _inline_table_text(artifact, artifact_text_resolver, options) else "table_reference"
    return artifacts


def _package_evidence(
    all_evidence: List[EvidenceItem],
    table_artifacts: List[TableArtifact],
    section_artifacts: List[DocumentSectionArtifact],
    evidence_limit: int,
) -> List[EvidenceItem]:
    artifacts_by_table_id = {
        artifact.table_id: artifact
        for artifact in table_artifacts
        if artifact.expanded
    }
    promoted_table_ids = set(artifacts_by_table_id)
    table_items = _group_table_items(all_evidence)
    promoted_tables = [
        _table_evidence_item(artifact, table_items.get(artifact.table_id, []))
        for artifact in artifacts_by_table_id.values()
    ]
    section_items = _group_section_items(all_evidence)
    artifacts_by_section_id = {artifact.section_id: artifact for artifact in section_artifacts}
    promoted_sections = [
        _section_evidence_item(artifact, section_items.get(artifact.section_id, []))
        for artifact in artifacts_by_section_id.values()
    ]
    remaining_items = [
        item
        for item in all_evidence
        if item.table_id not in promoted_table_ids and _item_section_id(item) not in artifacts_by_section_id
    ]
    evidence = sorted(promoted_tables + promoted_sections + remaining_items, key=lambda item: item.score, reverse=True)
    if evidence_limit:
        return evidence[:evidence_limit]
    return []


def _section_evidence_item(artifact: DocumentSectionArtifact, items: List[EvidenceItem]) -> EvidenceItem:
    ordered_items = _ordered_section_items(items)
    return EvidenceItem(
        text=_section_evidence_text(artifact, ordered_items),
        score=artifact.score,
        evidence_type="section",
        mode=artifact.mode,
        block_type="text",
        section_id=artifact.section_id,
        hit_blocks=artifact.hit_blocks,
        merged_block_ids=artifact.merged_block_ids,
        title=artifact.title,
        headers=artifact.headers,
        refs=artifact.refs,
        artifact_keys=artifact.artifact_keys,
    )


def _section_evidence_text(artifact: DocumentSectionArtifact, items: List[EvidenceItem]) -> str:
    return artifact.artifact_keys.get("_inline_text") or _merged_section_text(items)


def _table_evidence_item(artifact: TableArtifact, items: List[EvidenceItem]) -> EvidenceItem:
    return EvidenceItem(
        text=_table_evidence_text(artifact, items),
        score=artifact.score,
        evidence_type="table",
        mode=artifact.mode,
        block_type="table",
        table_id=artifact.table_id,
        row_range=(0, artifact.row_count) if artifact.mode == "full_table_inline" else None,
        row_count=artifact.row_count,
        hit_rows=artifact.hit_rows,
        hit_blocks=artifact.hit_blocks,
        hit_ratio=artifact.hit_ratio,
        hit_row_ranges=artifact.hit_row_ranges,
        merged_block_ids=[item.block_id for item in items if item.block_id],
        title=artifact.title,
        table=artifact.table,
        refs=artifact.refs,
        artifact_keys=artifact.artifact_keys,
    )


def _table_evidence_text(artifact: TableArtifact, items: List[EvidenceItem]) -> str:
    text = artifact.artifact_keys.get("_inline_text")
    if artifact.mode == "full_table_inline" and text:
        return text
    return "\n\n".join(item.text for item in items if item.text)


def _inline_table_text(
    artifact: TableArtifact,
    artifact_text_resolver: Optional[ArtifactTextResolver],
    options: EvidencePackageOptions,
) -> Optional[str]:
    ref = artifact.artifact_keys.get("llm_table")
    if not ref or artifact_text_resolver is None:
        return None
    text = artifact_text_resolver(ref)
    if not text:
        return None
    max_chars = max(int(options.max_inline_table_chars or 0), 0)
    if max_chars and len(text) > max_chars:
        return None
    artifact.artifact_keys["_inline_text"] = text
    return text


def _merged_refs(items: List[EvidenceItem]) -> Dict[str, str]:
    refs: Dict[str, str] = {}
    for item in items:
        for key, value in item.refs.items():
            refs.setdefault(key, value)
    return refs


def _merged_artifact_keys(items: List[EvidenceItem]) -> Dict[str, str]:
    refs: Dict[str, str] = {}
    for item in items:
        for key, value in item.artifact_keys.items():
            refs.setdefault(key, value)
    return refs


def _table_expand_ratio_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return 0.7
    return min(max(threshold, 0.0), 1.0)


def _package_score(
    evidence: List[EvidenceItem],
    total_evidence_count: int,
    document_artifact: Optional[DocumentArtifact],
) -> float:
    if not evidence:
        return 0.0
    max_score = max(item.score for item in evidence)
    average_score = sum(item.score for item in evidence) / len(evidence)
    evidence_signal = min(total_evidence_count, 10) / 10
    artifact_signal = 1.0 if document_artifact else 0.0
    return round(max_score * 0.7 + average_score * 0.2 + evidence_signal * 0.07 + artifact_signal * 0.03, 6)


def _extended_metadata(document: Any) -> Dict[str, Any]:
    metadata = getattr(document, "metadata", None)
    extended_metadata = getattr(metadata, "extended_metadata", None)
    return dict(extended_metadata or {})


def _document_source(document: Any) -> str:
    metadata = getattr(document, "metadata", None)
    return getattr(metadata, "source", "") or ""


def _document_title(document: Any, metadata: Dict[str, Any]) -> Optional[str]:
    if metadata.get("title"):
        return str(metadata["title"])
    retrieve_metadata = getattr(getattr(document, "metadata", None), "retrieve_metadata", None)
    title = getattr(retrieve_metadata, "title", None)
    return _optional_string(title)


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_dict(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return dict(value)
    return None


def _row_range(value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    return start, end


def _positive_int_from_value(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
