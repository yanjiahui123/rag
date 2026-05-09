from __future__ import annotations

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
}


class EvidencePackageOptions(BaseModel):
    package_top_k: int = 5
    max_evidence_per_package: int = 6
    artifact_mode: str = "key"


class ArtifactSummary(BaseModel):
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
    block_id: Optional[str] = None
    block_type: Optional[str] = None
    table_id: Optional[str] = None
    title: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    table: Optional[Dict[str, Any]] = None
    refs: Dict[str, str] = Field(default_factory=dict)
    es_index: Optional[str] = None
    es_doc_id: Optional[str] = None


class DocumentEvidencePackage(BaseModel):
    kb_sn: Optional[str]
    asset_name: Optional[str]
    doc_id: Optional[str]
    source: str
    title: Optional[str]
    score: float
    evidence_count: int
    artifact_summary: Optional[ArtifactSummary]
    evidence: List[EvidenceItem] = Field(default_factory=list)


class EvidencePackageResponse(BaseModel):
    query: str
    rewrite_query: str
    packages: List[DocumentEvidencePackage]

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


ArtifactUrlResolver = Callable[[str], str]
Retriever = Callable[[], Optional[Iterable[Any]]]


def run_parallel_retrievers(retrievers: Iterable[Retriever]) -> List[Any]:
    retriever_list = [retriever for retriever in retrievers if retriever is not None]
    if not retriever_list:
        return []
    if len(retriever_list) == 1:
        return _collect_retriever_result(retriever_list[0])

    results: List[Any] = []
    with ThreadPoolExecutor(max_workers=len(retriever_list)) as pool:
        futures = [pool.submit(retriever) for retriever in retriever_list]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                continue
            results.extend(_normalize_retriever_result(result))
    return results


def _collect_retriever_result(retriever: Retriever) -> List[Any]:
    try:
        return _normalize_retriever_result(retriever())
    except Exception:
        return []


def _normalize_retriever_result(result: Optional[Iterable[Any]]) -> List[Any]:
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return list(result)


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
) -> EvidencePackageResponse:
    options = options or EvidencePackageOptions()
    document_list = list(documents)
    artifact_url_resolver = _prepared_artifact_url_resolver(document_list, options, artifact_url_resolver)
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
                artifact_summary=_artifact_summary(metadata, options, artifact_url_resolver),
            )
        groups[group_key].add(_evidence_item(document, metadata, options, artifact_url_resolver))

    packages = [group.to_package(evidence_limit) for group in groups.values()]
    packages.sort(key=lambda package: package.score, reverse=True)
    if package_limit:
        packages = packages[:package_limit]
    else:
        packages = []
    return EvidencePackageResponse(query=query, rewrite_query=rewrite_query, packages=packages)


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


class _PackageAccumulator:
    def __init__(
        self,
        kb_sn: Optional[str],
        asset_name: Optional[str],
        doc_id: Optional[str],
        source: str,
        title: Optional[str],
        artifact_summary: Optional[ArtifactSummary],
    ):
        self.kb_sn = kb_sn
        self.asset_name = asset_name
        self.doc_id = doc_id
        self.source = source
        self.title = title
        self.artifact_summary = artifact_summary
        self._items_by_key: Dict[Tuple[str, str], EvidenceItem] = {}

    def add(self, item: EvidenceItem) -> None:
        key = _evidence_key(item)
        current = self._items_by_key.get(key)
        if current is None or item.score > current.score:
            self._items_by_key[key] = item

    def to_package(self, evidence_limit: int) -> DocumentEvidencePackage:
        evidence = sorted(self._items_by_key.values(), key=lambda item: item.score, reverse=True)
        if evidence_limit:
            evidence = evidence[:evidence_limit]
        else:
            evidence = []
        score = _package_score(evidence, len(self._items_by_key), self.artifact_summary)
        return DocumentEvidencePackage(
            kb_sn=self.kb_sn,
            asset_name=self.asset_name,
            doc_id=self.doc_id,
            source=self.source,
            title=self.title,
            score=score,
            evidence_count=len(self._items_by_key),
            artifact_summary=self.artifact_summary,
            evidence=evidence,
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
        block_type=_optional_string(metadata.get("block_type")),
        table_id=_optional_string(metadata.get("table_id")),
        title=_optional_string(metadata.get("title")),
        headers=list(metadata.get("headers") or []),
        table=_optional_dict(metadata.get("table")),
        refs=_artifact_refs(metadata, options, artifact_url_resolver),
        es_index=_optional_string(getattr(document, "es_index", None)),
        es_doc_id=_optional_string(getattr(document, "es_doc_id", None)),
    )


def _artifact_summary(
    metadata: Dict[str, Any],
    options: EvidencePackageOptions,
    artifact_url_resolver: Optional[ArtifactUrlResolver],
) -> Optional[ArtifactSummary]:
    for metadata_key in ARTIFACT_METADATA_KEYS:
        raw = metadata.get(metadata_key)
        if not isinstance(raw, dict) or not raw.get("artifact_prefix"):
            continue
        return ArtifactSummary(
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


def _evidence_key(item: EvidenceItem) -> Tuple[str, str]:
    if item.table_id:
        return "table", item.table_id
    if item.block_id:
        return "block", item.block_id
    return "text", item.text


def _package_score(
    evidence: List[EvidenceItem],
    total_evidence_count: int,
    artifact_summary: Optional[ArtifactSummary],
) -> float:
    if not evidence:
        return 0.0
    max_score = max(item.score for item in evidence)
    average_score = sum(item.score for item in evidence) / len(evidence)
    evidence_signal = min(total_evidence_count, 10) / 10
    artifact_signal = 1.0 if artifact_summary else 0.0
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


def _optional_dict(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return dict(value)
    return None
