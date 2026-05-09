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
    enable_table_expansion: bool = True
    table_expand_ratio_threshold: float = 0.7
    max_full_table_rows: int = 500
    max_inline_table_chars: int = 40000


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
    evidence_type: Optional[str] = None
    mode: Optional[str] = None
    block_id: Optional[str] = None
    block_type: Optional[str] = None
    table_id: Optional[str] = None
    row_range: Optional[Tuple[int, int]] = None
    row_count: Optional[int] = None
    hit_rows: Optional[int] = None
    hit_blocks: Optional[int] = None
    hit_ratio: Optional[float] = None
    merged_block_ids: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    table: Optional[Dict[str, Any]] = None
    refs: Dict[str, str] = Field(default_factory=dict)
    artifact_keys: Dict[str, str] = Field(default_factory=dict, exclude=True)
    es_index: Optional[str] = None
    es_doc_id: Optional[str] = None


class TableExpansion(BaseModel):
    type: str = "table"
    mode: str
    table_id: str
    title: Optional[str] = None
    row_count: int = 0
    hit_rows: int = 0
    hit_blocks: int = 0
    hit_ratio: float = 0.0
    score: float = 0.0
    table: Optional[Dict[str, Any]] = None
    refs: Dict[str, str] = Field(default_factory=dict)


class TableContext(BaseModel):
    type: str = "table"
    mode: str
    table_id: str
    title: Optional[str] = None
    row_count: int = 0
    hit_rows: int = 0
    hit_blocks: int = 0
    hit_ratio: float = 0.0
    score: float = 0.0
    expanded: bool = False
    table: Optional[Dict[str, Any]] = None
    refs: Dict[str, str] = Field(default_factory=dict)
    artifact_keys: Dict[str, str] = Field(default_factory=dict, exclude=True)


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
    expansions: List[TableExpansion] = Field(default_factory=list)
    table_contexts: List[TableContext] = Field(default_factory=list)


class EvidencePackageResponse(BaseModel):
    query: str
    rewrite_query: str
    packages: List[DocumentEvidencePackage]

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


ArtifactUrlResolver = Callable[[str], str]
ArtifactTextResolver = Callable[[str], Optional[str]]
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
                artifact_summary=_artifact_summary(metadata, options, artifact_url_resolver),
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

    def to_package(
        self,
        evidence_limit: int,
        options: EvidencePackageOptions,
        artifact_text_resolver: Optional[ArtifactTextResolver],
    ) -> DocumentEvidencePackage:
        all_evidence = sorted(self._items_by_key.values(), key=lambda item: item.score, reverse=True)
        table_contexts = _table_contexts(all_evidence, options, artifact_text_resolver)
        expansions = _table_expansions_from_contexts(table_contexts)
        evidence = _package_evidence(all_evidence, table_contexts, evidence_limit)
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
            expansions=expansions,
            table_contexts=table_contexts,
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
        row_range=_row_range(metadata.get("row_range")),
        title=_optional_string(metadata.get("title")),
        headers=list(metadata.get("headers") or []),
        table=_optional_dict(metadata.get("table")),
        refs=_artifact_refs(metadata, options, artifact_url_resolver),
        artifact_keys=_artifact_keys(metadata),
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


def _evidence_key(item: EvidenceItem) -> Tuple[str, str]:
    if item.block_id:
        return "block", item.block_id
    if item.table_id:
        return "table", item.table_id
    return "text", item.text


def _table_contexts(
    items: List[EvidenceItem],
    options: EvidencePackageOptions,
    artifact_text_resolver: Optional[ArtifactTextResolver],
) -> List[TableContext]:
    if not options.enable_table_expansion:
        return []
    contexts = [
        context
        for context in (
            _table_context(table_id, table_items, options)
            for table_id, table_items in _group_table_items(items).items()
        )
        if context is not None
    ]
    contexts = _resolve_expanded_table_contexts(contexts, artifact_text_resolver, options)
    return sorted(contexts, key=lambda context: (context.expanded, context.hit_ratio, context.hit_rows), reverse=True)


def _table_expansions_from_contexts(contexts: List[TableContext]) -> List[TableExpansion]:
    return [
        TableExpansion(
            mode=context.mode,
            table_id=context.table_id,
            title=context.title,
            row_count=context.row_count,
            hit_rows=context.hit_rows,
            hit_blocks=context.hit_blocks,
            hit_ratio=context.hit_ratio,
            score=context.score,
            table=context.table,
            refs=context.refs,
        )
        for context in contexts
        if context.expanded
    ]


def _group_table_items(items: List[EvidenceItem]) -> Dict[str, List[EvidenceItem]]:
    grouped: Dict[str, List[EvidenceItem]] = {}
    for item in items:
        if not item.table_id or not item.table:
            continue
        grouped.setdefault(item.table_id, []).append(item)
    return grouped


def _table_context(
    table_id: str,
    items: List[EvidenceItem],
    options: EvidencePackageOptions,
) -> Optional[TableContext]:
    table = _first_table(items)
    row_count = _positive_int_from_value(table.get("row_count") if table else None, 0)
    if row_count <= 0:
        return None
    hit_rows = _hit_table_rows(items, row_count)
    if hit_rows <= 0:
        return None
    hit_ratio = round(hit_rows / row_count, 6)
    expanded = hit_ratio >= _table_expand_ratio_threshold(options.table_expand_ratio_threshold)
    return TableContext(
        mode=_table_context_mode(row_count, options.max_full_table_rows, expanded),
        table_id=table_id,
        title=_optional_string(table.get("title") if table else None),
        row_count=row_count,
        hit_rows=hit_rows,
        hit_blocks=len(items),
        hit_ratio=hit_ratio,
        score=max(item.score for item in items),
        expanded=expanded,
        table=table,
        refs=_merged_refs(items),
        artifact_keys=_merged_artifact_keys(items),
    )


def _first_table(items: List[EvidenceItem]) -> Optional[Dict[str, Any]]:
    for item in items:
        if item.table:
            return dict(item.table)
    return None


def _hit_table_rows(items: List[EvidenceItem], row_count: int) -> int:
    ranges = [_clamped_row_range(item.row_range, row_count) for item in items if item.row_range]
    ranges = [row_range for row_range in ranges if row_range is not None]
    if ranges:
        return _count_merged_ranges(ranges)
    if len(items) == 1 and items[0].block_type == "table":
        return row_count
    return 0


def _clamped_row_range(row_range: Tuple[int, int], row_count: int) -> Optional[Tuple[int, int]]:
    start, end = row_range
    start = max(start, 0)
    end = min(end, row_count)
    if end <= start:
        return None
    return start, end


def _count_merged_ranges(ranges: List[Tuple[int, int]]) -> int:
    total = 0
    current_start, current_end = sorted(ranges)[0]
    for start, end in sorted(ranges)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def _table_context_mode(row_count: int, max_full_table_rows: int, expanded: bool) -> str:
    if not expanded:
        return "matched_table"
    max_rows = max(int(max_full_table_rows or 0), 0)
    if max_rows and row_count > max_rows:
        return "table_reference"
    return "full_table_candidate"


def _resolve_expanded_table_contexts(
    contexts: List[TableContext],
    artifact_text_resolver: Optional[ArtifactTextResolver],
    options: EvidencePackageOptions,
) -> List[TableContext]:
    for context in contexts:
        if not context.expanded or context.mode != "full_table_candidate":
            continue
        context.mode = "full_table_inline" if _inline_table_text(context, artifact_text_resolver, options) else "table_reference"
    return contexts


def _package_evidence(
    all_evidence: List[EvidenceItem],
    table_contexts: List[TableContext],
    evidence_limit: int,
) -> List[EvidenceItem]:
    contexts_by_table_id = {
        context.table_id: context
        for context in table_contexts
        if context.expanded
    }
    promoted_table_ids = set(contexts_by_table_id)
    grouped_items = _group_table_items(all_evidence)
    promoted_items = [
        _table_evidence_item(context, grouped_items.get(context.table_id, []))
        for context in contexts_by_table_id.values()
    ]
    remaining_items = [item for item in all_evidence if item.table_id not in promoted_table_ids]
    evidence = sorted(promoted_items + remaining_items, key=lambda item: item.score, reverse=True)
    if evidence_limit:
        return evidence[:evidence_limit]
    return []


def _table_evidence_item(context: TableContext, items: List[EvidenceItem]) -> EvidenceItem:
    return EvidenceItem(
        text=_table_evidence_text(context, items),
        score=context.score,
        evidence_type="table",
        mode=context.mode,
        block_type="table",
        table_id=context.table_id,
        row_range=(0, context.row_count) if context.mode == "full_table_inline" else None,
        row_count=context.row_count,
        hit_rows=context.hit_rows,
        hit_blocks=context.hit_blocks,
        hit_ratio=context.hit_ratio,
        merged_block_ids=[item.block_id for item in items if item.block_id],
        title=context.title,
        table=context.table,
        refs=context.refs,
        artifact_keys=context.artifact_keys,
    )


def _table_evidence_text(context: TableContext, items: List[EvidenceItem]) -> str:
    text = context.artifact_keys.get("_inline_text")
    if context.mode == "full_table_inline" and text:
        return text
    return "\n\n".join(item.text for item in items if item.text)


def _inline_table_text(
    context: TableContext,
    artifact_text_resolver: Optional[ArtifactTextResolver],
    options: EvidencePackageOptions,
) -> Optional[str]:
    ref = context.artifact_keys.get("llm_table")
    if not ref or artifact_text_resolver is None:
        return None
    text = artifact_text_resolver(ref)
    if not text:
        return None
    max_chars = max(int(options.max_inline_table_chars or 0), 0)
    if max_chars and len(text) > max_chars:
        return None
    context.artifact_keys["_inline_text"] = text
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
