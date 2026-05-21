from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from pydantic import BaseModel

from rag_service.document_loaders.structured_artifacts import (
    extract_parsed_markdown_artifact_prefix,
    extract_structured_docx_artifact_prefix,
    extract_structured_excel_artifact_prefix,
    extract_structured_html_artifact_prefix,
    extract_structured_image_object_keys,
    extract_structured_markdown_artifact_prefix,
    is_safe_structured_artifact_prefix,
)

DEFAULT_RESOURCE_DELETE_MAX_WORKERS = 8

DeleteObject = Callable[[str], None]
DeleteDir = Callable[[str], None]


class OriginalDocumentResource(Protocol):
    download_key: Optional[str]
    extended_metadata: Optional[Dict[str, Any]]


class VectorStoreResource(Protocol):
    original_documents: Iterable[OriginalDocumentResource]


class VectorStoreResourceDeletionPlan(BaseModel):
    download_keys: Tuple[str, ...] = ()
    artifact_prefixes: Tuple[str, ...] = ()
    image_object_keys: Tuple[str, ...] = ()
    unsafe_artifact_prefixes: Tuple[str, ...] = ()


def build_vector_store_resource_deletion_plan(
    vector_store: VectorStoreResource,
    delete_download_key: bool = True,
    covered_artifact_prefixes: Sequence[str] = (),
) -> VectorStoreResourceDeletionPlan:
    download_keys: List[str] = []
    artifact_prefixes: List[str] = []
    unsafe_artifact_prefixes: List[str] = []
    image_object_keys: List[str] = []
    seen_download_keys = set()
    seen_artifact_prefixes = set()
    seen_unsafe_artifact_prefixes = set()
    seen_image_object_keys = set()

    for original_document in vector_store.original_documents:
        download_key = original_document.download_key
        if (
            delete_download_key
            and download_key
            and not _is_covered_by_prefix(str(download_key), covered_artifact_prefixes)
        ):
            _append_unique(
                download_keys,
                seen_download_keys,
                download_key,
            )

        extended_metadata = original_document.extended_metadata
        for artifact_prefix in _artifact_prefixes_to_delete(extended_metadata):
            if not artifact_prefix:
                continue
            if is_safe_structured_artifact_prefix(artifact_prefix):
                if _is_covered_by_prefix(artifact_prefix, covered_artifact_prefixes):
                    continue
                _append_unique(artifact_prefixes, seen_artifact_prefixes, artifact_prefix)
            else:
                _append_unique(unsafe_artifact_prefixes, seen_unsafe_artifact_prefixes, artifact_prefix)

        for image_object_key in extract_structured_image_object_keys(extended_metadata):
            _append_unique(image_object_keys, seen_image_object_keys, image_object_key)

    filtered_image_object_keys = tuple(
        image_object_key
        for image_object_key in image_object_keys
        if not _is_covered_by_prefix(image_object_key, tuple(covered_artifact_prefixes) + tuple(artifact_prefixes))
    )

    return VectorStoreResourceDeletionPlan(
        download_keys=tuple(download_keys),
        artifact_prefixes=tuple(artifact_prefixes),
        image_object_keys=filtered_image_object_keys,
        unsafe_artifact_prefixes=tuple(unsafe_artifact_prefixes),
    )


def delete_vector_store_resources(
    vector_store: VectorStoreResource,
    delete_download_key: bool = True,
    *,
    delete_object_func: DeleteObject,
    delete_dir_func: DeleteDir,
    logger: Any,
    max_workers: int = DEFAULT_RESOURCE_DELETE_MAX_WORKERS,
    covered_artifact_prefixes: Sequence[str] = (),
) -> VectorStoreResourceDeletionPlan:
    plan = build_vector_store_resource_deletion_plan(
        vector_store,
        delete_download_key,
        covered_artifact_prefixes=covered_artifact_prefixes,
    )

    for artifact_prefix in plan.unsafe_artifact_prefixes:
        logger.warning("Skip unsafe structured artifact prefix deletion: %s", artifact_prefix)

    _run_delete_tasks(delete_object_func, plan.download_keys, max_workers)
    _run_delete_tasks(delete_dir_func, plan.artifact_prefixes, max_workers)
    _run_delete_tasks(delete_object_func, plan.image_object_keys, max_workers)

    return plan


def _artifact_prefixes_to_delete(extended_metadata: Optional[Dict[str, Any]]) -> List[Optional[str]]:
    return [
        extract_structured_docx_artifact_prefix(extended_metadata),
        extract_structured_excel_artifact_prefix(extended_metadata),
        extract_structured_html_artifact_prefix(extended_metadata),
        extract_structured_markdown_artifact_prefix(extended_metadata),
        extract_parsed_markdown_artifact_prefix(extended_metadata),
    ]


def _append_unique(values: List[str], seen: set, value: Optional[str]) -> None:
    if not value:
        return
    value = str(value)
    if value in seen:
        return
    seen.add(value)
    values.append(value)


def _is_covered_by_prefix(object_key: str, prefixes: Iterable[str]) -> bool:
    return any(object_key.startswith(prefix) for prefix in prefixes)


def _run_delete_tasks(delete_func: Callable[[str], None], keys: Sequence[str], max_workers: int) -> None:
    if not keys:
        return
    if len(keys) == 1 or max_workers <= 1:
        for key in keys:
            delete_func(key)
        return

    worker_count = min(max_workers, len(keys))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(delete_func, key) for key in keys]
        for future in as_completed(futures):
            future.result()
