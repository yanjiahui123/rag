import copy
import json
import os
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional


DATAOPS_MAX_PAYLOAD_BYTES_ENV = "IPD_RAG_DATAOPS_MAX_PAYLOAD_BYTES"
DATAOPS_PAYLOAD_MARGIN_BYTES_ENV = "IPD_RAG_DATAOPS_PAYLOAD_MARGIN_BYTES"
DEFAULT_DATAOPS_MAX_PAYLOAD_BYTES = 4_000_000
DEFAULT_DATAOPS_PAYLOAD_MARGIN_BYTES = 16_384


DocumentEntry = Dict[str, Any]


def configured_dataops_max_payload_bytes() -> int:
    transport_limit = _configured_int(
        DATAOPS_MAX_PAYLOAD_BYTES_ENV,
        DEFAULT_DATAOPS_MAX_PAYLOAD_BYTES,
        minimum=1,
    )
    margin = _configured_int(
        DATAOPS_PAYLOAD_MARGIN_BYTES_ENV,
        DEFAULT_DATAOPS_PAYLOAD_MARGIN_BYTES,
        minimum=0,
    )
    return max(1, transport_limit - margin)


def _configured_int(env_name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(env_name)
    if raw_value is None or raw_value == "":
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value >= minimum else default


def dataops_entries_payload_size(document_entries: List[DocumentEntry]) -> int:
    payload = json.dumps(document_entries, ensure_ascii=False, separators=(",", ":"))
    return len(payload.encode("utf-8"))


def batch_document_entries_for_dataops(
    document_entries: Iterable[DocumentEntry],
    max_payload_bytes: Optional[int] = None,
) -> Iterator[List[DocumentEntry]]:
    limit = max_payload_bytes or configured_dataops_max_payload_bytes()
    current_batch: List[DocumentEntry] = []
    for document_entry in document_entries:
        if not current_batch:
            current_batch = [document_entry]
            continue

        candidate_batch = current_batch + [document_entry]
        if dataops_entries_payload_size(candidate_batch) <= limit:
            current_batch.append(document_entry)
            continue

        yield current_batch
        current_batch = [document_entry]

    if current_batch:
        yield current_batch


def split_document_entry_for_dataops(
    document_entry: DocumentEntry,
    document_id_factory: Callable[[], str],
    max_payload_bytes: Optional[int] = None,
) -> List[DocumentEntry]:
    limit = max_payload_bytes or configured_dataops_max_payload_bytes()
    if dataops_entries_payload_size([document_entry]) <= limit:
        return [copy.deepcopy(document_entry)]

    slices = document_entry.get("slices") or []
    if len(slices) <= 1:
        return [copy.deepcopy(document_entry)]

    slice_groups = _split_slices_by_entry_payload(document_entry, slices, limit)
    return [
        _build_document_entry_part(document_entry, slice_group, part_index, document_id_factory)
        for part_index, slice_group in enumerate(slice_groups, start=1)
    ]


def send_document_entries_to_dataops(
    sender: Callable[..., Any],
    ipd_rag_kb_sn: str,
    kb_sn: str,
    document_entries: Iterable[DocumentEntry],
    doc_id: Optional[str] = None,
    max_payload_bytes: Optional[int] = None,
) -> None:
    for batch in batch_document_entries_for_dataops(document_entries, max_payload_bytes=max_payload_bytes):
        if doc_id is None:
            sender(ipd_rag_kb_sn, kb_sn, batch)
        else:
            sender(ipd_rag_kb_sn, kb_sn, batch, doc_id)


def _split_slices_by_entry_payload(
    document_entry: DocumentEntry,
    slices: List[Dict[str, Any]],
    limit: int,
) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    current_group: List[Dict[str, Any]] = []

    for slice_entry in slices:
        candidate_group = current_group + [slice_entry]
        candidate_entry = _copy_document_entry_with_slices(document_entry, candidate_group)
        if current_group and dataops_entries_payload_size([candidate_entry]) > limit:
            groups.append(current_group)
            current_group = [slice_entry]
            continue
        current_group = candidate_group

    if current_group:
        groups.append(current_group)
    return groups


def _build_document_entry_part(
    document_entry: DocumentEntry,
    slices: List[Dict[str, Any]],
    part_index: int,
    document_id_factory: Callable[[], str],
) -> DocumentEntry:
    part = _copy_document_entry_with_slices(document_entry, slices)
    if part_index == 1:
        return part

    part_id = document_id_factory()
    part_name = f"{document_entry.get('filename', '')}_{part_index}"
    part["id"] = part_id
    part["filename"] = part_name
    part["title"] = part_name
    return part


def _copy_document_entry_with_slices(
    document_entry: DocumentEntry,
    slices: List[Dict[str, Any]],
) -> DocumentEntry:
    copied_entry = copy.deepcopy(document_entry)
    copied_entry["slices"] = copy.deepcopy(slices)
    return copied_entry
