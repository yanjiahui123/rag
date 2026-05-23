import copy
import json
import os
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional


DATAOPS_MAX_PAYLOAD_BYTES_ENV = "IPD_RAG_DATAOPS_MAX_PAYLOAD_BYTES"
DATAOPS_PAYLOAD_MARGIN_BYTES_ENV = "IPD_RAG_DATAOPS_PAYLOAD_MARGIN_BYTES"
DATAOPS_MAX_SLICE_TEXT_CHARS_ENV = "IPD_RAG_DATAOPS_MAX_SLICE_TEXT_CHARS"
DATAOPS_MAX_SLICE_BYTES_ENV = "IPD_RAG_DATAOPS_MAX_SLICE_BYTES"
DATAOPS_SLICE_MARGIN_BYTES_ENV = "IPD_RAG_DATAOPS_SLICE_MARGIN_BYTES"
DEFAULT_DATAOPS_MAX_PAYLOAD_BYTES = 4_000_000
DEFAULT_DATAOPS_PAYLOAD_MARGIN_BYTES = 16_384
DEFAULT_DATAOPS_MAX_SLICE_TEXT_CHARS = 51_200
DEFAULT_DATAOPS_MAX_SLICE_BYTES = 99_000
DEFAULT_DATAOPS_SLICE_MARGIN_BYTES = 8_192
DEFAULT_DATAOPS_EMPTY_SLICE_TEXT = "No slice content"

DATAOPS_OVERSIZED_METADATA_KEYS = {
    "content",
    "display",
    "general_text",
    "header_contents",
    "html",
    "json",
    "llm_markdown",
    "markdown",
    "page_content",
    "raw_text",
    "table_html",
    "table_json",
    "table_llm_text",
    "text",
}
NATURAL_TEXT_BREAKS = set("\n。！？.!?；;")


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


def configured_dataops_max_slice_text_chars() -> int:
    return _configured_int(
        DATAOPS_MAX_SLICE_TEXT_CHARS_ENV,
        DEFAULT_DATAOPS_MAX_SLICE_TEXT_CHARS,
        minimum=1,
    )


def configured_dataops_max_slice_bytes() -> int:
    slice_limit = _configured_int(
        DATAOPS_MAX_SLICE_BYTES_ENV,
        DEFAULT_DATAOPS_MAX_SLICE_BYTES,
        minimum=1,
    )
    margin = _configured_int(
        DATAOPS_SLICE_MARGIN_BYTES_ENV,
        DEFAULT_DATAOPS_SLICE_MARGIN_BYTES,
        minimum=0,
    )
    return max(1, slice_limit - margin)


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
    payload = _dump_dataops_json(document_entries)
    return len(payload.encode("utf-8"))


def dataops_slice_payload_size(slice_entry: Dict[str, Any]) -> int:
    payload = _dump_dataops_json(slice_entry)
    return len(payload.encode("utf-8"))


def dataops_document_part_name(document_name: str, part_index: int, suffix_first: bool = False) -> str:
    if part_index <= 1 and not suffix_first:
        return document_name
    stem, extension = os.path.splitext(str(document_name or "document"))
    return f"{stem}_{part_index}{extension}"


def normalize_document_entry_for_dataops(
    document_entry: DocumentEntry,
    max_slice_text_chars: Optional[int] = None,
    max_slice_bytes: Optional[int] = None,
) -> DocumentEntry:
    normalized_entry = copy.deepcopy(document_entry)
    slices = normalized_entry.get("slices") or []
    normalized_slices = normalize_slices_for_dataops(
        slices,
        max_slice_text_chars=max_slice_text_chars,
        max_slice_bytes=max_slice_bytes,
    )
    if not normalized_slices:
        normalized_slices = [_fallback_slice(slices)]
    normalized_entry["slices"] = normalized_slices
    return normalized_entry


def normalize_slices_for_dataops(
    slices: Iterable[Dict[str, Any]],
    max_slice_text_chars: Optional[int] = None,
    max_slice_bytes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    text_limit = max_slice_text_chars or configured_dataops_max_slice_text_chars()
    byte_limit = max_slice_bytes or configured_dataops_max_slice_bytes()
    normalized_slices: List[Dict[str, Any]] = []
    for slice_entry in slices:
        normalized_slices.extend(_normalize_slice_for_dataops(slice_entry, text_limit, byte_limit))
    return normalized_slices


def batch_document_entries_for_dataops(
    document_entries: Iterable[DocumentEntry],
    max_payload_bytes: Optional[int] = None,
) -> Iterator[List[DocumentEntry]]:
    limit = max_payload_bytes or configured_dataops_max_payload_bytes()
    current_batch: List[DocumentEntry] = []
    for document_entry in document_entries:
        document_entry = normalize_document_entry_for_dataops(document_entry)
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
    document_entry = normalize_document_entry_for_dataops(document_entry)
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
        safe_batch = json_safe_document_entries(batch)
        if doc_id is None:
            sender(ipd_rag_kb_sn, kb_sn, safe_batch)
        else:
            sender(ipd_rag_kb_sn, kb_sn, safe_batch, doc_id)


def json_safe_document_entries(document_entries: List[DocumentEntry]) -> List[DocumentEntry]:
    return [_json_safe_value(document_entry) for document_entry in document_entries]


def _dump_dataops_json(value: Any) -> str:
    return json.dumps(_json_safe_value(value), ensure_ascii=False, separators=(",", ":"))


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _normalize_slice_for_dataops(
    slice_entry: Dict[str, Any],
    max_slice_text_chars: int,
    max_slice_bytes: int,
) -> List[Dict[str, Any]]:
    compacted_slice = _compact_slice_metadata_for_dataops(slice_entry, max_slice_bytes)
    text = str(compacted_slice.get("text") or "").strip()
    if not text:
        return []
    normalized_slices: List[Dict[str, Any]] = []
    text_parts = _split_text_for_dataops(
        text,
        compacted_slice,
        max_slice_text_chars,
        max_slice_bytes,
    )
    for text_part in text_parts:
        normalized_slices.append(_copy_slice_with_text(compacted_slice, text_part))
    return normalized_slices


def _compact_slice_metadata_for_dataops(slice_entry: Dict[str, Any], max_slice_bytes: int) -> Dict[str, Any]:
    compacted_slice = copy.deepcopy(slice_entry)
    if _slice_text_fits(compacted_slice, "x", max_slice_text_chars=1, max_slice_bytes=max_slice_bytes):
        return compacted_slice

    extended_metadata = _extended_metadata(compacted_slice)
    if isinstance(extended_metadata, dict):
        for key in DATAOPS_OVERSIZED_METADATA_KEYS:
            extended_metadata.pop(key, None)
        _drop_largest_metadata_values_until_slice_fits(compacted_slice, max_slice_bytes)

    if _slice_text_fits(compacted_slice, "x", max_slice_text_chars=1, max_slice_bytes=max_slice_bytes):
        return compacted_slice

    compacted_slice["meta_data"] = {"extended_metadata": {}}
    return compacted_slice


def _extended_metadata(slice_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    meta_data = slice_entry.get("meta_data")
    if not isinstance(meta_data, Mapping):
        return None
    extended_metadata = meta_data.get("extended_metadata")
    if isinstance(extended_metadata, dict):
        return extended_metadata
    if isinstance(extended_metadata, Mapping):
        meta_data["extended_metadata"] = dict(extended_metadata)
        return meta_data["extended_metadata"]
    return None


def _drop_largest_metadata_values_until_slice_fits(slice_entry: Dict[str, Any], max_slice_bytes: int) -> None:
    extended_metadata = _extended_metadata(slice_entry)
    while extended_metadata and not _slice_text_fits(
        slice_entry,
        "x",
        max_slice_text_chars=1,
        max_slice_bytes=max_slice_bytes,
    ):
        largest_key = max(
            extended_metadata,
            key=lambda key: len(_dump_dataops_json(extended_metadata[key]).encode("utf-8")),
        )
        extended_metadata.pop(largest_key, None)


def _split_text_for_dataops(
    text: str,
    slice_template: Dict[str, Any],
    max_slice_text_chars: int,
    max_slice_bytes: int,
) -> List[str]:
    text_parts: List[str] = []
    current_text = ""
    for segment in _natural_text_segments(text):
        candidate_text = current_text + segment
        if _slice_text_fits(
            slice_template,
            candidate_text,
            max_slice_text_chars=max_slice_text_chars,
            max_slice_bytes=max_slice_bytes,
        ):
            current_text = candidate_text
            continue

        if current_text.strip():
            text_parts.append(current_text.strip())
            current_text = ""

        if _slice_text_fits(
            slice_template,
            segment,
            max_slice_text_chars=max_slice_text_chars,
            max_slice_bytes=max_slice_bytes,
        ):
            current_text = segment
        else:
            text_parts.extend(
                _split_oversized_text_segment(
                    segment,
                    slice_template,
                    max_slice_text_chars,
                    max_slice_bytes,
                )
            )

    if current_text.strip():
        text_parts.append(current_text.strip())
    return [text_part for text_part in text_parts if text_part]


def _natural_text_segments(text: str) -> List[str]:
    segments = []
    current_chars = []
    for char in text:
        current_chars.append(char)
        if char in NATURAL_TEXT_BREAKS:
            segment = "".join(current_chars)
            if segment.strip():
                segments.append(segment)
            current_chars = []
    if current_chars:
        segment = "".join(current_chars)
        if segment.strip():
            segments.append(segment)
    return segments or [text]


def _split_oversized_text_segment(
    text: str,
    slice_template: Dict[str, Any],
    max_slice_text_chars: int,
    max_slice_bytes: int,
) -> List[str]:
    text_parts = []
    remaining_text = text.strip()
    while remaining_text:
        split_index = _largest_fitting_prefix_length(
            remaining_text,
            slice_template,
            max_slice_text_chars,
            max_slice_bytes,
        )
        if split_index <= 0:
            raise ValueError("DataOps slice metadata leaves no room for non-empty text")
        text_part = remaining_text[:split_index].strip()
        if text_part:
            text_parts.append(text_part)
        remaining_text = remaining_text[split_index:].lstrip()
    return text_parts


def _largest_fitting_prefix_length(
    text: str,
    slice_template: Dict[str, Any],
    max_slice_text_chars: int,
    max_slice_bytes: int,
) -> int:
    low = 1
    high = min(len(text), max_slice_text_chars)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        candidate_text = text[:middle].strip()
        if _slice_text_fits(
            slice_template,
            candidate_text,
            max_slice_text_chars=max_slice_text_chars,
            max_slice_bytes=max_slice_bytes,
        ):
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def _slice_text_fits(
    slice_template: Dict[str, Any],
    text: str,
    max_slice_text_chars: int,
    max_slice_bytes: int,
) -> bool:
    text = text.strip()
    return (
        bool(text)
        and len(text) <= max_slice_text_chars
        and dataops_slice_payload_size(_copy_slice_with_text(slice_template, text)) <= max_slice_bytes
    )


def _copy_slice_with_text(slice_entry: Dict[str, Any], text: str) -> Dict[str, Any]:
    copied_slice = copy.deepcopy(slice_entry)
    copied_slice["text"] = text
    return copied_slice


def _fallback_slice(slices: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    for slice_entry in slices:
        if isinstance(slice_entry, Mapping) and isinstance(slice_entry.get("meta_data"), Mapping):
            return {
                "text": DEFAULT_DATAOPS_EMPTY_SLICE_TEXT,
                "meta_data": copy.deepcopy(slice_entry["meta_data"]),
            }
    return {"text": DEFAULT_DATAOPS_EMPTY_SLICE_TEXT, "meta_data": {"extended_metadata": {}}}


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
    part_name = dataops_document_part_name(document_entry.get("filename", ""), part_index)
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
