import os
import unittest
from collections import UserDict
from unittest.mock import patch

from rag_service.dagster import ipd_rag_payload as payload


def slice_entry(text, extended_metadata=None, meta_data=None):
    if meta_data is None:
        meta_data = {"extended_metadata": extended_metadata or {}}
    return {"text": text, "meta_data": meta_data}


def document_entry(document_id, text_parts, filename="manual.pdf"):
    return {
        "operation": 1,
        "filename": filename,
        "title": filename,
        "id": document_id,
        "slices": [slice_entry(text) for text in text_parts],
    }


class IpdRagPayloadIfBranchTests(unittest.TestCase):
    def test_configured_limits_cover_invalid_values_and_minimum_budget(self):
        with patch.dict(
            os.environ,
            {
                payload.DATAOPS_MAX_PAYLOAD_BYTES_ENV: "",
                payload.DATAOPS_PAYLOAD_MARGIN_BYTES_ENV: "not-an-int",
                payload.DATAOPS_MAX_SLICE_TEXT_CHARS_ENV: "0",
                payload.DATAOPS_MAX_SLICE_BYTES_ENV: "20",
                payload.DATAOPS_SLICE_MARGIN_BYTES_ENV: "3",
            },
            clear=True,
        ):
            self.assertEqual(
                payload.configured_dataops_max_payload_bytes(),
                payload.DEFAULT_DATAOPS_MAX_PAYLOAD_BYTES - payload.DEFAULT_DATAOPS_PAYLOAD_MARGIN_BYTES,
            )
            self.assertEqual(
                payload.configured_dataops_max_slice_text_chars(),
                payload.DEFAULT_DATAOPS_MAX_SLICE_TEXT_CHARS,
            )
            self.assertEqual(payload.configured_dataops_max_slice_bytes(), 17)

        with patch.dict(
            os.environ,
            {
                payload.DATAOPS_MAX_PAYLOAD_BYTES_ENV: "5",
                payload.DATAOPS_PAYLOAD_MARGIN_BYTES_ENV: "9",
            },
            clear=True,
        ):
            self.assertEqual(payload.configured_dataops_max_payload_bytes(), 1)

    def test_document_part_name_covers_original_and_default_stem_paths(self):
        self.assertEqual(payload.dataops_document_part_name("manual.pdf", 1), "manual.pdf")
        self.assertEqual(payload.dataops_document_part_name("", 2), "document_2")

    def test_normalize_document_falls_back_with_preserved_or_empty_metadata(self):
        preserved = payload.normalize_document_entry_for_dataops(
            {"slices": [slice_entry("  ", {"source": "manual"})]},
            max_slice_text_chars=20,
            max_slice_bytes=1_000,
        )
        defaulted = payload.normalize_document_entry_for_dataops(
            {"slices": []},
            max_slice_text_chars=20,
            max_slice_bytes=1_000,
        )

        self.assertEqual(preserved["slices"][0]["text"], payload.DEFAULT_DATAOPS_EMPTY_SLICE_TEXT)
        self.assertEqual(preserved["slices"][0]["meta_data"], {"extended_metadata": {"source": "manual"}})
        self.assertEqual(defaulted["slices"], [slice_entry(payload.DEFAULT_DATAOPS_EMPTY_SLICE_TEXT)])

    def test_batching_covers_empty_input_appending_and_new_batch(self):
        entries = [
            document_entry("doc-1", ["short"]),
            document_entry("doc-2", ["x" * 100]),
            document_entry("doc-3", ["y" * 100]),
        ]
        limit = payload.dataops_entries_payload_size(entries[:2])

        batches = list(payload.batch_document_entries_for_dataops(entries, max_payload_bytes=limit))

        self.assertEqual([len(batch) for batch in batches], [2, 1])
        self.assertEqual(list(payload.batch_document_entries_for_dataops([], max_payload_bytes=limit)), [])

    def test_split_document_covers_already_safe_unsplittable_and_part_paths(self):
        safe_entry = document_entry("safe", ["short"])
        safe_result = payload.split_document_entry_for_dataops(
            safe_entry,
            document_id_factory=lambda: "unused",
            max_payload_bytes=10_000,
        )
        unsplittable = payload.split_document_entry_for_dataops(
            document_entry("single", ["x" * 30]),
            document_id_factory=lambda: "unused",
            max_payload_bytes=1,
        )

        split_entry = document_entry("root", ["a" * 30, "b" * 30])
        one_slice_limit = payload.dataops_entries_payload_size([document_entry("root", ["a" * 30])])
        split_result = payload.split_document_entry_for_dataops(
            split_entry,
            document_id_factory=lambda: "part-2",
            max_payload_bytes=one_slice_limit,
        )

        self.assertEqual(safe_result[0]["id"], "safe")
        self.assertEqual(unsplittable[0]["id"], "single")
        self.assertEqual([entry["id"] for entry in split_result], ["root", "part-2"])
        self.assertEqual([entry["filename"] for entry in split_result], ["manual.pdf", "manual_2.pdf"])

    def test_send_entries_covers_both_sender_signatures_and_json_safe_values(self):
        entry = document_entry("doc", ["text"])
        metadata = entry["slices"][0]["meta_data"]["extended_metadata"]
        metadata["tuple"] = ("a", "b")
        metadata["unknown"] = object()
        calls = []

        payload.send_document_entries_to_dataops(lambda *args: calls.append(args), "ipd", "kb", [entry])
        payload.send_document_entries_to_dataops(
            lambda *args: calls.append(args),
            "ipd",
            "kb",
            [entry],
            doc_id="source-doc",
        )

        sent_metadata = calls[0][2][0]["slices"][0]["meta_data"]["extended_metadata"]
        self.assertEqual(len(calls[0]), 3)
        self.assertEqual(len(calls[1]), 4)
        self.assertEqual(sent_metadata["tuple"], ["a", "b"])
        self.assertIsInstance(sent_metadata["unknown"], str)

    def test_compacts_oversized_metadata_by_dropping_known_and_largest_values(self):
        compact_limit = payload.dataops_slice_payload_size(slice_entry("x", {"small": "ok"}))
        oversized = slice_entry(
            "body",
            {
                "content": "known" * 100,
                "large_custom": "custom" * 100,
                "small": "ok",
            },
        )

        compacted = payload._compact_slice_metadata_for_dataops(oversized, compact_limit)

        self.assertEqual(compacted["meta_data"]["extended_metadata"], {"small": "ok"})

    def test_compacts_unusable_metadata_to_empty_metadata_and_converts_mappings(self):
        converted_entry = slice_entry("text", meta_data={"extended_metadata": UserDict({"source": "manual"})})

        converted = payload._extended_metadata(converted_entry)
        absent = payload._extended_metadata(slice_entry("text", meta_data={"extended_metadata": None}))
        compacted = payload._compact_slice_metadata_for_dataops(
            slice_entry("body", meta_data="not-a-mapping"),
            max_slice_bytes=1,
        )

        self.assertEqual(converted, {"source": "manual"})
        self.assertIsInstance(converted_entry["meta_data"]["extended_metadata"], dict)
        self.assertIsNone(absent)
        self.assertEqual(compacted["meta_data"], {"extended_metadata": {}})

    def test_split_text_covers_natural_breaks_and_oversized_segments(self):
        natural_parts = payload.normalize_slices_for_dataops(
            [slice_entry("aa\nbb\ncc")],
            max_slice_text_chars=3,
            max_slice_bytes=1_000,
        )
        oversized_parts = payload.normalize_slices_for_dataops(
            [slice_entry("abcdef")],
            max_slice_text_chars=2,
            max_slice_bytes=1_000,
        )

        self.assertEqual([part["text"] for part in natural_parts], ["aa", "bb", "cc"])
        self.assertEqual([part["text"] for part in oversized_parts], ["ab", "cd", "ef"])
        self.assertEqual(payload._natural_text_segments(" \n"), [" \n"])
        self.assertFalse(payload._slice_text_fits(slice_entry(""), " ", 1, 1_000))

    def test_oversized_text_raises_when_metadata_leaves_no_text_room(self):
        with self.assertRaisesRegex(ValueError, "leaves no room"):
            payload._split_oversized_text_segment(
                "x",
                slice_entry(""),
                max_slice_text_chars=1,
                max_slice_bytes=1,
            )


if __name__ == "__main__":
    unittest.main()
