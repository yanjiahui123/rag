# Document Parse/Split/Table Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the vectorization pipeline into explicit parse and split stages while adding a reusable parsed-block protocol for table-capable loaders.

**Architecture:** Add a dependency-light parsed block module that converts loader `Document` objects into `ParsedBlock` objects and later back into final LangChain `Document` chunks. Keep `load_file` as a compatibility wrapper, and expose new Dagster ops for parse, optional parsed-content persistence, and split.

**Tech Stack:** Python dataclasses, LangChain `Document` at integration boundaries, Dagster ops, stdlib `unittest` for local verification because third-party test dependencies are not installed in this workspace.

---

### Task 1: Parsed Block Core

**Files:**
- Create: `rag_service/document_loaders/parsed_blocks.py`
- Create: `tests/test_parsed_blocks.py`

- [ ] **Step 1: Write the failing tests**

```python
import copy
import unittest

from rag_service.document_loaders.parsed_blocks import (
    DISPLAY_METADATA_KEY,
    SPLIT_POLICY_NO_SPLIT,
    SPLIT_POLICY_TEXT,
    ParsedBlock,
    blocks_to_documents,
    documents_to_parsed_blocks,
)


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeSplitter:
    def split_text(self, text):
        return [part.strip() for part in text.split("|") if part.strip()]


class ParsedBlockTests(unittest.TestCase):
    def test_documents_to_parsed_blocks_preserves_content_and_metadata(self):
        source_metadata = {
            "source": "demo.xlsx",
            DISPLAY_METADATA_KEY: {"type": "table", "format": "html", "content": "<table></table>"},
        }
        docs = [FakeDocument("hello", source_metadata)]

        blocks = documents_to_parsed_blocks(docs)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "hello")
        self.assertEqual(blocks[0].block_type, "text")
        self.assertEqual(blocks[0].metadata[DISPLAY_METADATA_KEY]["type"], "table")
        self.assertIsNot(blocks[0].metadata, source_metadata)

    def test_blocks_to_documents_splits_text_blocks_and_keeps_metadata_isolated(self):
        block = ParsedBlock(text="alpha | beta", metadata={"source": "demo.txt", "headers": ["Intro"]})

        docs = blocks_to_documents([block], FakeDocument, text_splitter=FakeSplitter())

        self.assertEqual([doc.page_content for doc in docs], ["alpha", "beta"])
        self.assertEqual(docs[0].metadata["headers"], ["Intro"])
        docs[0].metadata["headers"].append("Changed")
        self.assertEqual(block.metadata["headers"], ["Intro"])

    def test_blocks_to_documents_does_not_split_no_split_blocks(self):
        block = ParsedBlock(
            text="alpha | beta",
            metadata={"source": "demo.txt"},
            split_policy=SPLIT_POLICY_NO_SPLIT,
        )

        docs = blocks_to_documents([block], FakeDocument, text_splitter=FakeSplitter())

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "alpha | beta")
        self.assertEqual(docs[0].metadata["split_policy"], SPLIT_POLICY_NO_SPLIT)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_parsed_blocks -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rag_service.document_loaders.parsed_blocks'`.

- [ ] **Step 3: Implement parsed block core**

```python
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

BLOCK_TYPE_TEXT = "text"
BLOCK_TYPE_TABLE = "table"
DISPLAY_METADATA_KEY = "display"
SPLIT_POLICY_TEXT = "text"
SPLIT_POLICY_TABLE_ROWS = "table_rows"
SPLIT_POLICY_NO_SPLIT = "no_split"


@dataclass
class DisplayPayload:
    type: str
    format: str
    content: Optional[str] = None
    ref: Optional[str] = None

    def to_metadata(self) -> Dict[str, Optional[str]]:
        payload = {"type": self.type, "format": self.format}
        if self.content is not None:
            payload["content"] = self.content
        if self.ref is not None:
            payload["ref"] = self.ref
        return payload


@dataclass
class ParsedBlock:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    block_type: str = BLOCK_TYPE_TEXT
    split_policy: str = SPLIT_POLICY_TEXT

    def __post_init__(self):
        self.metadata = copy.deepcopy(self.metadata)
        self.metadata.setdefault("block_type", self.block_type)
        self.metadata.setdefault("split_policy", self.split_policy)


def documents_to_parsed_blocks(documents: Iterable[Any], split_policy: str = SPLIT_POLICY_TEXT) -> List[ParsedBlock]:
    blocks = []
    for document in documents:
        text = getattr(document, "page_content", "")
        if not text:
            continue
        metadata = copy.deepcopy(getattr(document, "metadata", {}) or {})
        block_type = metadata.get("block_type", BLOCK_TYPE_TEXT)
        block_split_policy = metadata.get("split_policy", split_policy)
        blocks.append(
            ParsedBlock(
                text=text,
                metadata=metadata,
                block_type=block_type,
                split_policy=block_split_policy,
            )
        )
    return blocks


def blocks_to_documents(
    blocks: Iterable[ParsedBlock],
    document_factory: Callable[..., Any],
    text_splitter: Optional[Any] = None,
) -> List[Any]:
    documents = []
    for block in blocks:
        if not block.text:
            continue
        splits = _split_block_text(block, text_splitter)
        for split in splits:
            if not split:
                continue
            documents.append(document_factory(page_content=split, metadata=copy.deepcopy(block.metadata)))
    return documents


def _split_block_text(block: ParsedBlock, text_splitter: Optional[Any]) -> List[str]:
    if block.split_policy == SPLIT_POLICY_NO_SPLIT or text_splitter is None:
        return [block.text]
    return text_splitter.split_text(block.text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_parsed_blocks -v`

Expected: PASS, 3 tests.

### Task 2: Loader Parse/Split Compatibility

**Files:**
- Modify: `rag_service/document_loaders/loader.py`

- [ ] **Step 1: Write a focused import-free test through parsed block helpers**

The direct loader module imports third-party services unavailable in this workspace. Keep behavior in Task 1 covered by tests, then wire the helpers into `loader.py`.

- [ ] **Step 2: Add imports**

```python
from rag_service.document_loaders.parsed_blocks import ParsedBlock, blocks_to_documents, documents_to_parsed_blocks
```

- [ ] **Step 3: Extract metadata normalization**

Move the metadata rewrite loop from `load_file` into:

```python
def normalize_loaded_document(
    doc: Document,
    original_document: OriginalDocument,
    asset_type: AssetType,
) -> Optional[Document]:
    if not doc.page_content:
        return None
    metadata = copy.deepcopy(doc.metadata)
    metadata["doc_id"] = original_document.doc_id
    source = metadata.pop("source", None)
    doc.metadata.clear()
    doc.metadata = original_document.dict()
    doc.metadata["extended_metadata"] = metadata
    if original_document.download_key:
        doc.metadata["extended_metadata"]["download_key"] = original_document.download_key
    if original_document.extended_metadata:
        doc.metadata["extended_metadata"].update(original_document.extended_metadata)
    if original_document.online_url:
        online_url, online_url_type, video_start_time = get_online_url_info(original_document.online_url)
        doc.metadata["extended_metadata"]["online_url"] = online_url if online_url else ""
        doc.metadata["extended_metadata"]["online_url_type"] = online_url_type if online_url_type else ""
        doc.metadata["extended_metadata"]["video_start_time"] = video_start_time if video_start_time else ""
    if asset_type == AssetType.QA_PAIR:
        doc.metadata["source"] = source
    return doc
```

- [ ] **Step 4: Add parse and split functions**

```python
def parse_file(
    original_document: OriginalDocument,
    vectorization_config: VectorizationConfig,
    asset_type: AssetType,
) -> List[ParsedBlock]:
    return _parse_file(original_document, vectorization_config, asset_type)


def split_parsed_file(
    parsed_blocks: List[ParsedBlock],
    original_document: OriginalDocument,
    vectorization_config: VectorizationConfig,
    asset_type: AssetType,
) -> List[Document]:
    splitter = get_splitter(original_document.uri, vectorization_config)
    raw_docs = blocks_to_documents(parsed_blocks, Document, splitter)
    processed_docs = []
    for doc in raw_docs:
        normalized_doc = normalize_loaded_document(doc, original_document, asset_type)
        if normalized_doc:
            processed_docs.append(normalized_doc)
    return processed_docs
```

- [ ] **Step 5: Keep `load_file` as compatibility wrapper**

```python
def load_file(original_document, vectorization_config, asset_type):
    try:
        parsed_blocks = parse_file(original_document, vectorization_config, asset_type)
        return split_parsed_file(parsed_blocks, original_document, vectorization_config, asset_type)
    except FileTooLarge as e:
        ...
```

- [ ] **Step 6: Run syntax check**

Run: `python -m py_compile rag_service/document_loaders/loader.py rag_service/document_loaders/parsed_blocks.py`

Expected: exit code 0.

### Task 3: Dagster Parse/Save/Split Ops

**Files:**
- Modify: `rag_service/dagster/dagster_common_op.py`
- Modify: `rag_service/dagster/assets/init_knowledge_base_asset.py`
- Modify: `rag_service/dagster/sensors/init_knowledge_base_asset_sensor.py`

- [ ] **Step 1: Add loader imports**

```python
from rag_service.document_loaders.loader import load_file, parse_file, split_parsed_file
from rag_service.document_loaders.parsed_blocks import ParsedBlock, documents_to_parsed_blocks
```

- [ ] **Step 2: Add `parse_original_documents` op**

```python
@op(retry_policy=RetryPolicy(max_retries=3), tags={"dagster/priority": 1})
def parse_original_documents(
    context: OpExecutionContext, original_documents: List[OriginalDocument]
) -> List[Tuple[OriginalDocument, List[ParsedBlock]]]:
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(session, knowledge_base_serial_number, knowledge_base_asset_name)
        asset_type = knowledge_base_asset.asset_type
        vectorization_config = VectorizationConfig(**knowledge_base_asset.vectorization_config)

    parsed_documents = []
    if asset_type in AssetType.types_skip_loader_and_split():
        for original_document in original_documents:
            chunks = get_text_slices(Path(original_document.uri))
            parsed_documents.append((original_document, documents_to_parsed_blocks(chunks, split_policy="no_split")))
        return parsed_documents

    for original_document in original_documents:
        parsed_blocks = parse_file(original_document, vectorization_config, asset_type)
        parsed_documents.append((original_document, parsed_blocks))
    return parsed_documents
```

- [ ] **Step 3: Add `save_parsed_documents` op**

```python
@op(retry_policy=RetryPolicy(max_retries=1), tags={"dagster/priority": 1})
def save_parsed_documents(
    context: OpExecutionContext, parsed_documents: List[Tuple[OriginalDocument, List[ParsedBlock]]]
) -> List[Tuple[OriginalDocument, List[ParsedBlock]]]:
    return parsed_documents
```

- [ ] **Step 4: Add `split_parsed_documents` op**

```python
@op(retry_policy=RetryPolicy(max_retries=3), tags={"dagster/priority": 1})
def split_parsed_documents(
    context: OpExecutionContext, parsed_documents: List[Tuple[OriginalDocument, List[ParsedBlock]]]
) -> List[Tuple[OriginalDocument, List[Document]]]:
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(session, knowledge_base_serial_number, knowledge_base_asset_name)
        asset_type = knowledge_base_asset.asset_type
        kb_allow_synchronous_update = knowledge_base_asset.knowledge_base.allow_synchronous_update
        vectorization_config = VectorizationConfig(**knowledge_base_asset.vectorization_config)

    document_chunks = []
    for original_document, parsed_blocks in parsed_documents:
        chunks = split_parsed_file(parsed_blocks, original_document, vectorization_config, asset_type)
        document_chunks.append((original_document, chunks))

    if kb_allow_synchronous_update and document_chunks:
        save_added_document_sources(context, knowledge_base_serial_number, knowledge_base_asset_name, [doc for doc, _ in document_chunks])
    return document_chunks
```

- [ ] **Step 5: Extract synchronous source recording helper**

```python
def save_added_document_sources(context, knowledge_base_serial_number, knowledge_base_asset_name, original_documents):
    add_source_list = [doc.source for doc in original_documents]
    documents_info_dir = get_documents_info_root_dir(knowledge_base_serial_number, knowledge_base_asset_name)
    add_dir = documents_info_dir / ADDED_DOCUMENTS_INFO_DIR
    add_dir.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())
    file_name = "added_documents_info_" + file_id + ".json"
    output_file = add_dir / file_name
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(add_source_list, f, indent=2)
```

- [ ] **Step 6: Update graph asset**

Replace `load_original_documents` with `parse_original_documents -> save_parsed_documents -> split_parsed_documents`.

- [ ] **Step 7: Update sensor config**

Add config entries for the three new ops and keep `load_original_documents` only if still used by other jobs.

- [ ] **Step 8: Run syntax check**

Run: `python -m py_compile rag_service/dagster/dagster_common_op.py rag_service/dagster/assets/init_knowledge_base_asset.py rag_service/dagster/sensors/init_knowledge_base_asset_sensor.py`

Expected: exit code 0.

### Task 4: Verification

**Files:**
- Test: `tests/test_parsed_blocks.py`

- [ ] **Step 1: Run unit tests**

Run: `python -m unittest tests.test_parsed_blocks -v`

Expected: PASS, 3 tests.

- [ ] **Step 2: Run syntax checks**

Run: `python -m py_compile rag_service/document_loaders/parsed_blocks.py rag_service/document_loaders/loader.py rag_service/dagster/dagster_common_op.py rag_service/dagster/assets/init_knowledge_base_asset.py rag_service/dagster/sensors/init_knowledge_base_asset_sensor.py`

Expected: exit code 0.
