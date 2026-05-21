import itertools
import json
import pickle
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Type, Union

from dagster import DynamicOut, DynamicOutput, In, Nothing, OpExecutionContext, RetryPolicy, op
from langchain.schema import Document
from more_itertools import chunked
from sqlmodel import Session, select

from rag_service.config import (
    EMBEDDING_CHUNK_SIZE,
    LAYER_KB_ASSET_MAX_CONTAIN_DOCUMENT_NUMBER,
    PRIVATE_KB_ASSET_MAX_CONTAIN_DOCUMENT_NUMBER,
    VECTORIZATION_CHUNK_SIZE,
)
from rag_service.constants import (
    ADDED_DOCUMENTS_INFO_DIR,
    MAXIMUM_NUMBER_OF_SLICES, DOC_IMPORT_FAIL_NUM_LIMIT, DOC_IMPORT_FAIL_EMBEDDING, DATAOPS_OPERATION_INSERT,
    DELETED_DOCUMENTS_INFO_DIR,
)
from rag_service.corpus_detections.consistency_detector import ConsistencyDetector
from rag_service.corpus_detections.document_structure_detector import DocumentStructureDetector
from rag_service.corpus_detections.logic_detector import LogicDetector
from rag_service.corpus_detections.punction_detector import PunctuationRatioDetector
from rag_service.corpus_detections.readability_detector import ReadabilityDetector
from rag_service.corpus_detections.semantic_coherence_detector import SemanticCoherenceDetector
from rag_service.corpus_detections.semantic_integrity_detector import SemanticIntegrityDetector
from rag_service.corpus_detections.sensitive_word_detector import SensitiveWordDetector
from rag_service.corpus_detections.similar_doument_detector import SimilarDocumentDetector
from rag_service.corpus_detections.url_validation_detector import UrlValidityDetector
from rag_service.database import engine
from rag_service.dagster.ipd_rag_payload import (
    send_document_entries_to_dataops,
    split_document_entry_for_dataops,
)
from rag_service.document_loaders.loader import load_file, parse_file, split_parsed_file
from rag_service.document_loaders.parsed_blocks import (
    ParsedDocument,
    SPLIT_POLICY_NO_SPLIT,
    blocks_to_parsed_document,
    documents_to_parsed_blocks,
)
from rag_service.document_loaders.structured_artifacts import (
    extract_parsed_markdown_artifact_prefix,
    extract_structured_docx_artifact_prefix,
    extract_structured_excel_artifact_prefix,
    extract_structured_html_artifact_prefix,
    extract_structured_image_object_keys,
    extract_structured_markdown_artifact_prefix,
    is_safe_structured_artifact_prefix,
    merge_structured_metadata,
    merge_structured_docx_metadata,
    persist_parsed_markdown_artifact,
    persist_structured_excel_artifacts,
    persist_structured_html_artifacts,
    persist_structured_docx_artifacts,
    persist_structured_markdown_artifacts,
    PARSED_MARKDOWN_METADATA_KEY,
    STRUCTURED_DOCX_METADATA_KEY,
    STRUCTURED_EXCEL_METADATA_KEY,
    STRUCTURED_HTML_METADATA_KEY,
    STRUCTURED_MARKDOWN_METADATA_KEY,
)
from rag_service.logger import Module, get_logger
from rag_service.models.database.models import AutoJobInstances, ServiceConfig, UpdatedOriginalDocument, VectorStore
from rag_service.models.database.models import OriginalDocument as OriginalDocumentEntity
from rag_service.models.enums import (
    AssetType,
    CorpusAccessControlPolicy,
    DocumentLoadStatus,
    JobStatus,
    KnowledgeBaseType,
    StopWordOrSynonymType,
    UpdateOriginalDocumentType,
    VectorizationJobType,
    ServiceConfigType,
)
from rag_service.models.generic.models import OriginalDocument, VectorizationConfig
from rag_service.original_document_fetchers import Fetcher, select_fetcher
from rag_service.rag_app.dao.auto_job_instances_dao import get_vectorization_job_info
from rag_service.rag_app.dao.document_dao import query_document_by_ids
from rag_service.rag_app.dao.request_document_hit_dao import refresh_kba_id_count_and_max_create_to_redis
from rag_service.rag_app.dao.service_config_dao import query_by_config_type_and_name
from rag_service.rag_app.service.knowledge_base_asset_service import get_original_document_display_name
from rag_service.rag_app.service.knowledge_base_service import (
    get_knowledge_base_synonyms_or_stopwords,
)
from rag_service.utils.dagster_util import (
    get_detection_info_root_dir,
    get_documents_info_root_dir,
    get_knowledge_base_asset_root_dir,
    parse_asset_partition_key,
)
from rag_service.utils.db_util import (
    change_vectorization_job_status,
    get_kba_document_index_collection_by_source_list,
    get_knowledge_base_asset,
    get_knowledge_base_by_kb_sn,
    get_original_documents_by_source_list,
    validate_knowledge_base, update_doc_error_info, get_original_documents_by_kb_sn_and_asset_name,
)
from rag_service.utils.fetch_util import get_text_slices
from rag_service.utils.git_fetch_util import handle_remove_readonly
from rag_service.utils.his_util.obs_util import delete_dir, delete_object, get_obs_dir, upload_file_as_bytes
from rag_service.utils.ipd_rag_util import send_data_to_dataops
from rag_service.utils.serdes import deserialize
from rag_service.utils.time_util import now_with_time_zone
from rag_service.vectorize.embedding import embedding
from rag_service.vectorstore import get_vector_store_manager
from rag_service.vectorstore.elasticsearch.es_model import EsQueryResult

logger = get_logger(module=Module.VECTORIZATION)

CORPUS_QUALITY_DETECTION_ENABLED_CONFIG = "enable_corpus_quality_detection"


def is_corpus_quality_detection_enabled(session: Session) -> bool:
    config_value = session.exec(
        select(ServiceConfig.value).where(ServiceConfig.name == CORPUS_QUALITY_DETECTION_ENABLED_CONFIG)
    ).one_or_none()
    if config_value is None:
        return False
    return str(config_value).strip() == "1"


def should_skip_corpus_quality_detection(context: OpExecutionContext) -> bool:
    with Session(engine) as session:
        enabled = is_corpus_quality_detection_enabled(session)
    if enabled:
        return False
    logger.info("语料质量检测开关未开启，跳过检测任务")
    return True


@op(retry_policy=RetryPolicy(max_retries=3))
def change_vectorization_job_status_to_started(context: OpExecutionContext):
    with Session(engine) as session:
        vectorization_job = session.exec(
            select(AutoJobInstances).where(AutoJobInstances.id == uuid.UUID(context.op_config["job_id"]))
        ).one()
        change_vectorization_job_status(session, vectorization_job, JobStatus.STARTED)


def _save_delete_document_to_file(session: Session, knowledge_base_serial_number: str, knowledge_base_asset_name: str):
    original_documents = get_original_documents_by_kb_sn_and_asset_name(
        session, knowledge_base_serial_number, knowledge_base_asset_name
    )
    combined_dict = {}
    for original_document in original_documents:
        if not original_document.ipd_rag_document_list:
            continue
        source = original_document.source
        combined_dict[source] = original_document.ipd_rag_document_list
    documents_info_dir = get_documents_info_root_dir(knowledge_base_serial_number, knowledge_base_asset_name)
    delete_doc_dir = documents_info_dir / DELETED_DOCUMENTS_INFO_DIR
    delete_doc_dir.mkdir(parents=True, exist_ok=True)
    file_name = "delete_document_info_" + str(uuid.uuid4()) + ".json"
    output_file = delete_doc_dir / file_name
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_dict, f, indent=4, ensure_ascii=False)


@op(ins={"no_input": In(Nothing)})
def delete_knowledge_base_asset_vector_store(context: OpExecutionContext):
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )

        vector_stores = [vector_store.name for vector_store in knowledge_base_asset.vector_stores]
        get_vector_store_manager().delete_vector_stores(vector_stores)

        # 如果是重导入，需要同步到ipd。先将要删除的文档记录下来
        if (knowledge_base_asset.knowledge_base.allow_synchronous_update
                and context.dagster_run.job_name == "reload_knowledge_base_asset_job"):
            _save_delete_document_to_file(session, knowledge_base_serial_number, knowledge_base_asset_name)


@op(ins={"no_input": In(Nothing)})
def delete_knowledge_base_asset_resources(context: OpExecutionContext):
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
        for vector_store in knowledge_base_asset.vector_stores:
            delete_vector_store_resources(vector_store, delete_download_key=knowledge_base_asset.save_file)


def delete_vector_store_resources(vector_store: VectorStore, delete_download_key: bool = True):
    for original_document in vector_store.original_documents:
        if delete_download_key and original_document.download_key:
            delete_object(original_document.download_key)
        for artifact_prefix in _artifact_prefixes_to_delete(original_document.extended_metadata):
            _delete_structured_artifact_prefix(artifact_prefix)
        _delete_structured_image_objects(original_document.extended_metadata)


def _artifact_prefixes_to_delete(extended_metadata: Optional[Dict[str, Any]]) -> List[Optional[str]]:
    return [
        extract_structured_docx_artifact_prefix(extended_metadata),
        extract_structured_excel_artifact_prefix(extended_metadata),
        extract_structured_html_artifact_prefix(extended_metadata),
        extract_structured_markdown_artifact_prefix(extended_metadata),
        extract_parsed_markdown_artifact_prefix(extended_metadata),
    ]


def _delete_structured_artifact_prefix(artifact_prefix: Optional[str]) -> None:
    if not artifact_prefix:
        return
    if is_safe_structured_artifact_prefix(artifact_prefix):
        delete_dir(artifact_prefix)
        return
    logger.warning("Skip unsafe structured artifact prefix deletion: %s", artifact_prefix)


def _delete_structured_image_objects(extended_metadata: Optional[Dict[str, Any]]) -> None:
    for image_object_key in extract_structured_image_object_keys(extended_metadata):
        delete_object(image_object_key)


def document_deduplication(original_documents):
    """
    对同一个资产下的文档根据uri去重
    :param original_documents: 文档列表
    :return: 去重后的文档列表
    """
    seen = set()
    unique_docs = []
    for doc in original_documents:
        if doc.uri not in seen:
            seen.add(doc.uri)
            unique_docs.append(doc)
    return unique_docs


@op(ins={"no_input": In(Nothing)}, retry_policy=RetryPolicy(max_retries=3))
def fetch_raw_document(context: OpExecutionContext):
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    knowledge_base_asset_dir = get_knowledge_base_asset_root_dir(
        knowledge_base_serial_number, knowledge_base_asset_name
    )
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
        vectorization_config = VectorizationConfig(**knowledge_base_asset.vectorization_config)

    fetcher: Type[Fetcher] = select_fetcher(knowledge_base_asset.asset_type)
    original_documents = fetcher(
        asset_uri=knowledge_base_asset.asset_uri,
        asset_root_dir=knowledge_base_asset_dir,
        asset_type=knowledge_base_asset.asset_type,
        save_file=knowledge_base_asset.save_file,
        local_file_online_dic=vectorization_config.local_file_online_dic,
        attachment_vectorize=vectorization_config.attachment_vectorize,
    ).fetch(knowledge_base_serial_number, knowledge_base_asset_name)
    original_documents = document_deduplication(original_documents)
    return original_documents


def get_max_documents_num(max_documents_num, asset, session):
    default_num = (
        PRIVATE_KB_ASSET_MAX_CONTAIN_DOCUMENT_NUMBER
        if asset.knowledge_base.kb_type == KnowledgeBaseType.PRIVATE
        else LAYER_KB_ASSET_MAX_CONTAIN_DOCUMENT_NUMBER
    )

    config = query_by_config_type_and_name(session, str(asset.id), ServiceConfigType.DOC_COUNT)
    config_num = int(config.value) if config else None

    num_values = [value for value in [max_documents_num, default_num, config_num] if value is not None]

    return max(num_values)


def _count_asset_documents(knowledge_base_asset):
    return sum(len(vector_store.original_documents) for vector_store in knowledge_base_asset.vector_stores)


def _build_original_document_entity(document: OriginalDocument) -> OriginalDocumentEntity:
    return OriginalDocumentEntity(
        id=document.doc_id,
        uri=document.uri,
        source=document.source,
        online_url=document.online_url,
        mtime=document.mtime,
        display_name=document.display_name,
        download_key=document.download_key,
        extended_metadata=document.extended_metadata,
        security_level=document.security_level
    )


def _append_original_document_entities(vector_store, documents, can_add_documents_num, max_documents_num):
    processed_documents = []
    for idx, document in enumerate(documents):
        original_document_entity = _build_original_document_entity(document)
        if can_add_documents_num is None or idx < can_add_documents_num:
            original_document_entity.document_load_status = DocumentLoadStatus.LOADING
            processed_documents.append(document)
        else:
            original_document_entity.document_load_status = DocumentLoadStatus.QUOTA_EXCEEDED
            original_document_entity.error_info = DOC_IMPORT_FAIL_NUM_LIMIT.format(max_documents_num)
        vector_store.original_documents.append(original_document_entity)
    return processed_documents


@op(ins={"no_input": In(Nothing)}, out=DynamicOut(), retry_policy=RetryPolicy(max_retries=3))
def save_document_metadata(context: OpExecutionContext, documents: List[OriginalDocument]):
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    vector_store = VectorStore(name=uuid.uuid4().hex)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
        vectorization_config = VectorizationConfig(**knowledge_base_asset.vectorization_config)
        max_documents_num = get_max_documents_num(
            vectorization_config.max_documents_num,
            knowledge_base_asset,
            session
        )
        vector_store.knowledge_base_asset = knowledge_base_asset
        vector_store.knowledge_base_asset.vector_stores.append(vector_store)

        can_add_documents_num = _get_kba_document_num(
            knowledge_base_asset.asset_type, max_documents_num, _count_asset_documents(knowledge_base_asset))
        processed_documents = _append_original_document_entities(
            vector_store, documents, can_add_documents_num, max_documents_num
        )
        session.add(vector_store.knowledge_base_asset)
        session.commit()
        refresh_kba_id_count_and_max_create_to_redis(session, knowledge_base_asset.kb_id, knowledge_base_asset.id)

    # 记录文档更新详情
    insert_update_record(
        knowledge_base_serial_number, knowledge_base_asset_name, uuid.UUID(context.op_config["job_id"]),
        (set(), set(), {doc.source for doc in processed_documents}, processed_documents),
    )

    for idx, chunked_original_documents in enumerate(chunked(processed_documents, VECTORIZATION_CHUNK_SIZE)):
        yield DynamicOutput(chunked_original_documents, mapping_key=str(idx))


def _get_kba_document_num(asset: AssetType, max_documents_num: int, current_num: int):
    can_add_documents_num = None
    if asset in AssetType.types_require_limit_docs_num():
        can_add_documents_num = max(0, max_documents_num - current_num)
    return can_add_documents_num


def save_added_document_sources(
    context: OpExecutionContext,
    knowledge_base_serial_number: str,
    knowledge_base_asset_name: str,
    original_documents: List[OriginalDocument],
):
    add_source_list = [doc.source for doc in original_documents]
    documents_info_dir = get_documents_info_root_dir(knowledge_base_serial_number, knowledge_base_asset_name)
    add_dir = documents_info_dir / ADDED_DOCUMENTS_INFO_DIR
    add_dir.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())
    file_name = "added_documents_info_" + file_id + ".json"
    output_file = add_dir / file_name
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(add_source_list, f, indent=2)


@op(retry_policy=RetryPolicy(max_retries=3), tags={"dagster/priority": 1})
def parse_original_documents(
    context: OpExecutionContext, original_documents: List[OriginalDocument]
) -> List[Tuple[OriginalDocument, ParsedDocument]]:
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
        asset_type = knowledge_base_asset.asset_type
        vectorization_config = VectorizationConfig(**knowledge_base_asset.vectorization_config)

    parsed_documents = []
    if asset_type in AssetType.types_skip_loader_and_split():
        for original_document in original_documents:
            chunks = get_text_slices(Path(original_document.uri))
            parsed_blocks = documents_to_parsed_blocks(chunks, split_policy=SPLIT_POLICY_NO_SPLIT)
            parsed_documents.append((
                original_document,
                blocks_to_parsed_document(parsed_blocks),
            ))
        return parsed_documents

    for original_document in original_documents:
        parsed_document = parse_file(original_document, vectorization_config, asset_type)
        parsed_documents.append((original_document, parsed_document))
    return parsed_documents


@op(retry_policy=RetryPolicy(max_retries=1), tags={"dagster/priority": 1})
def save_parsed_documents(
    context: OpExecutionContext, parsed_documents: List[Tuple[OriginalDocument, ParsedDocument]]
) -> List[Tuple[OriginalDocument, ParsedDocument]]:
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    artifact_summaries = {}
    for original_document, parsed_document in parsed_documents:
        artifact_info = _persist_parsed_document_artifacts(
            knowledge_base_serial_number,
            knowledge_base_asset_name,
            original_document,
            parsed_document,
        )
        if artifact_info:
            artifact_summaries[original_document.doc_id] = artifact_info
    if artifact_summaries:
        with Session(engine) as session:
            _save_artifact_summaries(session, artifact_summaries)
            session.commit()
    return parsed_documents


def _persist_parsed_document_artifacts(
    knowledge_base_serial_number: str,
    knowledge_base_asset_name: str,
    original_document: OriginalDocument,
    parsed_document: ParsedDocument,
) -> Dict[str, Any]:
    for persist_func, metadata_key in _artifact_persisters():
        summary = persist_func(
            parsed_document,
            knowledge_base_serial_number,
            knowledge_base_asset_name,
            str(original_document.doc_id),
            upload_file_as_bytes,
        )
        if not summary:
            continue
        original_document.extended_metadata = _merge_artifact_metadata(
            original_document.extended_metadata,
            metadata_key,
            summary,
        )
        return {"metadata_key": metadata_key, "summary": summary}
    return {}


def _artifact_persisters():
    return [
        (persist_structured_docx_artifacts, STRUCTURED_DOCX_METADATA_KEY),
        (persist_structured_excel_artifacts, STRUCTURED_EXCEL_METADATA_KEY),
        (persist_structured_html_artifacts, STRUCTURED_HTML_METADATA_KEY),
        (persist_structured_markdown_artifacts, STRUCTURED_MARKDOWN_METADATA_KEY),
        (persist_parsed_markdown_artifact, PARSED_MARKDOWN_METADATA_KEY),
    ]


def _merge_artifact_metadata(
    extended_metadata: Optional[Dict[str, Any]],
    metadata_key: str,
    summary: Dict[str, Any],
):
    if metadata_key == STRUCTURED_DOCX_METADATA_KEY:
        return merge_structured_docx_metadata(extended_metadata, summary)
    return merge_structured_metadata(extended_metadata, metadata_key, summary)


def _save_artifact_summaries(session: Session, artifact_summaries: Dict[Any, Dict[str, Any]]) -> None:
    summaries_by_id = {str(doc_id): info for doc_id, info in artifact_summaries.items()}
    documents = query_document_by_ids(session, list(artifact_summaries.keys()))
    for document in documents:
        artifact_info = summaries_by_id.get(str(document.id))
        if artifact_info:
            document.extended_metadata = merge_structured_metadata(
                document.extended_metadata,
                artifact_info["metadata_key"],
                artifact_info["summary"],
            )
    session.add_all(documents)


@op(retry_policy=RetryPolicy(max_retries=3), tags={"dagster/priority": 1})
def split_parsed_documents(
    context: OpExecutionContext, parsed_documents: List[Tuple[OriginalDocument, ParsedDocument]]
) -> List[Tuple[OriginalDocument, List[Document]]]:
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
        asset_type = knowledge_base_asset.asset_type
        kb_allow_synchronous_update = knowledge_base_asset.knowledge_base.allow_synchronous_update
        vectorization_config = VectorizationConfig(**knowledge_base_asset.vectorization_config)

    document_chunks = []
    for original_document, parsed_document in parsed_documents:
        chunks = split_parsed_file(parsed_document, original_document, vectorization_config, asset_type)
        document_chunks.append((original_document, chunks))

    if kb_allow_synchronous_update and document_chunks:
        save_added_document_sources(
            context,
            knowledge_base_serial_number,
            knowledge_base_asset_name,
            [doc for doc, _ in document_chunks],
        )
    return document_chunks


@op(retry_policy=RetryPolicy(max_retries=3), tags={"dagster/priority": 1})
def load_original_documents(
    context: OpExecutionContext, original_documents: List[OriginalDocument]
) -> List[Tuple[OriginalDocument, List[Document]]]:
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
        asset_type = knowledge_base_asset.asset_type
        kb_allow_synchronous_update = knowledge_base_asset.knowledge_base.allow_synchronous_update
    # 对于已经在fetch阶段直接组装成切片的文档内容（如issue类），此处跳过加载和切片的步骤，直接取出组装好的切片即可
    document_chunks = []
    if knowledge_base_asset.asset_type in AssetType.types_skip_loader_and_split():
        for original_document in original_documents:
            chunks = get_text_slices(Path(original_document.uri))
            document_chunks.append((original_document, chunks))
    else:
        for original_document in original_documents:
            chunks = load_file(
                original_document, VectorizationConfig(**knowledge_base_asset.vectorization_config), asset_type
            )
            document_chunks.append((original_document, chunks))
    if kb_allow_synchronous_update and document_chunks:
        add_source_list = [doc.source for doc in original_documents]
        documents_info_dir = get_documents_info_root_dir(knowledge_base_serial_number, knowledge_base_asset_name)
        add_dir = documents_info_dir / ADDED_DOCUMENTS_INFO_DIR
        add_dir.mkdir(parents=True, exist_ok=True)
        file_id = str(uuid.uuid4())
        file_name = "added_documents_info_" + file_id + ".json"
        output_file = add_dir / file_name
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(add_source_list, f, indent=2)
    return document_chunks


@op()
def document_structure_detect(context: OpExecutionContext, documents: List[Tuple[OriginalDocument, List[Document]]]):
    if should_skip_corpus_quality_detection(context):
        return
    kb_sn, kba_name = parse_asset_partition_key(context.partition_key)
    kb_id, kba_id, _ = _get_kb_id_and_kba_id_by_context(kb_sn, kba_name)
    job_id = uuid.UUID(context.op_config["job_id"])
    detector = DocumentStructureDetector()
    operation_details = detector.detect(documents, job_id)
    detector.save_record(job_id, CorpusAccessControlPolicy.DOCUMENT_STRUCTURE.value, operation_details, kb_id, kba_id)


@op()
def punctuation_detect(context: OpExecutionContext, documents: List[Tuple[OriginalDocument, List[Document]]]):
    if should_skip_corpus_quality_detection(context):
        return
    kb_sn, kba_name = parse_asset_partition_key(context.partition_key)
    kb_id, kba_id, _ = _get_kb_id_and_kba_id_by_context(kb_sn, kba_name)
    job_id = uuid.UUID(context.op_config["job_id"])
    detector = PunctuationRatioDetector()
    operation_details = detector.detect(documents, job_id)
    detector.save_record(
        job_id, CorpusAccessControlPolicy.PUNCTUATION_PERCENTAGE.value, operation_details, kb_id, kba_id
    )


@op()
def document_url_detect(context: OpExecutionContext, documents: List[Tuple[OriginalDocument, List[Document]]]):
    if should_skip_corpus_quality_detection(context):
        return
    kb_sn, kba_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(session, kb_sn, kba_name)
    # PDM的文档包含大量链接，但是都是pdm平台的有效链接，无需检测
    if knowledge_base_asset.asset_type == AssetType.PDM_PART:
        return

    job_id = uuid.UUID(context.op_config["job_id"])
    detector = UrlValidityDetector()
    operation_details = detector.detect(documents, job_id)
    detector.save_record(job_id, CorpusAccessControlPolicy.URL_VALIDITY.value, operation_details,
                         knowledge_base_asset.kb_id, knowledge_base_asset.id)


@op()
def document_similarity_detect(
    context: OpExecutionContext, ins: Tuple[List[Tuple[OriginalDocument, List[Document]]], str, str, str]
):
    if should_skip_corpus_quality_detection(context):
        return
    documents_info, vectorization_job_id, kb_sn, kba_name = ins
    vectorization_job_id = uuid.UUID(vectorization_job_id)
    kb_id, kba_id, _ = _get_kb_id_and_kba_id_by_context(kb_sn, kba_name)
    detector = SimilarDocumentDetector()
    operation_details = detector.detect(documents_info, vectorization_job_id, {"kb_sn": kb_sn})
    detector.save_record(
        vectorization_job_id, CorpusAccessControlPolicy.DOCUMENT_SIMILARITY.value, operation_details, kb_id, kba_id
    )


@op()
def document_consistency_detect(
    context: OpExecutionContext, ins: Tuple[List[Tuple[OriginalDocument, List[Document]]], str, str, str]
):
    if should_skip_corpus_quality_detection(context):
        return
    documents_info, vectorization_job_id, kb_sn, kba_name = ins
    vectorization_job_id = uuid.UUID(vectorization_job_id)
    kb_id, kba_id, _ = _get_kb_id_and_kba_id_by_context(kb_sn, kba_name)
    detector = ConsistencyDetector()
    operation_details = detector.detect(documents_info, vectorization_job_id, {"kb_sn": kb_sn})
    detector.save_record(
        vectorization_job_id, CorpusAccessControlPolicy.DOCUMENT_CONSISTENCY.value, operation_details, kb_id, kba_id
    )


@op()
def document_semantic_coherence_detect(
    context: OpExecutionContext, ins: Tuple[List[Tuple[OriginalDocument, List[Document]]], str, str, str]
):
    if should_skip_corpus_quality_detection(context):
        return
    documents_info, vectorization_job_id, kb_sn, kba_name = ins
    vectorization_job_id = uuid.UUID(vectorization_job_id)
    kb_id, kba_id, _ = _get_kb_id_and_kba_id_by_context(kb_sn, kba_name)
    detector = SemanticCoherenceDetector()
    operation_details = detector.detect(documents_info, vectorization_job_id)
    detector.save_record(
        vectorization_job_id, CorpusAccessControlPolicy.SEMANTIC_COHERENCE.value, operation_details, kb_id, kba_id
    )


@op()
def document_semantic_integrity_detect(
    context: OpExecutionContext, document_infos: List[Tuple[OriginalDocument, List[Document], List[List[float]]]]
):
    if should_skip_corpus_quality_detection(context):
        return
    kb_sn, kba_name = parse_asset_partition_key(context.partition_key)
    kb_id, kba_id, embedding_model = _get_kb_id_and_kba_id_by_context(kb_sn, kba_name)
    job_id = uuid.UUID(context.op_config["job_id"])
    detector = SemanticIntegrityDetector()
    operation_details = detector.detect(
        document_infos, job_id, {"embedding_model": embedding_model, "kb_sn": kb_sn, "asset_name": kba_name}
    )
    detector.save_record(job_id, CorpusAccessControlPolicy.SEMANTIC_INTEGRITY.value, operation_details, kb_id, kba_id)


@op()
def document_logic_detect(
    context: OpExecutionContext, ins: Tuple[List[Tuple[OriginalDocument, List[Document]]], str, str, str]
):
    if should_skip_corpus_quality_detection(context):
        return
    documents_info, vectorization_job_id, kb_sn, kba_name = ins
    vectorization_job_id = uuid.UUID(vectorization_job_id)
    kb_id, kba_id, _ = _get_kb_id_and_kba_id_by_context(kb_sn, kba_name)
    detector = LogicDetector()
    operation_details = detector.detect(documents_info, vectorization_job_id)
    detector.save_record(
        vectorization_job_id, CorpusAccessControlPolicy.DOCUMENT_LOGIC.value, operation_details, kb_id, kba_id
    )


@op()
def sensitive_words_detect(context: OpExecutionContext, documents: List[Tuple[OriginalDocument, List[Document]]]):
    kb_sn, kba_name = parse_asset_partition_key(context.partition_key)
    kb_id, kba_id, _ = _get_kb_id_and_kba_id_by_context(kb_sn, kba_name)
    job_id = uuid.UUID(context.op_config["job_id"])
    detector = SensitiveWordDetector()
    operation_details = detector.detect(documents)
    detector.save_record(
        job_id, CorpusAccessControlPolicy.SENSITIVE_WORD.value, operation_details, kb_id, kba_id
    )
    return documents


@op()
def dummy_immediate_consumer(
    vectorize_result,
    url_valid_detect=In(Nothing),
    punctuation_ratio_detect=In(Nothing),
    deep_document_structure_detect=In(Nothing),
    semantic_integrity_detect=In(Nothing),
    delayed_detection_job_info=In(Nothing),
):
    return vectorize_result


@op()
def dummy_delay_consumer(
    context: OpExecutionContext,
    similar_document_detect=In(Nothing),
    non_consistence_document_detect=In(Nothing),
    semantic_coherence_detect=In(Nothing),
    readability_detect=In(Nothing),
    logic_detect=In(Nothing)
):
    job_id = context.op_config["job_id"]
    with Session(engine) as session:
        job = get_vectorization_job_info(session, job_id)
        file_path = Path(job.extra_info["save_path"])
        file_path.unlink(missing_ok=True)
        job.status = JobStatus.SUCCESS
        session.add(job)
        session.commit()


# 为每个文档对创建唯一标识
@op(retry_policy=RetryPolicy(max_retries=1))
def save_delayed_detection_document_data(
    context: OpExecutionContext, documents: List[Tuple[OriginalDocument, List[Document]]]
):
    """保存当前任务信息，用于延迟检测，并下发延迟检测的任务"""
    vectorization_job_id = context.op_config["job_id"]

    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)

    if should_skip_corpus_quality_detection(context):
        return

    # 保存路径
    save_dir = get_detection_info_root_dir(knowledge_base_serial_number, knowledge_base_asset_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{vectorization_job_id}.pkl"

    # 保存数据
    with open(save_path, "wb") as f:
        pickle.dump(documents, f)

    logger.info(f"任务id {vectorization_job_id} 下的向量化文档内容已成功保存")
    # 启动延迟检测任务
    delayed_detection_job = AutoJobInstances(
        status=JobStatus.PENDING,
        job_type=VectorizationJobType.DELAYED_DOCUMENT_DETECTION,
        extra_info={
            "kb_sn": knowledge_base_serial_number,
            "asset_name": knowledge_base_asset_name,
            "vectorization_job_id": vectorization_job_id,
            "save_path": str(save_path),
        },
    )

    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
        knowledge_base_asset.auto_job_instances.append(delayed_detection_job)
        session.add(knowledge_base_asset)
        session.commit()


@op(retry_policy=RetryPolicy(max_retries=1), out=DynamicOut(Tuple[List, Any, Any, Any]))
def load_document_pairs(context: OpExecutionContext):
    """加载所有保存的文档对"""
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    save_dir = get_detection_info_root_dir(knowledge_base_serial_number, knowledge_base_asset_name)
    job_id = context.op_config["job_id"]
    with Session(engine) as session:
        job = get_vectorization_job_info(session, job_id)
        job_info = job.extra_info

    pkl_file = Path(job_info["save_path"])
    if not save_dir.exists() or not pkl_file.exists():
        context.log.warning(f"检测数据文件不存在: {pkl_file}")
        return

    # 加载对应的 pickle 文件
    with open(pkl_file, "rb") as f:
        documents_info = pickle.load(f)

    if not job_info["vectorization_job_id"] or not documents_info:
        yield DynamicOutput(value=[], mapping_key="empty")
    else:
        # 生成动态输出
        yield DynamicOutput(
            value=(documents_info, job_info["vectorization_job_id"], job_info["kb_sn"], job_info["asset_name"]),
            mapping_key="empty",
        )


@op(retry_policy=RetryPolicy(max_retries=3), tags={"dagster/priority": 2})
def embedding_documents(
    context: OpExecutionContext, document_chunks: List[Tuple[OriginalDocument, List[Document]]]
) -> List[Tuple[OriginalDocument, List[Document], List[List[float]]]]:
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
    try:
        # 1. 收集所有chunks并记录每个文档的chunk数量
        all_chunks = []
        doc_chunk_counts = []
        for original_doc, doc_chunks in document_chunks:
            all_chunks.extend(doc_chunks)
            doc_chunk_counts.append(len(doc_chunks))

        # 2. 批量生成所有embeddings（保持原有的批量调用效率）
        all_embeddings = list(
            itertools.chain.from_iterable([
                embedding(
                    tuple(document.page_content for document in chunked_batch), knowledge_base_asset.embedding_model
                )
                for chunked_batch in chunked(all_chunks, EMBEDDING_CHUNK_SIZE)
            ])
        )

        # 3. 按原始文档分组，将embeddings分配回对应的chunks
        result = []
        embedding_offset = 0

        for (original_doc, doc_chunks), chunk_count in zip(document_chunks, doc_chunk_counts):
            # 切片获取当前文档对应的embeddings
            doc_embeddings = all_embeddings[embedding_offset: embedding_offset + chunk_count]
            embedding_offset += chunk_count

            result.append((original_doc, doc_chunks, doc_embeddings))

        return result
    except Exception as e:
        logger.exception("Embedding error")
        doc_id_list = []
        for original_doc, doc_chunks in document_chunks:
            doc_id_list.append(original_doc.doc_id)
        update_doc_error_info(doc_id_list, DOC_IMPORT_FAIL_EMBEDDING, DocumentLoadStatus.FAILURE)
        raise e


@op(retry_policy=RetryPolicy(max_retries=3), tags={"dagster/priority": 3})
def store_chunks_to_vector_db(
    context: OpExecutionContext, document_infos: List[Tuple[OriginalDocument, List[Document], List[List[float]]]]
) -> Tuple[List[uuid.UUID], Optional[str]]:
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
        vector_stores = knowledge_base_asset.vector_stores
        knowledge_base = validate_knowledge_base(session, knowledge_base_serial_number)
        synonyms = get_knowledge_base_synonyms_or_stopwords(knowledge_base, StopWordOrSynonymType.SYNONYM)
        stopwords = get_knowledge_base_synonyms_or_stopwords(knowledge_base, StopWordOrSynonymType.STOPWORD)

    vector_store_name = vector_stores[-1].name
    doc_ids = [doc.doc_id for doc, _, _ in document_infos]
    all_chunks = list(itertools.chain.from_iterable(doc_chunks for _, doc_chunks, _ in document_infos))
    all_embeddings = list(itertools.chain.from_iterable(chunk_embeddings for _, _, chunk_embeddings in document_infos))

    get_vector_store_manager().add_documents(
        all_chunks, all_embeddings, vector_store_name, knowledge_base.analyzer, synonyms, stopwords
    )

    return doc_ids, vector_store_name


@op(retry_policy=RetryPolicy(max_retries=3), tags={"dagster/priority": 4})
def update_document_status(context: OpExecutionContext, ins: Tuple[List[uuid.UUID], Optional[str]]):
    doc_ids, vector_store_name = ins
    if not vector_store_name:
        return

    with Session(engine) as session:
        documents = query_document_by_ids(session, doc_ids)
        for doc in documents:
            if doc.document_load_status == DocumentLoadStatus.LOADING:
                doc.document_load_status = DocumentLoadStatus.SUCCESS

        session.add_all(documents)
        session.commit()


@op(ins={"no_input": In(Nothing)})
def no_op_fan_in(): ...


@op(ins={"no_input": In(Nothing)}, retry_policy=RetryPolicy(max_retries=3))
def delete_original_documents(context: OpExecutionContext):
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    knowledge_base_asset_dir = get_knowledge_base_asset_root_dir(
        knowledge_base_serial_number, knowledge_base_asset_name
    )
    delete_dir(get_obs_dir(knowledge_base_serial_number, knowledge_base_asset_name))
    if Path(knowledge_base_asset_dir).exists():
        shutil.rmtree(knowledge_base_asset_dir, onerror=handle_remove_readonly)


@op(ins={"no_input": In(Nothing)}, retry_policy=RetryPolicy(max_retries=3))
def change_vectorization_job_status_to_success(context: OpExecutionContext):
    with Session(engine) as session:
        vectorization_job = session.exec(
            select(AutoJobInstances).where(AutoJobInstances.id == uuid.UUID(context.op_config["job_id"]))
        ).one()
        change_vectorization_job_status(session, vectorization_job, JobStatus.SUCCESS)
        if (
                (vectorization_job.job_type == VectorizationJobType.INIT
                 or vectorization_job.job_type == VectorizationJobType.RELOAD)
                and vectorization_job.knowledge_base_asset.knowledge_base.allow_synchronous_update
        ):
            # 如果是重导入操作，新增一个ipd同步更新的任务
            job_type = (VectorizationJobType.INIT_IN_IPD_RAG
                        if vectorization_job.job_type == VectorizationJobType.INIT
                        else VectorizationJobType.INCREMENTAL_IN_IPD_RAG)
            ipd_rag_job = AutoJobInstances(status=JobStatus.PENDING, job_type=job_type)
            vectorization_job.knowledge_base_asset.auto_job_instances.append(ipd_rag_job)
            session.add(vectorization_job)
            session.commit()


@op(ins={"no_input": In(Nothing)}, retry_policy=RetryPolicy(max_retries=3))
def update_knowledge_base_and_asset_updated_at(context: OpExecutionContext):
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
        updated_at = now_with_time_zone()
        knowledge_base_asset.updated_at = updated_at
        knowledge_base_asset.knowledge_base.updated_at = updated_at
        session.add(knowledge_base_asset)
        session.commit()


def parse_documents_info(root_path: Path) -> Generator[Union[Dict[str, Any], List[str]], None, None]:
    for path in root_path.iterdir():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            yield data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.exception("Error processing %s: %s", path, str(e))


@op(ins={"no_input": In(Nothing)}, out=DynamicOut(dagster_type=List[str]), retry_policy=RetryPolicy(max_retries=3))
def get_added_documents_info(context: OpExecutionContext) -> DynamicOut[List[str]]:
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    documents_info_dir = get_documents_info_root_dir(knowledge_base_serial_number, knowledge_base_asset_name)
    added_documents_info_dir = documents_info_dir / ADDED_DOCUMENTS_INFO_DIR
    if not added_documents_info_dir.exists():
        yield DynamicOutput(value=[], mapping_key="empty")
    else:
        added_documents_info_list = parse_documents_info(added_documents_info_dir)
        for idx, added_documents_info in enumerate(added_documents_info_list):
            yield DynamicOutput(value=added_documents_info, mapping_key=str(idx))


@op(retry_policy=RetryPolicy(max_retries=3), tags={"dagster/priority": 1})
def get_added_documents_index(context: OpExecutionContext, sources: List[str]) -> Dict[str, Any]:
    if not sources:
        return {}
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        results = get_kba_document_index_collection_by_source_list(
            session, knowledge_base_serial_number, knowledge_base_asset_name, sources
        )
    documents_index_dict = {}
    for row in results:
        documents_index_dict[row.source] = {"display_name": row.display_name, "index": row.name if row.name else "", "doc_id": row.id}
    return documents_index_dict


def _get_document_name(source: str, document_index: Dict[str, Any]) -> str:
    return document_index.get("display_name", "") or source


def _append_regular_document_entry(state, source, document_name, slice_list):
    document_id = str(uuid.uuid4())
    document_entry = create_document_entry(document_id, source, document_name, DATAOPS_OPERATION_INSERT, slice_list)
    document_entries = split_document_entry_for_dataops(
        document_entry,
        document_id_factory=lambda: str(uuid.uuid4()),
    )
    state["source_document_list_dict"][source] = _document_entries_to_ipd_rag_document_list(document_entries)
    _append_document_entries_to_state(state, document_entries)


def _append_large_document_entries(
    state, es_manager, ipd_rag_kb_sn, kb_sn, source, document_name, doc_id, scroll_id, slice_list
):
    ipd_rag_document_list = []
    count = 0
    current_document_name = document_name
    while slice_list:
        count += 1
        document_id = str(uuid.uuid4())
        current_document_name = current_document_name + "_" + str(count)
        ipd_rag_document_list.append({"document_name": current_document_name, "document_id": document_id})
        document_entry = create_document_entry(document_id, source, current_document_name, DATAOPS_OPERATION_INSERT,
                                               slice_list)
        document_entries = split_document_entry_for_dataops(
            document_entry,
            document_id_factory=lambda: str(uuid.uuid4()),
        )
        ipd_rag_document_list[-1:] = _document_entries_to_ipd_rag_document_list(document_entries)
        if len(slice_list) < MAXIMUM_NUMBER_OF_SLICES:
            _append_document_entries_to_state(state, document_entries)
            break
        send_document_entries_to_dataops(send_data_to_dataops, ipd_rag_kb_sn, kb_sn, document_entries, doc_id)
        scroll_id, slice_list = es_manager.iterator_batch_search_document_data_by_source(scroll_id)
    state["source_document_list_dict"][source] = ipd_rag_document_list


def _document_entries_to_ipd_rag_document_list(document_entries):
    return [
        {"document_name": document_entry["filename"], "document_id": document_entry["id"]}
        for document_entry in document_entries
    ]


def _append_document_entries_to_state(state, document_entries):
    state["document_entry_list"].extend(document_entries)
    state["slice_count"] += sum(len(document_entry.get("slices") or []) for document_entry in document_entries)


def _collect_source_document_entries(state, es_manager, ipd_rag_kb_sn, kb_sn, source, document_index):
    document_name = _get_document_name(source, document_index)
    index = document_index["index"]
    doc_id = str(document_index["doc_id"])
    scroll_id, slice_list = es_manager.first_batch_search_document_data_by_source(
        index, source, MAXIMUM_NUMBER_OF_SLICES
    )
    if state["document_entry_list"] and state["slice_count"] + len(slice_list) >= MAXIMUM_NUMBER_OF_SLICES:
        send_document_entries_to_dataops(
            send_data_to_dataops,
            ipd_rag_kb_sn,
            kb_sn,
            state["document_entry_list"],
            doc_id,
        )
        state["document_entry_list"] = []
        state["slice_count"] = 0
    if len(slice_list) < MAXIMUM_NUMBER_OF_SLICES:
        _append_regular_document_entry(state, source, document_name, slice_list)
    else:
        _append_large_document_entries(state, es_manager, ipd_rag_kb_sn, kb_sn, source, document_name, doc_id, scroll_id,
                                       slice_list)
    es_manager.clear_client(scroll_id=scroll_id)
    return doc_id


@op(retry_policy=RetryPolicy(max_retries=3), tags={"dagster/priority": 2})
def add_asset_documents_in_ipd_rag(
    context: OpExecutionContext, documents_index_dict: Dict[str, Any]
) -> Dict[str, List[Dict[str, str]]]:
    if not documents_index_dict:
        return {}
    kb_sn, asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base = get_knowledge_base_by_kb_sn(session, kb_sn)
    ipd_rag_knowledge_base_set_sn = knowledge_base.ipd_rag_kb_sn

    state = {"source_document_list_dict": {}, "document_entry_list": [], "slice_count": 0}
    es_manager = get_vector_store_manager()
    doc_id = ""
    for source, document_index in documents_index_dict.items():
        doc_id = _collect_source_document_entries(
            state, es_manager, ipd_rag_knowledge_base_set_sn, kb_sn, source, document_index
        )
    if state["document_entry_list"]:
        send_document_entries_to_dataops(
            send_data_to_dataops,
            ipd_rag_knowledge_base_set_sn,
            kb_sn,
            state["document_entry_list"],
            doc_id,
        )
    return state["source_document_list_dict"]


@op(retry_policy=RetryPolicy(max_retries=3), tags={"dagster/priority": 3})
def save_document_list_to_database(context: OpExecutionContext, source_document_list_dict: Dict):
    if source_document_list_dict:
        kb_sn, asset_name = parse_asset_partition_key(context.partition_key)
        with Session(engine) as session:
            document_source_list: Set[str] = set(source_document_list_dict.keys())
            original_documents = get_original_documents_by_source_list(session, kb_sn, asset_name, document_source_list)
            for original_document in original_documents:
                original_document.ipd_rag_document_list = source_document_list_dict[original_document.source]
                session.add(original_document)
            session.commit()


@op(ins={"no_input": In(Nothing)}, retry_policy=RetryPolicy(max_retries=4))
def delete_documents_info(context: OpExecutionContext):
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    documents_info_dir = get_documents_info_root_dir(knowledge_base_serial_number, knowledge_base_asset_name)
    if Path(documents_info_dir).exists():
        shutil.rmtree(documents_info_dir)


def clean_table_text(text: str) -> str:
    """清理表格结构，提取纯文本内容"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_lines = []
    for line in lines:
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        if cells:
            cleaned_lines.append(" ".join(cells))
    return "\n".join(cleaned_lines)


def split_text(general_text: str, extended_metadata: str, slice_entry_list: List[Dict[str, Any]]):
    cleaned_text = clean_table_text(general_text)
    if not cleaned_text.strip():
        return

    sentences = []
    for s in re.split(r"(?<=[.!?。？！])\s*", cleaned_text):
        stripped = s.strip()
        if stripped:
            sentences.append(stripped)
    current_block = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        # 尝试将当前句子加入当前块
        if current_length + sentence_length > 5120:
            # 当前块已满，保存并开始新块
            temp_text = "".join(current_block)
            if not temp_text:
                continue
            slice_entry_list.append({
                "text": temp_text,
                "meta_data": {"extended_metadata": get_doc_metadata(extended_metadata)},
            })
            current_block = [sentence]
            current_length = sentence_length
        else:
            current_block.append(sentence)
            current_length += sentence_length
    if current_block:
        temp_text = "".join(current_block)
        if temp_text:
            slice_entry_list.append({"text": temp_text, "meta_data": {"extended_metadata": get_doc_metadata(extended_metadata)}})


def get_doc_metadata(extended_metadata: str):
    dic = deserialize(extended_metadata)
    return {k: v for k, v in dic.items() if k != "header_contents"}


def create_document_entry(
    document_id: str,
    source: str,
    document_name: str,
    operation: int,
    data_list: List[EsQueryResult] = None,
):
    slice_entry_list = []
    if data_list:
        for data in data_list:
            general_text = data.general_text.strip()
            if not general_text:
                continue
            if len(general_text) > 5120:
                split_text(general_text, data.extended_metadata, slice_entry_list)
            else:
                slice_entry = {
                    "text": general_text,
                    "meta_data": {"extended_metadata": get_doc_metadata(data.extended_metadata)},
                }
                slice_entry_list.append(slice_entry)
    if not slice_entry_list:
        slice_entry_list = [
            {
                "text": "无切片内容",
                "meta_data": {"extended_metadata": get_doc_metadata(data_list[0].extended_metadata) if data_list else {}},
            }
        ]
    return {
        "operation": operation,
        "filename": document_name,
        "title": document_name,
        "uri": source,
        "id": document_id,
        "from": "Libing_rag",
        "slices": slice_entry_list,
    }


def record_delete_doc(
    session, knowledge_base_serial_number, knowledge_base_asset_name, job_id, deleted_original_document_source_set
):
    deleted_original_document_list = get_original_documents_by_source_list(
        session, knowledge_base_serial_number, knowledge_base_asset_name, deleted_original_document_source_set
    )
    for deleted_original_document in deleted_original_document_list:
        delete_record = UpdatedOriginalDocument(
            update_type=UpdateOriginalDocumentType.DELETE,
            source=deleted_original_document.source,
            display_name=get_original_document_display_name(
                deleted_original_document, knowledge_base_serial_number, knowledge_base_asset_name
            ),
            job_id=job_id,
        )
        session.add(delete_record)


def record_update_doc(
    session, knowledge_base_serial_number, knowledge_base_asset_name, job_id, updated_original_document_source_set
):
    updated_original_document_list = get_original_documents_by_source_list(
        session, knowledge_base_serial_number, knowledge_base_asset_name, updated_original_document_source_set
    )
    for updated_original_document in updated_original_document_list:
        update_record = UpdatedOriginalDocument(
            update_type=UpdateOriginalDocumentType.UPDATE,
            source=updated_original_document.source,
            display_name=get_original_document_display_name(
                updated_original_document, knowledge_base_serial_number, knowledge_base_asset_name
            ),
            job_id=job_id,
        )
        session.add(update_record)


def record_increment_doc(
    session,
    knowledge_base_serial_number,
    knowledge_base_asset_name,
    job_id,
    uploaded_original_documents,
    incremented_original_document_set,
):
    incremented_original_document_list = [
        doc for doc in uploaded_original_documents if doc.source in incremented_original_document_set
    ]
    for incremented_original_document in incremented_original_document_list:
        increment_record = UpdatedOriginalDocument(
            update_type=UpdateOriginalDocumentType.INCREMENTAL,
            source=incremented_original_document.source,
            display_name=get_original_document_display_name(
                incremented_original_document, knowledge_base_serial_number, knowledge_base_asset_name
            ),
            job_id=job_id,
        )
        session.add(increment_record)


def insert_update_record(
    knowledge_base_serial_number: str,
    knowledge_base_asset_name: str,
    job_id: uuid.UUID,
    ins: Tuple[Set[str], Set[str], Set[str], List[OriginalDocument]],
):
    (
        union_deleted_original_document_set,
        updated_original_document_set,
        incremented_original_document_set,
        uploaded_original_documents,
    ) = ins
    with Session(engine) as session:
        if union_deleted_original_document_set - updated_original_document_set:
            record_delete_doc(
                session,
                knowledge_base_serial_number,
                knowledge_base_asset_name,
                job_id,
                union_deleted_original_document_set - updated_original_document_set,
            )

        if updated_original_document_set:
            record_update_doc(
                session, knowledge_base_serial_number, knowledge_base_asset_name, job_id, updated_original_document_set
            )
        if uploaded_original_documents:
            record_increment_doc(
                session,
                knowledge_base_serial_number,
                knowledge_base_asset_name,
                job_id,
                uploaded_original_documents,
                incremented_original_document_set,
            )
        session.commit()


@op()
def document_readability_detect(
    context: OpExecutionContext, ins: Tuple[List[Tuple[OriginalDocument, List[Document]]], str, str, str]
):
    if should_skip_corpus_quality_detection(context):
        return
    documents_info, vectorization_job_id, kb_sn, kba_name = ins
    vectorization_job_id = uuid.UUID(vectorization_job_id)
    kb_id, kba_id, _ = _get_kb_id_and_kba_id_by_context(kb_sn, kba_name)
    detector = ReadabilityDetector()
    operation_details = detector.detect(documents_info, vectorization_job_id)
    detector.save_record(
        vectorization_job_id, CorpusAccessControlPolicy.READABILITY.value, operation_details, kb_id, kba_id
    )


def _get_kb_id_and_kba_id_by_context(kb_sn: str, kba_name: str):
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(session, kb_sn, kba_name)
        kb_id, kba_id = knowledge_base_asset.kb_id, knowledge_base_asset.id
    return kb_id, kba_id, knowledge_base_asset.embedding_model
