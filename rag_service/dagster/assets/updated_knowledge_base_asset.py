import collections
import json
import uuid
from typing import List, Set, Tuple, Type

from dagster import DynamicOut, DynamicOutput, In, Nothing, OpExecutionContext, RetryPolicy, graph_asset, op
from more_itertools import chunked
from sqlalchemy import delete
from sqlmodel import Session, col, select, or_

from rag_service.config import (
    VECTORIZATION_CHUNK_SIZE,
)
from rag_service.constants import DELETED_DOCUMENTS_INFO_DIR, DOC_IMPORT_FAIL_NUM_LIMIT
from rag_service.dagster.dagster_common_op import (
    change_vectorization_job_status_to_started,
    delete_original_documents,
    document_semantic_integrity_detect,
    document_structure_detect,
    document_url_detect,
    dummy_immediate_consumer,
    embedding_documents,
    insert_update_record,
    load_original_documents,
    no_op_fan_in,
    punctuation_detect,
    save_delayed_detection_document_data,
    store_chunks_to_vector_db,
    update_document_status,
    update_knowledge_base_and_asset_updated_at, document_deduplication, get_max_documents_num, sensitive_words_detect,
)
from rag_service.dagster.partitions.knowledge_base_asset_partition import knowledge_base_asset_partitions_def
from rag_service.database import engine
from rag_service.models.database.models import (
    AutoJobInstances,
    KnowledgeBase,
    KnowledgeBaseAsset,
    VectorStore, DocumentDefect, RequestDocumentHit
)
from rag_service.models.database.models import (
    OriginalDocument as OriginalDocumentEntity,
)
from rag_service.models.enums import (
    AssetType,
    DocumentLoadStatus,
    JobStatus,
    VectorizationJobType,
)
from rag_service.models.generic.models import OriginalDocument, VectorizationConfig
from rag_service.original_document_fetchers import Fetcher, select_fetcher
from rag_service.rag_app.dao.request_document_hit_dao import refresh_kba_id_count_and_max_create_to_redis
from rag_service.utils.dagster_util import (
    get_documents_info_root_dir,
    get_knowledge_base_asset_root_dir,
    parse_asset_partition_key,
)
from rag_service.utils.db_util import (
    change_vectorization_job_status,
    get_knowledge_base_asset,
    get_original_documents_by_source_list,
)
from rag_service.utils.time_util import now_with_time_zone
from rag_service.vectorstore import get_vector_store_manager


@op(ins={"no_input": In(Nothing)}, retry_policy=RetryPolicy(max_retries=3))
def fetch_updated_original_document_set(
    context: OpExecutionContext,
) -> Tuple[Set[str], Set[str], Set[str], List[OriginalDocument]]:
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    knowledge_base_asset_dir = get_knowledge_base_asset_root_dir(
        knowledge_base_serial_number, knowledge_base_asset_name
    )
    original_document_sources = []
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
        for vector_store in knowledge_base_asset.vector_stores:
            original_document_sources.extend(
                original_document.source for original_document in vector_store.original_documents
            )
    fetcher: Type[Fetcher] = select_fetcher(knowledge_base_asset.asset_type)
    vectorization_config = VectorizationConfig(**knowledge_base_asset.vectorization_config)
    deleted_original_document_sources, uploaded_original_document_sources, uploaded_original_documents = fetcher(
        asset_uri=knowledge_base_asset.asset_uri,
        asset_root_dir=knowledge_base_asset_dir,
        asset_type=knowledge_base_asset.asset_type,
        save_file=knowledge_base_asset.save_file,
        local_file_online_dic=vectorization_config.local_file_online_dic,
        attachment_vectorize=vectorization_config.attachment_vectorize,
    ).update_fetch(knowledge_base_serial_number, knowledge_base_asset_name)
    uploaded_original_documents_set = document_deduplication(uploaded_original_documents)

    original_document_set = set(original_document_sources)
    deleted_original_document_set = set(deleted_original_document_sources)
    uploaded_original_document_set = set(uploaded_original_document_sources)
    union_deleted_original_document_set = (
        deleted_original_document_set | uploaded_original_document_set
    ) & original_document_set
    updated_original_document_set = union_deleted_original_document_set & uploaded_original_document_set
    incremented_original_document_set = uploaded_original_document_set - union_deleted_original_document_set
    return (
        union_deleted_original_document_set,
        updated_original_document_set,
        incremented_original_document_set,
        uploaded_original_documents_set,
    )


@op(retry_policy=RetryPolicy(max_retries=3))
def delete_documents_in_vector_store(
    context: OpExecutionContext, ins: Tuple[Set[str], Set[str], Set[str], List[OriginalDocument]]
) -> Tuple[Set[str], Set[str], Set[str], List[OriginalDocument]]:
    (
        union_deleted_original_document_set,
        updated_original_document_set,
        incremented_original_document_set,
        uploaded_original_documents,
    ) = ins
    # 删除ES上的内容
    deleted_original_document_dict = collections.defaultdict(list)
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
        deleted_term = session.exec(
            select(VectorStore.name, OriginalDocumentEntity.source)
            .join(VectorStore, VectorStore.id == OriginalDocumentEntity.vs_id)
            .join(KnowledgeBaseAsset, KnowledgeBaseAsset.id == VectorStore.kba_id)
            .join(KnowledgeBase, KnowledgeBase.id == KnowledgeBaseAsset.kb_id)
            .where(
                KnowledgeBase.sn == knowledge_base_serial_number,
                KnowledgeBaseAsset.name == knowledge_base_asset_name,
                col(OriginalDocumentEntity.source).in_(union_deleted_original_document_set),
            )
        ).all()
        for vector_store_name, deleted_original_document_source in deleted_term:
            deleted_original_document_dict[vector_store_name].append(deleted_original_document_source)
        kb_allow_synchronous_update = knowledge_base_asset.knowledge_base.allow_synchronous_update
    get_vector_store_manager().delete_by_document_sources(deleted_original_document_dict)
    if kb_allow_synchronous_update and union_deleted_original_document_set:
        original_documents = get_original_documents_by_source_list(
            session, knowledge_base_serial_number, knowledge_base_asset_name, union_deleted_original_document_set
        )
        combined_dict = {}
        for original_document in original_documents:
            if not original_document.ipd_rag_document_list:
                continue
            source = original_document.source
            combined_dict[source] = original_document.ipd_rag_document_list
        documents_info_dir = get_documents_info_root_dir(knowledge_base_serial_number, knowledge_base_asset_name)
        delete_dir = documents_info_dir / DELETED_DOCUMENTS_INFO_DIR
        delete_dir.mkdir(parents=True, exist_ok=True)
        file_name = "delete_document_info_" + str(uuid.uuid4()) + ".json"
        output_file = delete_dir / file_name
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_dict, f, indent=4, ensure_ascii=False)
    return (
        union_deleted_original_document_set,
        updated_original_document_set,
        incremented_original_document_set,
        uploaded_original_documents,
    )


@op(retry_policy=RetryPolicy(max_retries=3))
def delete_database_original_documents(
    context: OpExecutionContext, ins: Tuple[Set[str], Set[str], Set[str], List[OriginalDocument]]
) -> List[OriginalDocument]:
    (
        union_deleted_original_document_set,
        updated_original_document_set,
        incremented_original_document_set,
        uploaded_original_documents,
    ) = ins

    if not union_deleted_original_document_set:
        return uploaded_original_documents

    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)

    # 记录文档更新详情
    insert_update_record(knowledge_base_serial_number, knowledge_base_asset_name,
                         uuid.UUID(context.op_config["job_id"]), ins)

    sources_list = list(union_deleted_original_document_set)

    with Session(engine) as session:
        # 1. 查询资产id
        kba_id = session.exec(
            select(KnowledgeBaseAsset.id).join(KnowledgeBase, KnowledgeBase.id == KnowledgeBaseAsset.kb_id).where(
                KnowledgeBase.sn == knowledge_base_serial_number,
                KnowledgeBaseAsset.name == knowledge_base_asset_name
            )
            .limit(1)
        ).first()

        if not kba_id:
            return uploaded_original_documents

        _batch_delete_document_relation(kba_id, session, sources_list)

    return uploaded_original_documents


def _batch_delete_document_relation(kba_id, session, sources_list):
    """
    删除文档前批量删除关联数据
    :param kba_id: 资产id
    :param session: 会话
    :param sources_list: 文档列表
    :return: 无
    """
    BATCH_SIZE = 500  # 每批处理500个source
    # 分批处理
    for i in range(0, len(sources_list), BATCH_SIZE):
        batch_sources = sources_list[i: i + BATCH_SIZE]

        # 使用子查询直接删除，避免加载ID
        doc_ids_subquery = (
            select(OriginalDocumentEntity.id).join(VectorStore, VectorStore.id == OriginalDocumentEntity.vs_id)
            .where(
                VectorStore.kba_id == kba_id,
                OriginalDocumentEntity.source.in_(batch_sources)
            )
        )

        # 删除文档相关的告警
        session.exec(
            delete(DocumentDefect).where(
                or_(
                    DocumentDefect.doc_id.in_(doc_ids_subquery),
                    DocumentDefect.compare_doc_id.in_(doc_ids_subquery)
                )
            )
        )

        # 删除资产关联的request_document_hit
        session.exec(
            delete(RequestDocumentHit)
            .where(
                RequestDocumentHit.doc_id.in_(doc_ids_subquery)
            )
        )

        # 删除文档
        session.exec(
            delete(OriginalDocumentEntity).where(
                OriginalDocumentEntity.id.in_(doc_ids_subquery)
            )
        )

        session.commit()


@op(out=DynamicOut(), retry_policy=RetryPolicy(max_retries=3))
def fetch_updated_original_documents(
    context: OpExecutionContext, updated_original_documents: List[OriginalDocument]
) -> List[OriginalDocument]:
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base_asset = get_knowledge_base_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
        if not knowledge_base_asset.vector_stores:
            knowledge_base_asset.vector_stores.append(
                VectorStore(name=uuid.uuid4().hex)
            )
        asset_documents = [
            original_documents
            for vector_store in knowledge_base_asset.vector_stores
            for original_documents in vector_store.original_documents
        ]
        max_documents_num = get_max_documents_num(
            VectorizationConfig(**knowledge_base_asset.vectorization_config).max_documents_num,
            knowledge_base_asset,
            session
        )

        can_add_documents_num = None
        if knowledge_base_asset.asset_type in AssetType.types_require_limit_docs_num():
            can_add_documents_num = max(0, max_documents_num - len(asset_documents))
        processed_documents = []
        for idx, document in enumerate(updated_original_documents):
            original_document_entity = OriginalDocumentEntity(
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

            if can_add_documents_num is None or idx < can_add_documents_num:
                original_document_entity.document_load_status = DocumentLoadStatus.LOADING
                processed_documents.append(document)
            else:
                original_document_entity.document_load_status = DocumentLoadStatus.QUOTA_EXCEEDED
                original_document_entity.error_info = DOC_IMPORT_FAIL_NUM_LIMIT.format(max_documents_num)
            knowledge_base_asset.vector_stores[-1].original_documents.append(original_document_entity)
        session.add(knowledge_base_asset)
        session.commit()
        refresh_kba_id_count_and_max_create_to_redis(session, knowledge_base_asset.kb_id, knowledge_base_asset.id)
    for idx, chunked_update_original_documents in enumerate(
        chunked(processed_documents, VECTORIZATION_CHUNK_SIZE)
    ):
        yield DynamicOutput(chunked_update_original_documents, mapping_key=str(idx))


@op(ins={"no_input": In(Nothing)}, retry_policy=RetryPolicy(max_retries=3))
def change_update_vectorization_job_status_to_success(context: OpExecutionContext):
    with Session(engine) as session:
        job = session.exec(select(AutoJobInstances).where(AutoJobInstances.id == context.op_config["job_id"])).one()
        if job.knowledge_base_asset.automated_job_schedule:
            job.knowledge_base_asset.automated_job_schedule.last_updated_at = now_with_time_zone()
        change_vectorization_job_status(session, job, JobStatus.SUCCESS)
        if (
            job.knowledge_base_asset.knowledge_base.allow_synchronous_update
            and job.job_type == VectorizationJobType.INCREMENTAL
        ):
            job_type = VectorizationJobType.INCREMENTAL_IN_IPD_RAG
            ipd_rag_job = AutoJobInstances(status=JobStatus.PENDING, job_type=job_type)
            job.knowledge_base_asset.auto_job_instances.append(ipd_rag_job)
            session.add(job)
            session.commit()


@graph_asset(partitions_def=knowledge_base_asset_partitions_def)
def update_knowledge_base_asset():
    # 文档向量化流程
    documents = fetch_updated_original_documents(
        delete_database_original_documents(
            delete_documents_in_vector_store(
                fetch_updated_original_document_set(change_vectorization_job_status_to_started())
            )
        )
    ).map(load_original_documents)
    # 执行敏感信息检查，对包含敏感信息的文本进行脱敏处理
    documents = documents.map(sensitive_words_detect)
    embeddings = documents.map(embedding_documents)
    vectorize_result = change_update_vectorization_job_status_to_success(
        update_knowledge_base_and_asset_updated_at(
            delete_original_documents(
                no_op_fan_in(embeddings.map(store_chunks_to_vector_db).map(update_document_status).collect())
            )
        )
    )

    # 立即执行的检测任务
    url_valid_detect = documents.map(document_url_detect).collect()
    punctuation_ratio_detect = documents.map(punctuation_detect).collect()
    deep_document_structure_detect = documents.map(document_structure_detect).collect()
    semantic_integrity_detect = embeddings.map(document_semantic_integrity_detect).collect()
    delayed_detection_job_info = documents.map(save_delayed_detection_document_data).collect()

    return dummy_immediate_consumer(
        vectorize_result,
        url_valid_detect,
        punctuation_ratio_detect,
        deep_document_structure_detect,
        semantic_integrity_detect,
        delayed_detection_job_info,
    )
