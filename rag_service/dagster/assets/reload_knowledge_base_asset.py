from dagster import In, Nothing, OpExecutionContext, graph_asset, op
from sqlmodel import Session

from rag_service.dagster.dagster_common_op import (
    change_vectorization_job_status_to_started,
    change_vectorization_job_status_to_success,
    delete_knowledge_base_asset_resources,
    delete_knowledge_base_asset_vector_store,
    delete_original_documents,
    document_semantic_integrity_detect,
    document_structure_detect,
    document_url_detect,
    dummy_immediate_consumer,
    embedding_documents,
    fetch_raw_document,
    no_op_fan_in,
    parse_original_documents,
    punctuation_detect,
    save_delayed_detection_document_data,
    save_document_metadata,
    save_parsed_documents,
    split_parsed_documents,
    store_chunks_to_vector_db,
    update_document_status,
    update_knowledge_base_and_asset_updated_at,
    sensitive_words_detect,
)
from rag_service.dagster.partitions.knowledge_base_asset_partition import knowledge_base_asset_partitions_def
from rag_service.database import engine
from rag_service.rag_app.dao.vector_store_dao import delete_vector_by_kba_id_and_vs_ids
from rag_service.utils.dagster_util import parse_asset_partition_key
from rag_service.utils.db_util import get_knowledge_base_asset


@op(ins={"no_input": In(Nothing)})
def delete_kba_database_vector_store(context: OpExecutionContext):
    kb_sn, kba_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        kba = get_knowledge_base_asset(session, kb_sn, kba_name)
        if not kba:
            return
        vs_ids = [vs.id for vs in kba.vector_stores]
        delete_vector_by_kba_id_and_vs_ids(session, kba.id, vs_ids)
        session.commit()


@graph_asset(partitions_def=knowledge_base_asset_partitions_def)
def reload_knowledge_base_asset():
    # 文档向量化流程
    document_batches = save_document_metadata(
        fetch_raw_document(
            delete_kba_database_vector_store(
                delete_knowledge_base_asset_resources(
                    delete_knowledge_base_asset_vector_store(change_vectorization_job_status_to_started())
                )
            )
        )
    )
    parsed_documents = document_batches.map(parse_original_documents)
    parsed_documents = parsed_documents.map(save_parsed_documents)
    documents = parsed_documents.map(split_parsed_documents)
    # 执行敏感信息检查，对包含敏感信息的文本进行脱敏处理
    documents = documents.map(sensitive_words_detect)
    embeddings = documents.map(embedding_documents)
    vectorize_result = change_vectorization_job_status_to_success(
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
