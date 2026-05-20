from typing import Any, Dict

from dagster import DynamicOut, DynamicOutput, In, Nothing, OpExecutionContext, RetryPolicy, graph_asset, op
from sqlmodel import Session

from rag_service.constants import SPLIT_SIZE
from rag_service.dagster.dagster_common_op import (
    add_asset_documents_in_ipd_rag,
    change_vectorization_job_status_to_started,
    change_vectorization_job_status_to_success,
    delete_documents_info,
    no_op_fan_in,
    save_document_list_to_database,
)
from rag_service.dagster.partitions.knowledge_base_asset_partition import knowledge_base_asset_partitions_def
from rag_service.database import engine
from rag_service.utils.dagster_util import parse_asset_partition_key
from rag_service.utils.db_util import get_kba_document_index_collection_by_asset


@op(ins={"no_input": In(Nothing)}, out=DynamicOut(dagster_type=Dict[str, Any]), retry_policy=RetryPolicy(max_retries=3))
def get_existed_documents_index(context: OpExecutionContext) -> DynamicOut[Dict[str, Any]]:
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        results = get_kba_document_index_collection_by_asset(
            session, knowledge_base_serial_number, knowledge_base_asset_name
        )
    documents_index_dict = {}
    for row in results:
        documents_index_dict[row.source] = {"display_name": row.display_name, "index": row.name, "doc_id": row.id}

    items = list(documents_index_dict.items())
    for idx in range(0, len(items), SPLIT_SIZE):
        chunk_items = items[idx: idx + SPLIT_SIZE]
        chunk_dict = dict(chunk_items)
        yield DynamicOutput(value=chunk_dict, mapping_key=str(idx))


@graph_asset(partitions_def=knowledge_base_asset_partitions_def)
def migrate_knowledge_base_asset_to_ipd_rag():
    return change_vectorization_job_status_to_success(
        delete_documents_info(
            no_op_fan_in(
                get_existed_documents_index(change_vectorization_job_status_to_started())
                .map(add_asset_documents_in_ipd_rag)
                .map(save_document_list_to_database)
                .collect()
            )
        )
    )