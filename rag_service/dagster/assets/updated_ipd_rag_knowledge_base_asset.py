from typing import Any, Dict

from dagster import DynamicOut, DynamicOutput, In, Nothing, OpExecutionContext, RetryPolicy, graph_asset, op
from more_itertools import chunked
from sqlmodel import Session

from rag_service.constants import DELETED_DOCUMENTS_INFO_DIR, SPLIT_SIZE, DATAOPS_OPERATION_DELETE
from rag_service.dagster.dagster_common_op import (
    add_asset_documents_in_ipd_rag,
    change_vectorization_job_status_to_started,
    change_vectorization_job_status_to_success,
    create_document_entry,
    delete_documents_info,
    get_added_documents_index,
    get_added_documents_info,
    no_op_fan_in,
    parse_documents_info,
    save_document_list_to_database,
)
from rag_service.dagster.partitions.knowledge_base_asset_partition import knowledge_base_asset_partitions_def
from rag_service.database import engine
from rag_service.utils.dagster_util import get_documents_info_root_dir, parse_asset_partition_key
from rag_service.utils.db_util import (
    get_knowledge_base_by_kb_sn,
)
from rag_service.utils.ipd_rag_util import send_data_to_dataops


@op(ins={"no_input": In(Nothing)}, out=DynamicOut(), retry_policy=RetryPolicy(max_retries=3))
def get_deleted_documents_info(context: OpExecutionContext) -> Dict[str, Any]:
    knowledge_base_serial_number, knowledge_base_asset_name = parse_asset_partition_key(context.partition_key)
    documents_info_dir = get_documents_info_root_dir(knowledge_base_serial_number, knowledge_base_asset_name)
    deleted_documents_info_dir = documents_info_dir / DELETED_DOCUMENTS_INFO_DIR
    # 检查目录是否存在
    if not deleted_documents_info_dir.exists():
        yield DynamicOutput(value={}, mapping_key="empty")
    else:
        deleted_documents_info_list = parse_documents_info(deleted_documents_info_dir)
        for idx, deleted_documents_info in enumerate(deleted_documents_info_list):
            yield DynamicOutput(value=deleted_documents_info, mapping_key=str(idx))


@op(retry_policy=RetryPolicy(max_retries=3))
def delete_asset_documents_in_ipd_rag(context: OpExecutionContext, deleted_documents_info: Dict[str, Any]):
    if not deleted_documents_info:
        return
    kb_sn, asset_name = parse_asset_partition_key(context.partition_key)
    with Session(engine) as session:
        knowledge_base = get_knowledge_base_by_kb_sn(session, kb_sn)
    ipd_rag_knowledge_base_set_sn = knowledge_base.ipd_rag_kb_sn
    document_entry_list = []
    for source in deleted_documents_info:
        for document in deleted_documents_info[source]:
            document_entry = create_document_entry(document["document_id"], source, document["document_name"],
                                                   DATAOPS_OPERATION_DELETE)
            document_entry_list.append(document_entry)
    for chunked_document_entry_list in chunked(document_entry_list, SPLIT_SIZE):
        send_data_to_dataops(ipd_rag_knowledge_base_set_sn, kb_sn, chunked_document_entry_list)


@graph_asset(partitions_def=knowledge_base_asset_partitions_def)
def update_ipd_rag_knowledge_base_asset():
    return change_vectorization_job_status_to_success(
        delete_documents_info(
            no_op_fan_in(
                get_added_documents_info(
                    no_op_fan_in(
                        get_deleted_documents_info(change_vectorization_job_status_to_started())
                        .map(delete_asset_documents_in_ipd_rag)
                        .collect()
                    )
                )
                .map(get_added_documents_index)
                .map(add_asset_documents_in_ipd_rag)
                .map(save_document_list_to_database)
                .collect()
            )
        )
    )