from dagster import graph_asset

from rag_service.dagster.dagster_common_op import (
    add_asset_documents_in_ipd_rag,
    change_vectorization_job_status_to_started,
    change_vectorization_job_status_to_success,
    delete_documents_info,
    get_added_documents_index,
    get_added_documents_info,
    no_op_fan_in,
    save_document_list_to_database,
)
from rag_service.dagster.partitions.knowledge_base_asset_partition import knowledge_base_asset_partitions_def


@graph_asset(partitions_def=knowledge_base_asset_partitions_def)
def init_ipd_rag_knowledge_base_asset():
    return change_vectorization_job_status_to_success(
        delete_documents_info(
            no_op_fan_in(
                get_added_documents_info(change_vectorization_job_status_to_started())
                .map(get_added_documents_index)
                .map(add_asset_documents_in_ipd_rag)
                .map(save_document_list_to_database)
                .collect()
            )
        )
    )
