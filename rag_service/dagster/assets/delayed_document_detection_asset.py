from dagster import graph_asset

from rag_service.dagster.dagster_common_op import (
    document_consistency_detect,
    document_logic_detect,
    document_readability_detect,
    document_semantic_coherence_detect,
    document_similarity_detect,
    dummy_delay_consumer,
    load_document_pairs,
)
from rag_service.dagster.partitions.knowledge_base_asset_partition import knowledge_base_asset_partitions_def


@graph_asset(
    partitions_def=knowledge_base_asset_partitions_def,
)
def delayed_detection_asset():
    # 加载保存的文档（返回 DynamicOutput）
    documents = load_document_pairs()

    # 使用LLM的耗时检测任务
    similar_document_detect = documents.map(document_similarity_detect).collect()
    non_consistence_document_detect = documents.map(document_consistency_detect).collect()
    semantic_coherence_detect = documents.map(document_semantic_coherence_detect).collect()
    readability_detect = documents.map(document_readability_detect).collect()
    logic_detect = documents.map(document_logic_detect).collect()

    # 可以将结果更新回数据库
    return dummy_delay_consumer(
        similar_document_detect,
        non_consistence_document_detect,
        semantic_coherence_detect,
        readability_detect,
        logic_detect,
    )