from typing import TypeVar

from rag_service.config import VECTOR_STORE_TYPE
from rag_service.models.enums import VectorStoreType
from rag_service.vectorstore.base import BaseVectorStoreManager

VectorStoreManager = TypeVar("VectorStoreManager", bound=BaseVectorStoreManager)


def get_vector_store_manager() -> VectorStoreManager:
    vector_store_type: VectorStoreType = VectorStoreType[VECTOR_STORE_TYPE]
    if vector_store_type == VectorStoreType.ELASTICSEARCH:
        from rag_service.vectorstore.elasticsearch.elasticsearch import ElasticsearchManager

        return ElasticsearchManager()
    raise Exception(f"Invalid vector store type: <{vector_store_type}>")