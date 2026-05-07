from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks
from langchain.schema import Document

from rag_service.models.api.models import RetrievedDocument
from rag_service.models.enums import Analyzer, EmbeddingModel, QueryStrategy
from rag_service.telemetry.collect_answer_info import collect_time_use_info


class BaseVectorStore(ABC):
    @abstractmethod
    def bulk_insert(self, data: Any, **kwargs: Any) -> None:
        """Bulk insert data to the vector store.
        @param data: Data to add to the vector store.
        @param kwargs: Vector store specific parameters.
        """
        ...

    @abstractmethod
    def bulk_delete(self, **kwargs: Any) -> None:
        """Bulk delete data from the vector store.
        @param kwargs: Vector store specific parameters.
        """
        ...

    @abstractmethod
    def drop(self, **kwargs: Any) -> None:
        """Drop the vector store.
        @param kwargs: Vector store specific parameters.
        """
        ...

    @abstractmethod
    def search(self, query: str, **kwargs: Any) -> List[Any]:
        """Return documents most similar to query.
        @param query: The input query.
        @param kwargs: Vector store specific parameters.
        @return: Retrieved documents most similar to query.
        """
        ...


class BaseVectorStoreManager(ABC):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.retrieve = collect_time_use_info(cls.retrieve)

    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        vector_store: Optional[str],
        analyzer: Optional[Analyzer],
        synonyms: Optional[List[str]],
        stopwords: Optional[List[str]],
    ) -> None:
        """@param documents: Documents to add to the vector store.
        @param embeddings: Corresponding embeddings of the documents to add to the vector store.
        @param vector_store: Name of the vector store.
        @param analyzer: knowledge base serial number
        @param synonyms: Synonyms
        @param stopwords: Stopwords
        """
        ...

    @abstractmethod
    def delete_by_document_sources(self, vector_store_to_sources: Dict[str, List[str]]) -> None:
        """Delete documents in vector store(s) by sources.
        @param vector_store_to_sources: Mapping from vector store to sources.
        """
        ...

    @abstractmethod
    def delete_vector_stores(self, vector_stores: List[str]):
        """Delete vector stores.
        @param vector_stores: Vector stores to delete.
        """
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int,
        embedding_model_to_vector_stores: Dict[EmbeddingModel, List[str]],
        document_score_threshold: float = 0.0,
        collect_info: bool = True,
        analyzer: Analyzer = Analyzer.IK_ANALYZER,
        query_strategy: QueryStrategy = QueryStrategy.HYBRID_QUERY,
        request_id: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> List[RetrievedDocument]:
        """Retrieve documents most similar to query from vector store(s).
        @param query: The input query.
        @param k: Number of documents to return.
        @param embedding_model_to_vector_stores: Mapping from embedding model to vector stores.
        @param document_score_threshold: Filter document by score threshold.
        @param collect_info: If collect retrieve time info.
        @param analyzer: Index search analyzer
        @param query_strategy: hybrid query / full text query / vector query / text_vector_hybrid query
        @return: Retrieved documents most similar to query.
        """
        ...