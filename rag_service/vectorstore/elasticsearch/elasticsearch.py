import copy
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import validators
from elastic_transport import ObjectApiResponse
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import scan, streaming_bulk
from fastapi import BackgroundTasks
from langchain.schema import Document
from more_itertools import chunked

from rag_service.config import (
    ES_CONNECT_URL,
    RETRIEVE_AMPLIFICATION_FACTOR,
    SEARCH_TIMEOUT,
    VECTOR_STORE_GET_REQUEST_MAX_INDICES,
)
from rag_service.constants import DBOX_URL_PREFIX, DEFAULT_STOPWORDS, ES_DELETE_DOC_BATCH_SIZE, RESPONSE_STATUS, \
    ES_BATCH_CHUNK_SIZE, ES_MAX_RETRIES
from rag_service.dagster.jobs.copy_knowledge_base_asset_job import get_target_doc_uri
from rag_service.exceptions import VectorStoreIndexNotExistException, VectorStoreReindexTimeOut
from rag_service.logger import Module, get_logger
from rag_service.logger.wrapper_trace import safe_trace
from rag_service.models.api.models import RetrievedDocument, RetrievedDocumentMetadata
from rag_service.models.enums import Analyzer, AssetType, EmbeddingModel, QueryStrategy
from rag_service.models.generic.models import VectorStoreElasticSearchQueryInfo, KbaCopyConfig, EsDocumentChunk, \
    EsDocumentEmbeddingResult
from rag_service.utils.serdes import deserialize, serialize
from rag_service.vectorize.embedding import embedding
from rag_service.vectorstore.base import BaseVectorStore, BaseVectorStoreManager
from rag_service.vectorstore.parallel_retrieval import query_strategy_requires_embedding, run_ordered_parallel_tasks
from rag_service.vectorstore.elasticsearch.es_model import (
    EsIndexMapping,
    EsQueryResult,
    IntentQueryResult,
    es_ik_index_mapping,
    es_index_mapping,
    es_index_match_query,
    es_intent_index_mapping,
    es_knn_query,
    es_match_query,
    es_phrase_query,
    es_query_by_source,
    es_query_by_sources,
    es_scroll_query,
    es_text_match_query,
    es_text_phrase_query,
    es_update_doc_metadata,
    RetrieveMetadata,
    es_match_all,
)

logger = get_logger(module=Module.VECTORIZATION)


class ElasticsearchStore(BaseVectorStore):
    _client: Elasticsearch

    def __init__(self, es_url: str, es_user: Optional[str] = None, es_password: Optional[str] = None):
        self._client = self._connect_to_elasticsearch(es_url, es_user, es_password)

    @classmethod
    def _connect_to_elasticsearch(
        cls, es_url: str, username: Optional[str] = None, password: Optional[str] = None
    ) -> Elasticsearch:
        connection_params: Dict[str, Any] = {"hosts": [es_url], "timeout": SEARCH_TIMEOUT}
        if username and password:
            connection_params["basic_auth"] = (username, password)
        return Elasticsearch(**connection_params)

    def _create_index_if_not_exists(
        self, index: str, dim: int, analyzer: Analyzer, synonyms: List[str], stopwords: List[str]
    ):
        if self._client.indices.exists(index=index):
            logger.error("Index %s already exists. Skipping creation.", index)
            return

        if analyzer == Analyzer.STANDARD_ANALYZER:
            mappings = es_index_mapping(dim)
        else:
            stopwords = stopwords.extend(DEFAULT_STOPWORDS) if stopwords else DEFAULT_STOPWORDS
            mappings = es_ik_index_mapping(dim, synonyms, stopwords)
        self._client.indices.create(index=index, body=mappings)
        logger.info("Create index %s successfully.", index)

    def bulk_insert(self, data: List[Dict], **kwargs: Any) -> None:
        index: str = kwargs["index"]
        dim: int = kwargs["dim"]
        analyzer: Analyzer = kwargs["analyzer"]
        synonyms: List[str] = kwargs["synonyms"]
        stopwords: List[str] = kwargs["stopwords"]
        self._create_index_if_not_exists(index, dim, analyzer, synonyms, stopwords)
        helpers.bulk(self._client, data, index=index)

    def bulk_delete(self, **kwargs: Any) -> None:
        index: str = kwargs["index"]
        query = kwargs["query"]
        if not self._client.indices.exists(index=index):
            logger.error("Index %s not exists. Skipping deletion.", index)
            return

        task_id = self._client.delete_by_query(index=index, body=query, wait_for_completion=False).body["task"]
        number_of_calls = 2160  # 3小时每隔5秒调用2160次
        while not self._client.tasks.get(task_id=task_id).body["completed"] and number_of_calls:
            time.sleep(5)
            number_of_calls -= 1

    def drop(self, **kwargs: Any):
        index = kwargs["index"]
        if not self._client.indices.exists(index=index):
            logger.error("Index %s not exists. Skipping drop.", index)
            return

        self._client.indices.delete(index=index)
        logger.info("Delete index %s successfully.", index)

    @safe_trace(log_args=True, include_args=['query'])
    def search(self, query: str, **kwargs: Any) -> List[EsQueryResult]:
        embedding_vector: List[float] = kwargs["embedding"]
        k: int = kwargs["k"]
        vector_stores_search_info: VectorStoreElasticSearchQueryInfo = kwargs["vector_stores_search_info"]
        document_score_threshold: float = kwargs["document_score_threshold"]
        query_strategy: QueryStrategy = kwargs["query_strategy"]

        if not vector_stores_search_info.vs_indexes:
            return []

        if query_strategy == QueryStrategy.TEXT_VECTOR_HYBRID_QUERY:
            ret = self._text_vector_hybrid_search(
                query, embedding_vector, RETRIEVE_AMPLIFICATION_FACTOR * k, vector_stores_search_info
            )
        else:
            ret = self._query_search(
                query,
                embedding_vector,
                RETRIEVE_AMPLIFICATION_FACTOR * k,
                vector_stores_search_info,
                query_strategy,
                document_score_threshold,
            )
        results = self.remove_duplicates(ret)
        results.sort(key=lambda _: _.score, reverse=True)
        return results

    @classmethod
    def _get_query_keywords(cls, query: str):
        query = query.lower()
        keywords = re.findall(r"[a-zA-Z0-9]+(?:[-._/][a-zA-Z0-9]+)*", query)

        # 关键词去重并保持原先相对顺序
        seen = set()
        unique_words = []
        for word in keywords:
            if word not in DEFAULT_STOPWORDS and word not in seen:
                seen.add(word)
                unique_words.append(word)
        return " ".join(unique_words)

    @classmethod
    def _get_query(
        cls,
        query: str,
        embedding_vector: List[float],
        k: int,
        query_strategy: QueryStrategy,
        vs_search_info: VectorStoreElasticSearchQueryInfo,
    ):
        return cls._get_standard_strategy_query(query_strategy, query, embedding_vector, k, vs_search_info)

    @classmethod
    def _get_text_query(cls, query: str, vs_search_info: VectorStoreElasticSearchQueryInfo):
        keywords = cls._get_query_keywords(query)
        return (
            es_text_phrase_query(query, keywords, vs_search_info)
            if keywords
            else es_text_match_query(query, vs_search_info)
        )

    @classmethod
    def _get_knn_query(cls, embedding_vector: List[float], vs_search_info: VectorStoreElasticSearchQueryInfo, k: int):
        return es_knn_query(embedding_vector, vs_search_info, k)

    @classmethod
    def _get_ik_strategy_query(
        cls,
        query_strategy: QueryStrategy,
        query: str,
        embedding_vector: List[float],
        k: int,
        vs_search_info: VectorStoreElasticSearchQueryInfo,
    ) -> Dict:
        if query_strategy == QueryStrategy.HYBRID_QUERY:
            return es_match_query(query, embedding_vector, k, vs_search_info)
        if query_strategy == QueryStrategy.FULL_TEXT_QUERY:
            return es_text_match_query(query, vs_search_info)
        return es_knn_query(embedding_vector, vs_search_info, k)

    @classmethod
    def _get_standard_strategy_query(
        cls,
        query_strategy: QueryStrategy,
        query: str,
        embedding_vector: List[float],
        k: int,
        vs_search_info: VectorStoreElasticSearchQueryInfo,
    ) -> Dict:
        keywords = cls._get_query_keywords(query)
        if query_strategy == QueryStrategy.HYBRID_QUERY:
            return (
                es_phrase_query(query, keywords, embedding_vector, k, vs_search_info)
                if keywords
                else es_match_query(query, embedding_vector, k, vs_search_info)
            )
        if query_strategy == QueryStrategy.FULL_TEXT_QUERY:
            return (
                es_text_phrase_query(query, keywords, vs_search_info)
                if keywords
                else es_text_match_query(query, vs_search_info)
            )
        return es_knn_query(embedding_vector, vs_search_info, k)

    @classmethod
    def _parse_msearch_response(cls, ret: ObjectApiResponse) -> List[EsQueryResult]:
        res = []
        responses = ret.body.get("responses", None)
        if not responses:
            return res
        for item in responses:
            if item["status"] != RESPONSE_STATUS:
                continue
            for doc in item["hits"]["hits"]:
                if not doc:
                    continue
                res.append(
                    EsQueryResult(
                        score=doc["_score"],
                        index=doc["_index"],
                        id=doc["_id"],
                        uri=doc["_source"]["uri"],
                        general_text=doc["_source"]["general_text"],
                        source=doc["_source"]["source"],
                        mtime=doc["_source"]["mtime"],
                        extended_metadata=doc["_source"]["extended_metadata"],
                        retrieve_metadata=doc["_source"].get("retrieve_metadata", None)
                    )
                )
        return res

    def vector_store_reindex(self, index: str, synonyms: List[str], stopwords: List[str]) -> str:
        if not self._client.indices.exists(index=index):
            logger.error("Index %s not exists. Skipping reindex.", index)
            raise VectorStoreIndexNotExistException("Vector store name %s not exist.", index)
        # 创建新索引
        new_index = uuid.uuid4().hex
        while self._client.indices.exists(index=new_index):
            new_index = uuid.uuid4().hex

        # 获取向量维度
        dim = self._client.indices.get_mapping(index=index).body[index]["mappings"]["properties"][
            "general_text_vector"
        ]["dims"]

        # 创建mapping
        new_mappings = es_ik_index_mapping(dim, synonyms, stopwords)
        self._client.indices.create(index=new_index, body=new_mappings)
        source_map, dest_map = {"index": index}, {"index": new_index}
        start_time = time.time()
        task_id = self._client.reindex(dest=dest_map, source=source_map, wait_for_completion=False).body["task"]
        number_of_calls = 2160  # 3小时每隔5秒调用2160次
        while not self._client.tasks.get(task_id=task_id).body["completed"] and number_of_calls:
            time.sleep(5)
            number_of_calls -= 1
        if not number_of_calls:
            raise VectorStoreReindexTimeOut("Reindex from %s to %s time out.", index, new_index)
        end_time = time.time()
        logger.info(
            "Old index %s reindex to new index %s completed, time use %s seconds",
            index,
            new_index,
            end_time - start_time,
        )
        self._client.indices.delete(index=index)
        return new_index

    def _generate_actions(self, kba_copy_config: KbaCopyConfig, docs, dest_index: str, doc_id_map: Dict, download_key_map: Dict):
        for hit in docs:
            doc = hit['_source']
            modified_doc = self._get_copy_doc_extend_metadata(doc, kba_copy_config, doc_id_map, download_key_map)

            yield {
                '_op_type': 'index',
                '_index': dest_index,
                '_id': hit['_id'],
                '_source': modified_doc
            }

    def _create_dest_index(self, source_index: str, dest_index: str):
        try:
            # 检查目标索引是否已存在
            if self._client.indices.exists(index=dest_index):
                logger.warning(f"目标索引已存在: {dest_index}, 将追加数据")
                return True

            # 获取向量维度
            source_mapping = self._client.indices.get_mapping(index=source_index).body
            dim = source_mapping[source_index]["mappings"]["properties"]["general_text_vector"]["dims"]

            # 创建mapping
            new_mappings = es_ik_index_mapping(dim, [], DEFAULT_STOPWORDS)

            # 创建索引
            self._client.indices.create(index=dest_index, body=new_mappings)
            logger.info(f"成功创建目标索引: {dest_index}, 向量维度: {dim}")

            return True

        except KeyError as e:
            logger.error(f"获取向量维度失败,字段不存在: {e}")
            return False
        except Exception as e:
            logger.error(f"创建目标索引失败: {str(e)}", exc_info=True)
            return False

    def _bulk_write_documents(self, kba_copy_config: KbaCopyConfig, docs, dest_index: str, doc_id_map: Dict, download_key_map: Dict):
        success_count, error_count = 0, 0
        try:
            for ok, result in streaming_bulk(
                self._client,
                self._generate_actions(kba_copy_config, docs, dest_index, doc_id_map, download_key_map),
                chunk_size=ES_BATCH_CHUNK_SIZE,  # 每批写入2000条
                max_retries=ES_MAX_RETRIES,  # 失败重试3次
                raise_on_error=False,  # 不因单条失败而中断
                request_timeout=SEARCH_TIMEOUT
            ):
                if ok:
                    success_count += 1
                else:
                    error_count += 1
            # 最终统计
            if error_count > 0:
                logger.error(f"写入完成,共 {error_count} 条失败")
            return success_count, error_count
        except Exception as e:
            logger.error(f"批量写入过程发生异常: {str(e)}", exc_info=True)
            return success_count, error_count

    def copy_index(self, kba_copy_config: KbaCopyConfig, source_index: str, dest_index: str, doc_id_map: Dict, download_key_map: Dict):
        """
        复制ES索引并修改字段

        Args:
            kba_copy_config: 知识库复制配置
            source_index: 源索引名称
            dest_index: 目标索引名称
        """
        try:
            # 1. 创建目标索引
            if not self._create_dest_index(source_index, dest_index):
                logger.error(f"创建目标索引失败: {dest_index}")
                return

            # 2. 扫描源索引
            logger.info((
                            f"开始扫描源知识库: <{kba_copy_config.source_kb_sn}>， 资产: <{kba_copy_config.source_asset_name}>，源索引{source_index}..."))
            docs = scan(
                self._client,
                index=source_index,
                query=es_match_all(),
                scroll='10m',  # 增加scroll超时
                size=ES_BATCH_CHUNK_SIZE,  # 每批获取2000条
                request_timeout=SEARCH_TIMEOUT
            )

            # 3. 批量写入目标索引
            success_count, error_count = self._bulk_write_documents(kba_copy_config, docs, dest_index, doc_id_map, download_key_map)

            # 4. 刷新目标索引
            self._client.indices.refresh(index=dest_index)

            logger.info(
                f"知识库: <{kba_copy_config.source_kb_sn}> "
                f"资产: <{kba_copy_config.source_asset_name}> 复制完成! "
                f"成功: {success_count}, 失败: {error_count}"
            )
        except Exception as e:
            logger.error(f"索引复制过程发生异常: {str(e)}")

    def _get_copy_doc_extend_metadata(self, doc, kba_copy_config: KbaCopyConfig, doc_id_map: Dict, download_key_map: Dict):
        new_doc = copy.deepcopy(doc)
        metadata = deserialize(doc['extended_metadata'])
        if doc.get("uri", None):
            new_doc["uri"] = get_target_doc_uri(doc["uri"], kba_copy_config.source_kb_sn, kba_copy_config.target_kb_sn)
        metadata['kb_sn'] = kba_copy_config.target_kb_sn
        metadata['asset_name'] = kba_copy_config.target_asset_name
        if metadata.get('doc_id', None):
            metadata['doc_id'] = doc_id_map[metadata['doc_id']]
        if metadata.get('download_key', None):
            metadata['download_key'] = download_key_map[metadata['download_key']]
        new_doc['extended_metadata'] = serialize(metadata)
        return new_doc

    def fetch_es_query_results(self, index: str, query: Dict, k: int):
        search_ret = self._client.search(index=index, ignore_unavailable=True, body=query, size=k)
        return self._parse_search_response(search_ret)

    def remove_duplicates(self, es_query_results: List[EsQueryResult]):
        seen = set()
        ret = []
        for query_result in es_query_results:
            if query_result.general_text not in seen:
                seen.add(query_result.general_text)
                ret.append(query_result)
        return ret

    def count(self, indexes: List[str]) -> int:
        document_count = 0
        for index in chunked(indexes, VECTOR_STORE_GET_REQUEST_MAX_INDICES):
            document_count += (
                self._client.count(index=index, ignore_unavailable=True).body.get("count") if indexes else 0
            )
        return document_count

    def search_documents_by_indices(self, indices: List[str], query: Dict) -> List[EsQueryResult]:
        results = []
        for i in range(0, len(indices), VECTOR_STORE_GET_REQUEST_MAX_INDICES):
            index = ",".join(indices[i : i + VECTOR_STORE_GET_REQUEST_MAX_INDICES])
            ret = self._client.search(index=index, body=copy.deepcopy(query))
            results.extend(self._parse_search_response(ret))
        return results

    def create_intent_index_if_not_exists(self, index: str, dim: int):
        if self._client.indices.exists(index=index):
            logger.error("Index %s already exists. Skipping creation.", index)
            return

        mappings = es_intent_index_mapping(dim)
        self._client.indices.create(index=index, body=mappings)
        logger.info("Create index %s successfully.", index)

    def insert_intent_index(self, data: List[Dict], **kwargs: Any) -> None:
        index: str = kwargs["index"]
        dim: int = kwargs["dim"]
        self.create_intent_index_if_not_exists(index, dim)
        helpers.bulk(self._client, data, index=index)

    def intent_search(self, query: str, **kwargs: Any) -> List[IntentQueryResult]:
        embedding_vector: List[float] = kwargs["embedding"]
        k: int = kwargs["k"]
        vector_stores: List[str] = kwargs["vector_stores"]
        document_score_threshold: float = kwargs["document_score_threshold"]

        query_json = es_index_match_query(query, embedding_vector, k)
        results = []
        for i in range(0, len(vector_stores), VECTOR_STORE_GET_REQUEST_MAX_INDICES):
            index = ",".join(vector_stores[i : i + VECTOR_STORE_GET_REQUEST_MAX_INDICES])
            ret = self._client.search(
                index=index,
                ignore_unavailable=True,
                body=copy.deepcopy(query_json),
                size=k,
                min_score=document_score_threshold,
            )
            results.extend([
                IntentQueryResult(
                    score=item["_score"],
                    intent_name=item["_source"]["intent_name"],
                    intent_value=item["_source"]["intent_value"],
                )
                for item in ret["hits"]["hits"]
                if item
            ])
        results.sort(key=lambda _: _.score, reverse=True)
        return results

    def analyze_text(self, index: str, text: str) -> List[str]:
        if not self._client.indices.exists(index=index):
            logger.error("Index %s not exists. Skipping analyze text.", index)
            return []
        ret = self._client.indices.analyze(index=index, body={"text": text})
        return [token["token"] for token in ret["tokens"]]

    def _text_vector_hybrid_search(
        self,
        query: str,
        embedding_vector: List[float],
        k: int,
        vs_search_info: VectorStoreElasticSearchQueryInfo,
    ):
        text_query_json = self._get_text_query(query, vs_search_info)
        text_query_json.update({"size": k})
        knn_query_json = self._get_knn_query(embedding_vector, vs_search_info, k)
        knn_query_json.update({"size": k})
        body = []
        for i in range(0, len(vs_search_info.vs_indexes), VECTOR_STORE_GET_REQUEST_MAX_INDICES):
            index = ",".join(vs_search_info.vs_indexes[i : i + VECTOR_STORE_GET_REQUEST_MAX_INDICES])
            body.append({"index": index})
            body.append(copy.deepcopy(text_query_json))
            body.append({"index": index})
            body.append(copy.deepcopy(knn_query_json))
        response = self._client.msearch(body=body, ignore_unavailable=True)
        ret = ElasticsearchStore._parse_msearch_response(response)
        return ret

    def _query_search(
        self,
        query: str,
        embedding_vector: List[float],
        k: int,
        vs_search_info: VectorStoreElasticSearchQueryInfo,
        query_strategy: QueryStrategy,
        document_score_threshold: float,
    ):
        query_json = self._get_query(query, embedding_vector, k, query_strategy, vs_search_info)
        results = []
        for i in range(0, len(vs_search_info.vs_indexes), VECTOR_STORE_GET_REQUEST_MAX_INDICES):
            index = ",".join(vs_search_info.vs_indexes[i : i + VECTOR_STORE_GET_REQUEST_MAX_INDICES])
            ret = self._client.search(
                index=index,
                ignore_unavailable=True,
                body=copy.deepcopy(query_json),
                size=k,
                min_score=document_score_threshold,
            )
            results.extend(self._parse_search_response(ret))
        return results

    def _parse_search_response(self, ret: ObjectApiResponse) -> List[EsQueryResult]:
        return [
            EsQueryResult(
                score=item["_score"],
                index=item["_index"],
                id=item["_id"],
                uri=item["_source"]["uri"],
                general_text=item["_source"]["general_text"],
                source=item["_source"]["source"],
                mtime=item["_source"]["mtime"],
                extended_metadata=item["_source"]["extended_metadata"],
                retrieve_metadata=item["_source"].get("retrieve_metadata", None)
            )
            for item in ret["hits"]["hits"]
            if item
        ]

    def _parse_search_count(self, ret: ObjectApiResponse) -> int:
        return ret["count"]

    def scroll_first_search(self, index: str, source: str, batch_size: int, scroll_time: str = "1m"):
        query = es_scroll_query(source, batch_size)
        ret = self._client.search(index=index, ignore_unavailable=True, body=query, scroll=scroll_time)
        scroll_id = ret["_scroll_id"]
        return scroll_id, self._parse_search_response(ret)

    def scroll_iterator_search(self, scroll_id: str, scroll_time: str = "1m"):
        ret = self._client.scroll(scroll_id=scroll_id, scroll=scroll_time)
        scroll_id = ret["_scroll_id"]
        return scroll_id, self._parse_search_response(ret)

    def get_document_data_count(self, index: str, source: str):
        query = es_query_by_source(source)
        ret = self._client.count(index=index, body=query, ignore_unavailable=True)
        return self._parse_search_count(ret)

    def clear_scroll(self, scroll_id: str):
        self._client.clear_scroll(scroll_id=scroll_id)

    def update_doc_metadata(self, source: str, extend_metadata: str, index: str):
        query = es_update_doc_metadata(source, extend_metadata)
        self._client.update_by_query(index=index, body=query, ignore_unavailable=True)

    def fetch_index_docs(self, index: str) -> List[EsDocumentChunk]:
        logger.info(f"开始扫描索引: {index}")
        count = 0

        scan_kwargs = {
            "client": self._client,
            "index": index,
            "query": es_match_all(),
            "size": ES_BATCH_CHUNK_SIZE,
            "scroll": "10m",
        }
        chunks = []

        for doc in scan(**scan_kwargs):
            chunks.append(
                EsDocumentChunk(
                    index=doc["_index"],
                    doc_id=doc["_id"],
                    content=doc["_source"]
                )
            )
            count += 1

            if count % 5000 == 0:
                logger.info(f"索引 {index} 已扫描 {count} 个切片")

        logger.info(f"索引 {index} 共扫描 {count} 个切片")
        return chunks

    def generate_bulk_actions(self, embeddings: List[EsDocumentEmbeddingResult], vector_field: str = "general_text_vector"):
        """执行单批次更新"""
        for item in embeddings:
            yield {
                "_op_type": "update",
                "_index": item.index,
                "_id": item.doc_id,
                "doc": {vector_field: item.embedding}
            }

    def rebuild_index_with_new_dims(self, index_name: str, new_dims: int):
        temp_index = f"{index_name}_temp_{int(time.time())}"

        logger.info(f"开始重建索引 {index_name}，新向量维度: {new_dims}")

        # 1. 获取新索引信息
        mapping = es_ik_index_mapping(new_dims, [], DEFAULT_STOPWORDS)

        # 2. 创建临时索引
        self._client.indices.create(index=temp_index, body=mapping)
        logger.info(f"已创建临时索引: {temp_index}")

        # 3. 迁移数据到临时索引（排除向量字段）
        logger.info("正在迁移数据到临时索引...")
        self._client.reindex(
            body={
                "source": {
                    "index": index_name,
                    "_source": {"excludes": ["general_text_vector"]}
                },
                "dest": {"index": temp_index}
            },
            wait_for_completion=True,
            request_timeout=3600
        )

        # 4.删除源索引
        self._client.indices.delete(index=index_name)
        logger.info(f"已删除原索引: {index_name}")

        # 5. 重建原索引
        self._client.indices.create(
            index=index_name,
            body=mapping
        )
        logger.info(f"已重建索引: {index_name}")

        # 6. 迁移数据回原索引
        logger.info("正在迁移数据回原索引...")
        self._client.reindex(
            body={
                "source": {"index": temp_index},
                "dest": {"index": index_name}
            },
            wait_for_completion=True,
            request_timeout=3600
        )

        # 8. 删除临时索引
        self._client.indices.delete(index=temp_index)
        logger.info(f"已删除临时索引: {temp_index}")

        logger.info(f"索引 {index_name} 重建完成")

    def judge_index_exists(self, index: str) -> bool:
        if self._client.indices.exists(index=index):
            return True
        else:
            return False

    def write_embeddings_streaming(
        self,
        embedding_list: List[EsDocumentEmbeddingResult],
        vector_field: str = "general_text_vector",
        chunk_size: int = 500,
        max_retries: int = 3,
        raise_on_error: bool = False
    ) -> Dict[str, int]:
        """
            使用 streaming_bulk 流式写入 embedding 到 ES
        """
        stats = {"total": 0, "success": 0, "failed": 0}
        indices_seen = set()

        # 生成 bulk actions
        actions = self.generate_bulk_actions(embedding_list, vector_field)

        # 使用 streaming_bulk 流式处理
        for ok, result in streaming_bulk(
            client=self._client,
            actions=actions,
            chunk_size=chunk_size,
            max_retries=max_retries,
            raise_on_error=raise_on_error,
            raise_on_exception=raise_on_error
        ):
            stats["total"] += 1

            if ok:
                stats["success"] += 1
                # 记录涉及的索引
                if "update" in result:
                    indices_seen.add(result["update"]["_index"])
                elif "index" in result:
                    indices_seen.add(result["index"]["_index"])
            else:
                stats["failed"] += 1
                logger.error(f"写入失败: {result}")

            # 定期打印进度
            if stats["total"] % 1000 == 0:
                logger.info(
                    f"进度: {stats['total']} 个文档 "
                    f"(成功: {stats['success']}, 失败: {stats['failed']})"
                )

            # 刷新所有涉及的索引
        for index in indices_seen:
            try:
                self._client.indices.refresh(index=index)
                logger.info(f"已刷新索引: {index}")
            except Exception as e:
                logger.warning(f"刷新索引 {index} 失败: {e}")

        logger.info(
            f"写入完成 - 总计: {stats['total']}, "
            f"成功: {stats['success']}, 失败: {stats['failed']}"
        )
        return stats


def process_extended_metadata(extended_metadata: Dict[Any, Any]):
    """
    该方法主要用于临时处理历史脏数据：
    历史数据中，wiki类型资产的来源链接或附件的下载链接存储在download_key中，后续需要将该类信息剔除，不再显示
    """
    if extended_metadata.get("download_key") and validators.url(extended_metadata.get("download_key")):
        extended_metadata["download_key"] = None
    return extended_metadata


class ElasticsearchManager(BaseVectorStoreManager):
    _vector_store: ElasticsearchStore

    def __init__(self, es_url: str = ES_CONNECT_URL, es_user: Optional[str] = None, es_password: Optional[str] = None):
        self._vector_store = ElasticsearchStore(es_url, es_user, es_password)

    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        vector_store: Optional[str],
        analyzer: Optional[Analyzer],
        synonyms: Optional[List[str]],
        stopwords: Optional[List[str]],
    ) -> None:
        if not vector_store or not documents or not embeddings:
            return
        data = [
            EsIndexMapping(
                general_text=document.page_content,
                general_text_vector=embedding_vector,
                source=document.metadata["source"],
                uri=document.metadata["uri"],
                mtime=str(document.metadata["mtime"]),
                extended_metadata=serialize(document.metadata["extended_metadata"]),
                retrieve_metadata=RetrieveMetadata(
                    title=document.metadata["extended_metadata"].get("title", ""),
                    file_address=document.metadata["source"]
                )
            ).dict()
            for document, embedding_vector in zip(documents, embeddings)
        ]
        self._vector_store.bulk_insert(
            data, index=vector_store, dim=len(embeddings[0]), analyzer=analyzer, synonyms=synonyms, stopwords=stopwords
        )

    def delete_by_document_sources(self, vector_store_to_sources: Dict[str, List[str]]) -> None:
        for index, sources in vector_store_to_sources.items():
            for source in chunked(sources, ES_DELETE_DOC_BATCH_SIZE):
                param = {"index": index, "query": es_query_by_sources(source)}
                self._vector_store.bulk_delete(**param)

    def delete_vector_stores(self, vector_stores: List[str]):
        for vector_store in vector_stores:
            self._vector_store.drop(index=vector_store)

    @safe_trace()
    def retrieve(
        self,
        query: str,
        k: int,
        embedding_model_to_vector_stores: Dict[EmbeddingModel, VectorStoreElasticSearchQueryInfo],
        document_score_threshold: float = 0.0,
        collect_info: bool = True,
        analyzer: Analyzer = Analyzer.IK_ANALYZER,
        query_strategy: QueryStrategy = QueryStrategy.HYBRID_QUERY,
        request_id: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> List[RetrievedDocument]:
        tasks = [
            self._embedding_model_search_task(
                query,
                k,
                embedding_model,
                vector_stores_search_info,
                document_score_threshold,
                analyzer,
                query_strategy,
            )
            for embedding_model, vector_stores_search_info in embedding_model_to_vector_stores.items()
            if vector_stores_search_info.vs_indexes
        ]
        results = run_ordered_parallel_tasks(tasks)
        return [
            RetrievedDocument(
                text=document.general_text,
                metadata=RetrievedDocumentMetadata(
                    source=DBOX_URL_PREFIX + document.source
                    if deserialize(document.extended_metadata).get("asset_type") == AssetType.DBOX.value
                    else document.source,
                    mtime=document.mtime,
                    extended_metadata=process_extended_metadata(deserialize(document.extended_metadata)),
                    retrieve_metadata=document.retrieve_metadata
                ),
                score=document.score,
                es_doc_id=document.id,
                es_index=document.index
            )
            for document in results
        ]

    def _embedding_model_search_task(
        self,
        query: str,
        k: int,
        embedding_model: EmbeddingModel,
        vector_stores_search_info: VectorStoreElasticSearchQueryInfo,
        document_score_threshold: float,
        analyzer: Analyzer,
        query_strategy: QueryStrategy,
    ):
        return lambda: self._retrieve_by_embedding_model(
            query,
            k,
            embedding_model,
            vector_stores_search_info,
            document_score_threshold,
            analyzer,
            query_strategy,
        )

    def _retrieve_by_embedding_model(
        self,
        query: str,
        k: int,
        embedding_model: EmbeddingModel,
        vector_stores_search_info: VectorStoreElasticSearchQueryInfo,
        document_score_threshold: float,
        analyzer: Analyzer,
        query_strategy: QueryStrategy,
    ) -> List[EsQueryResult]:
        return self._vector_store.search(
            query,
            embedding=self._embedding_vector(query, embedding_model, query_strategy),
            k=k,
            vector_stores_search_info=vector_stores_search_info,
            document_score_threshold=document_score_threshold,
            analyzer=analyzer,
            query_strategy=query_strategy,
        )

    def _embedding_vector(
        self,
        query: str,
        embedding_model: EmbeddingModel,
        query_strategy: QueryStrategy,
    ) -> List[float]:
        if not query_strategy_requires_embedding(query_strategy):
            return []
        return embedding((query,), embedding_model, True)[0]

    def count(self, indexes: List[str]):
        return self._vector_store.count(indexes)

    def search_documents_by_indices_query(
        self,
        indices: List[str],
        query: Dict,
    ):
        results = self._vector_store.search_documents_by_indices(indices, query)
        return [
            RetrievedDocument(
                text=document.general_text,
                metadata=RetrievedDocumentMetadata(
                    source=document.source,
                    mtime=document.mtime,
                    extended_metadata=process_extended_metadata(deserialize(document.extended_metadata)),
                ),
            )
            for document in results
        ]

    def analyze_text_by_index(self, index: str, text: str):
        return self._vector_store.analyze_text(index, text)

    def reindex(self, index: str, synonyms: List[str], stopwords: List[str]):
        return self._vector_store.vector_store_reindex(index, synonyms, stopwords)

    def first_batch_search_document_data_by_source(self, index: str, source: str, batch_size: int, scroll_time="1m"):
        return self._vector_store.scroll_first_search(index, source, batch_size, scroll_time)

    def iterator_batch_search_document_data_by_source(self, scroll_id: str, scroll_time: str = "1m"):
        return self._vector_store.scroll_iterator_search(scroll_id, scroll_time)

    def get_source_document_data_count(self, index: str, source: str):
        return self._vector_store.get_document_data_count(index, source)

    def clear_client(self, scroll_id: str):
        return self._vector_store.clear_scroll(scroll_id)

    def update_source_document_metadata(self, source: str, online_url: str, online_url_type: str,
                                        video_start_time: str, index: str):
        """
        更新es里面的文档信息
        """
        query = es_query_by_source(source)
        docs = self.search_documents_by_indices_query([index], query)
        if not docs:
            return None
        extend_metadata = docs[0].metadata.extended_metadata
        extend_metadata["online_url"] = online_url
        if online_url_type:
            extend_metadata['online_url_type'] = online_url_type
        if video_start_time:
            extend_metadata["video_start_time"] = video_start_time
        extend_metadata_serial = serialize(extend_metadata)
        return self._vector_store.update_doc_metadata(source, extend_metadata_serial, index)

    def copy_index(self, kba_copy_config: KbaCopyConfig, source_index: str, dest_index: str, doc_id_map: Dict, download_key_map: Dict):
        return self._vector_store.copy_index(kba_copy_config, source_index, dest_index, doc_id_map, download_key_map)

    def fetch_all_docs(self, index: str) -> List[EsDocumentChunk]:
        return self._vector_store.fetch_index_docs(index)

    def rebuild_index_with_new_embed(self, index_name: str, new_dims: int = 1024):
        return self._vector_store.rebuild_index_with_new_dims(index_name, new_dims)

    def if_index_exists(self, index: str) -> bool:
        return self._vector_store.judge_index_exists(index)

    def write_embeddings(self, embedding_list: List[EsDocumentEmbeddingResult]):
        return self._vector_store.write_embeddings_streaming(embedding_list)

    def get_all_slices(self, index: str, source: str, batch_size: int) -> str:
        scroll_id = None
        try:
            text_parts = []
            scroll_id, slice_list = self.first_batch_search_document_data_by_source(
                index, source, batch_size
            )
            text_parts.append("".join([one_slice.general_text.strip() for one_slice in slice_list]))

            # 迭代查询
            while len(slice_list) == batch_size:
                scroll_id, slice_list = self.iterator_batch_search_document_data_by_source(scroll_id)
                text_parts.append("".join([one_slice.general_text.strip() for one_slice in slice_list]))

            return "".join(text_parts)
        except Exception as e:
            logger.error(f"get all slices error: {e}")
            return ""
        finally:
            if scroll_id:
                try:
                    self.clear_client(scroll_id)
                except Exception as e:
                    logger.error(f"clear scroll error: {e}")
