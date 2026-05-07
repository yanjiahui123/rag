from typing import Dict, List, Optional

from pydantic import BaseModel

from rag_service.models.api.models import RetrieveMetadata
from rag_service.models.generic.models import VectorStoreElasticSearchQueryInfo

NUM_CANDIDATES = 100


def es_index_mapping(dim: int) -> Dict:
    return {
        "mappings": {
            "properties": {
                "general_text_vector": {"type": "dense_vector", "dims": dim, "index": "true", "similarity": "cosine"},
                "general_text": {"type": "text"},
                "source": {"type": "keyword"},
                "uri": {"type": "text"},
                "mtime": {"type": "text"},
                "extended_metadata": {"type": "text"},
                "retrieve_metadata": {
                    "properties": {
                        "title": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword"} # 支持精确匹配
                            }
                        },
                        "file_address": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword"} # 完整路径精确匹配
                            }
                        }
                    }
                }
            }
        }
    }


def es_ik_index_mapping(dim: int, synonyms: List[str], stopwords: List[str]) -> Dict:
    return {
        "settings": {
            "index": {"number_of_replicas": 1},
            "analysis": {
                "filter": {
                    "my_synonym": {"type": "synonym", "synonyms": synonyms},
                    "my_stop": {"type": "stop", "stopwords": stopwords},
                },
                "analyzer": {
                    "custom_index_analyzer": {
                        "type": "custom",
                        "tokenizer": "ik_max_word",
                        "filter": ["lowercase", "my_synonym", "my_stop"],
                    },
                    "custom_search_analyzer": {
                        "type": "custom",
                        "tokenizer": "ik_smart",
                        "filter": ["lowercase", "my_synonym", "my_stop"],
                    },
                },
            },
        },
        "mappings": {
            "properties": {
                "general_text_vector": {"type": "dense_vector", "dims": dim, "index": "true", "similarity": "cosine"},
                "general_text": {
                    "type": "text",
                    "analyzer": "custom_index_analyzer",
                    "search_analyzer": "custom_search_analyzer",
                },
                "source": {"type": "keyword"},
                "uri": {"type": "text"},
                "mtime": {"type": "text"},
                "extended_metadata": {"type": "text"},
                "retrieve_metadata": {
                    "properties": {
                        "title": {
                            "type": "text",
                            "analyzer": "custom_index_analyzer",
                            "search_analyzer": "custom_search_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword"}  # 支持精确匹配
                            }
                        },
                        "file_address": {
                            "type": "text",
                            "analyzer": "custom_index_analyzer",
                            "search_analyzer": "custom_search_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword"}  # 完整路径精确匹配
                            }
                        }
                    }
                }
            }
        },
    }


ES_FETCHED_MAPPING_PROPERTIES = ["general_text", "source", "mtime", "extended_metadata", "uri", "retrieve_metadata"]


def es_phrase_query(
    query: str, keywords: str, embedding: List[float], k: int, vs_search_info: VectorStoreElasticSearchQueryInfo
) -> Dict:
    k = min(k, NUM_CANDIDATES)
    return {
        "query": {
            "bool": {
                "should": [
                    {"match": {"general_text": query}},
                    {"match": {"general_text": {"query": keywords, "operator": "and"}}},
                    {"match_phrase": {"general_text": {"query": keywords, "slop": 10, "boost": 2}}},
                    {"match": {"retrieve_metadata.title": {"query": query, "boost": 2.0}}},
                    {"match": {"retrieve_metadata.file_address": {"query": query, "boost": 2.0}}}
                ],
                "must_not": {"terms": {"source": vs_search_info.document_close_sources}},
            }
        },
        "knn": {
            "field": "general_text_vector",
            "query_vector": embedding,
            "k": k,
            "num_candidates": NUM_CANDIDATES,
            "filter": {"bool": {"must_not": {"terms": {"source": vs_search_info.document_close_sources}}}},
        },
        "_source": ES_FETCHED_MAPPING_PROPERTIES,
    }


def es_match_query(
    query: str, embedding: List[float], k: int, vs_search_info: VectorStoreElasticSearchQueryInfo
) -> Dict:
    k = min(k, NUM_CANDIDATES)
    return {
        "query": {
            "bool": {
                "should": [
                    {"match": {"general_text": query}},
                    {"match": {"retrieve_metadata.title": {"query": query, "boost": 2.0}}},
                    {"match": {"retrieve_metadata.file_address": {"query": query, "boost": 2.0}}}
                ],
                "must_not": {"terms": {"source": vs_search_info.document_close_sources}},
            }
        },
        "knn": {
            "field": "general_text_vector",
            "query_vector": embedding,
            "k": k,
            "num_candidates": NUM_CANDIDATES,
            "filter": {"bool": {"must_not": {"terms": {"source": vs_search_info.document_close_sources}}}},
        },
        "_source": ES_FETCHED_MAPPING_PROPERTIES,
    }


def es_knn_query(embedding: List[float], vs_search_info: VectorStoreElasticSearchQueryInfo, k: int) -> Dict:
    k = min(k, NUM_CANDIDATES)
    return {
        "knn": {
            "field": "general_text_vector",
            "query_vector": embedding,
            "k": k,
            "num_candidates": NUM_CANDIDATES,
            "filter": {"bool": {"must_not": {"terms": {"source": vs_search_info.document_close_sources}}}},
        },
        "_source": ES_FETCHED_MAPPING_PROPERTIES,
    }


def es_text_match_query(query: str, vs_search_info: VectorStoreElasticSearchQueryInfo) -> Dict:
    return {
        "query": {
            "bool": {
                "should": [
                    {"match": {"general_text": query}},
                    {"match": {"retrieve_metadata.title": {"query": query, "boost": 2.0}}},
                    {"match": {"retrieve_metadata.file_address": {"query": query, "boost": 2.0}}}
                ],
                "must_not": {"terms": {"source": vs_search_info.document_close_sources}},
            }
        },
        "_source": ES_FETCHED_MAPPING_PROPERTIES,
    }


def es_text_phrase_query(query: str, keywords: str, vs_search_info: VectorStoreElasticSearchQueryInfo) -> Dict:
    return {
        "query": {
            "bool": {
                "should": [
                    {"match": {"general_text": query}},
                    {"match": {"general_text": {"query": keywords, "operator": "and"}}},
                    {"match_phrase": {"general_text": {"query": keywords, "slop": 10, "boost": 2}}},
                    {"match": {"retrieve_metadata.title": {"query": query, "boost": 2.0}}},
                    {"match": {"retrieve_metadata.file_address": {"query": query, "boost": 2.0}}}
                ],
                "must_not": {"terms": {"source": vs_search_info.document_close_sources}},
            }
        },
        "_source": ES_FETCHED_MAPPING_PROPERTIES,
    }


def es_query_by_source(source: str):
    return {"query": {"term": {"source": source}}}


def es_query_by_sources(sources: List[str]):
    return {"query": {"terms": {"source": sources}}}


def es_query_k_documents(k: int) -> Dict:
    return {
        "query": {"function_score": {"query": {"match_all": {}}, "random_score": {}}},
        "size": k,
        "_source": ES_FETCHED_MAPPING_PROPERTIES,
    }


def es_query_k_documents_by_specify_source(k: int, sources: List) -> Dict:
    return {
        "query": {"function_score": {"query": {"terms": {"source": sources}}, "random_score": {}}},
        "size": k,
        "_source": ES_FETCHED_MAPPING_PROPERTIES,
    }


def es_scroll_query(source: str, batch_size: int):
    return {"query": {"term": {"source": source}}, "size": batch_size}


def es_intent_index_mapping(dim: int) -> Dict:
    return {
        "mappings": {
            "properties": {
                "intent_name": {"type": "keyword", "index": "true"},
                "intent_sn": {"type": "keyword", "index": "true"},
                "intent_value": {"type": "text", "index": "true"},
                "intent_value_vector": {"type": "dense_vector", "dims": dim, "index": "true", "similarity": "cosine"},
                "update_time": {"type": "text", "index": "true"},
            }
        }
    }


def es_index_match_query(query: str, embedding: List[float], k: int) -> Dict:
    k = min(k, NUM_CANDIDATES)
    return {
        "query": {"bool": {"should": [{"match": {"general_text": query}}]}},
        "knn": {"field": "intent_value_vector", "query_vector": embedding, "k": k, "num_candidates": NUM_CANDIDATES},
    }


def es_update_doc_metadata(source: str, extend_metadata: str):
    return {
        "query": {"term": {"source": source}},
        "script": {
            "source": "ctx._source['extended_metadata'] = params.extended_metadata",
            "params": {"extended_metadata": extend_metadata},
        },
    }


def es_match_all():
    return {"query": {"match_all": {}}}


class EsIndexMapping(BaseModel):
    general_text: str
    general_text_vector: List[float]
    source: str
    uri: str
    mtime: str
    extended_metadata: str
    retrieve_metadata: Optional[RetrieveMetadata] = None


class EsQueryResult(BaseModel):
    index: str
    id: str
    score: float
    general_text: str
    source: str
    mtime: str
    extended_metadata: str
    uri: Optional[str]
    retrieve_metadata: Optional[RetrieveMetadata] = None


class IntentQueryResult(BaseModel):
    score: float
    intent_name: str
    intent_value: str