from typing import List, Tuple, Dict, Any

import requests
from cachetools import TTLCache, cached

from rag_service.constants import QWEN_EMBEDDING_DIM, QWEN_MODEL_MAX_LENGTH, QWEN_MODEL_BATCH_MAX_LENGTH
from rag_service.exceptions import EmbeddingConnectException, EmbeddingResponseException
from rag_service.logger import Module, get_logger
from rag_service.logger.wrapper_trace import safe_trace
from rag_service.models.enums import EmbeddingModel
from rag_service.utils.llm_util import get_embedding_urls, get_request_header

logger = get_logger(module=Module.VECTORIZATION)


@cached(cache=TTLCache(maxsize=1000, ttl=3600 * 1))
def embedding(
    texts: Tuple[str], embedding_model: EmbeddingModel = EmbeddingModel.get_default_model(),
    enable_instruct: bool = False
) -> List[List[float]]:
    if not texts:
        return []

    return _maas_embedding(texts, embedding_model, enable_instruct)


def _truncate_text(text: str, max_length: int = QWEN_MODEL_MAX_LENGTH) -> str:
    """截断文本到指定长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length]


def _split_texts_by_batch_limit(texts: Tuple[str], max_text_length: int = QWEN_MODEL_MAX_LENGTH,
                                max_batch_length: int = QWEN_MODEL_BATCH_MAX_LENGTH) -> List[List[str]]:
    """
    根据单文本和批次总长度限制分割文本

    Args:
        texts: 原始文本元组
        max_text_length: 单个文本最大长度
        max_batch_length: 批次总长度限制

    Returns:
        分割后的文本批次列表
    """
    batches = []
    current_batch = []
    current_batch_length = 0

    for text in texts:
        # 截断过长的文本
        truncated_text = _truncate_text(text, max_text_length)
        text_length = len(truncated_text)

        # 如果当前批次加上新文本会超过限制，则开始新批次
        if current_batch and current_batch_length + text_length > max_batch_length:
            batches.append(current_batch)
            current_batch = []
            current_batch_length = 0

        current_batch.append(truncated_text)
        current_batch_length += text_length

    # 添加最后一个批次
    if current_batch:
        batches.append(current_batch)

    return batches


def _prepare_request_data(texts: List[str], embedding_model: EmbeddingModel, enable_instruct: bool = False) -> Dict[str, Any]:
    """
    根据模型类型准备请求数据

    Args:
        texts: 文本列表
        embedding_model: 模型类型

    Returns:
        请求数据字典
    """
    if embedding_model == EmbeddingModel.QWEN3_EMBEDDING_4B:
        return {
            "model": embedding_model.value,
            "input": texts,
            "dimensions": QWEN_EMBEDDING_DIM,
            "enable_instruct": enable_instruct,
            "truncate": "END"
        }
    else:
        return {
            "inputs": texts,
            "model": embedding_model.value
        }


@safe_trace()
def _request_embedding_api(data: Dict[str, Any], url: str, headers: Dict[str, str]) -> List[List[float]]:
    """
    调用embedding API

    Args:
        data: 请求数据
        url: API地址
        headers: 请求头

    Returns:
        embedding结果

    Raises:
        EmbeddingResponseException: API返回错误
    """
    try:
        response = requests.post(headers=headers, url=url, json=data, timeout=60, verify=False)
    except requests.exceptions.ConnectionError as e:
        logger.exception("embedding服务请求连接错误")
        return None
    except requests.exceptions.Timeout as e:
        logger.exception("embedding服务请求超时(超时时间: %s)", 60)
        return None

    if not response.ok:
        message = f"embedding 服务连接失败,url：{url}"
        logger.error(message)
        raise EmbeddingConnectException("embedding 服务连接失败")

    response_data = response.json()
    if data.get("model") == EmbeddingModel.BGE_LARGE_ZH_V1_5.value and isinstance(response_data, dict):
        message = f"embedding 服务请求错误, error info:{response_data.get('status')}"
        logger.error(message)
        raise EmbeddingResponseException("embedding 服务请求错误")
    if data.get("model") == EmbeddingModel.QWEN3_EMBEDDING_4B.value:
        return _parse_qwen_embedding_response(response_data)
    return response_data


@safe_trace()
def _maas_embedding(texts: Tuple[str], embedding_model: EmbeddingModel, enable_instruct: bool = False) -> List[List[float]]:
    """
    获取文本的embedding向量

    Args:
        texts: 文本元组
        embedding_model: 模型类型

    Returns:
        embedding向量列表

    Raises:
        EmbeddingConnectException: 所有服务端点连接失败
        EmbeddingResponseException: API返回错误
    """

    # 根据模型配置分割文本
    if embedding_model == EmbeddingModel.QWEN3_EMBEDDING_4B:
        text_batches = _split_texts_by_batch_limit(texts)
    else:
        # 原有模型不需要分批
        text_batches = [list(texts)]

    all_embeddings = []

    # 处理每个批次
    for batch in text_batches:
        data = _prepare_request_data(batch, embedding_model, enable_instruct)
        batch_result = None

        # 尝试所有端点
        for url in get_embedding_urls():
            batch_result = _request_embedding_api(data, url, get_request_header())
            if batch_result is not None:
                break

        if batch_result is None:
            raise EmbeddingConnectException("embedding 服务连接失败")

        all_embeddings.extend(batch_result)

    return all_embeddings


def _parse_qwen_embedding_response(response):
    if not response:
        return response
    result = []
    data = response.get("data", [])
    if not data:
        message = f"embedding 服务请求错误, error info:{response}"
        logger.error(message)
        raise EmbeddingResponseException("embedding 服务请求错误")
    for item in data:
        embedding_data = item.get("embedding", [])
        result.append(embedding_data)
    return result