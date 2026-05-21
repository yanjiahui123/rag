from rag_service.original_document_fetchers.base import BaseFetcher, select_fetcher
from rag_service.original_document_fetchers.local_fetcher import LocalFetcher

Fetcher = BaseFetcher

__all__ = ["BaseFetcher", "Fetcher", "LocalFetcher", "select_fetcher"]
