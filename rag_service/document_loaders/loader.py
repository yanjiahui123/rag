import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type
from zipfile import BadZipFile

from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.document_loaders.base import BaseLoader as LCBaseLoader

from rag_service.config import SPLITTER_CHUNK_SIZE_DEFAULT
from rag_service.constants import DOC_IMPORT_FAIL_LOAD
from rag_service.document_loaders.docx_section_and_length_loader import DocxLoaderByHeadAndLength
from rag_service.document_loaders.docx_section_loader import DocxLoaderByHead
from rag_service.document_loaders.excel_loader import (
    XlsLoader as XLSLoader,
)
from rag_service.document_loaders.excel_loader import (
    XlsToMarkdownLoader,
    XlsxToMarkdownLoader,
)
from rag_service.document_loaders.excel_loader import (
    XlsxLoader as XLSXLoader,
)
from rag_service.document_loaders.html_section_and_length_loader import HTMLLoaderByHeadAndLength
from rag_service.document_loaders.html_section_loader import HTMLLoaderByHead
from rag_service.document_loaders.html_to_docx_section_loader import HTMLToDocxLoaderByHead
from rag_service.document_loaders.mardown_to_html_section_loader import MarkdownToHTMLLoaderByHead
from rag_service.document_loaders.markdown_to_html_section_and_length_loader import MarkdownToHTMLLoaderByHeadAndLength
from rag_service.document_loaders.parsed_blocks import ParsedBlock, blocks_to_documents, documents_to_parsed_blocks
from rag_service.document_loaders.ppt_helper_loader import PowerPointHelperLoader
from rag_service.document_loaders.qa_loader import XlsxForQaLoader
from rag_service.document_loaders.structured_docx_loader import StructuredDocxLoader
from rag_service.document_loaders.txt_loader import TextLoader
from rag_service.logger import Module, get_logger
from rag_service.models.enums import (
    AssetType,
    FileExtension, DocumentLoadStatus,
)
from rag_service.models.generic.models import (
    LoaderConfig,
    OriginalDocument,
    VectorizationConfig,
)
from rag_service.rag_app.api_exceptions import FileTooLarge
from rag_service.text_splitters.splitter import get_splitter
from rag_service.utils.db_util import update_doc_error_info
from rag_service.utils.online_url_util import get_online_url_info

logger = get_logger(module=Module.VECTORIZATION)

_NAME_TO_LOADER: Dict[str, Type["BaseLoader"]] = {}
_TYPE_TO_LOADERS: Dict[FileExtension, List[str]] = {}


class BaseLoader(ABC):
    loader: LCBaseLoader
    degrade_loader: Optional[LCBaseLoader] = None

    @abstractmethod
    def __init__(self, file_path: str): ...

    def __init_subclass__(cls, loader_name: str, description: str, processable_types: List[FileExtension], **kwargs):
        super().__init_subclass__(**kwargs)
        if loader_name in _NAME_TO_LOADER:
            raise Exception(f"Duplicate loader name: <{loader_name}>")
        cls.description = description
        _NAME_TO_LOADER[loader_name] = cls
        for processable_type in processable_types:
            if processable_type not in _TYPE_TO_LOADERS:
                _TYPE_TO_LOADERS[processable_type] = [loader_name]
            else:
                _TYPE_TO_LOADERS[processable_type].append(loader_name)

    def load_and_split(self, text_splitter: Optional[TextSplitter] = None) -> List[Document]:
        try:
            return self.loader.load_and_split(text_splitter=text_splitter)
        except Exception as e:
            if self.degrade_loader:
                return self.degrade_loader.load_and_split(text_splitter=text_splitter)
            raise e

    def load(self) -> List[Document]:
        try:
            return self.loader.load()
        except Exception as e:
            if self.degrade_loader:
                return self.degrade_loader.load()
            raise e

    def parse_blocks(self) -> List[ParsedBlock]:
        try:
            if hasattr(self.loader, "parse_blocks"):
                return self.loader.parse_blocks()
            return documents_to_parsed_blocks(self.load())
        except Exception as e:
            if self.degrade_loader:
                if hasattr(self.degrade_loader, "parse_blocks"):
                    return self.degrade_loader.parse_blocks()
                return documents_to_parsed_blocks(self.degrade_loader.load())
            raise e


class PdfLoader(
    BaseLoader,
    loader_name="PDF加载器",
    description="LangChain Unstructured PDF加载器",
    processable_types=[FileExtension.PDF],
):
    def __init__(self, file_path: str):
        self.loader = UnstructuredPDFLoader(file_path=file_path)


class XlsLoader(
    BaseLoader,
    loader_name="Xls加载器",
    description="Xls加载器：将Xls表头与行数据拼接",
    processable_types=[FileExtension.XLS],
):
    def __init__(self, file_path: str):
        self.loader = XLSLoader(file_path)


class XlsxLoader(
    BaseLoader,
    loader_name="Xlsx加载器",
    description="Xlsx加载器：将Xlsx表头与行数据拼接",
    processable_types=[FileExtension.XLSX],
):
    def __init__(self, file_path: str):
        self.loader = XLSXLoader(file_path)


class XlsMarkdownLoader(
    BaseLoader,
    loader_name="Xls转Markdown加载器",
    description="将Xls表头与行数据转化为Markdown表格",
    processable_types=[FileExtension.XLS],
):
    def __init__(self, file_path: str):
        self.loader = XlsToMarkdownLoader(file_path)


class XlsxMarkdownLoader(
    BaseLoader,
    loader_name="Xlsx转Markdown加载器",
    description="将Xlsx表头与行数据转化为Markdown表格",
    processable_types=[FileExtension.XLSX],
):
    def __init__(self, file_path: str):
        self.loader = XlsxToMarkdownLoader(file_path)


class WordLoader(
    BaseLoader,
    loader_name="Word加载器",
    description="LangChain Unstructured Word加载器",
    processable_types=[FileExtension.DOC, FileExtension.DOCX],
):
    def __init__(self, file_path: str):
        self.loader = UnstructuredWordDocumentLoader(file_path)
        self.degrade_loader = Docx2txtLoader(file_path)


class HTMLByHeadLoader(
    BaseLoader,
    loader_name="章节分段HTML加载器",
    description="按标题划分文档块，并将标题信息融合到正文，可在一定程度上避免标题信息丢失，保持语义的完整性",
    processable_types=[FileExtension.HTML],
):
    def __init__(self, file_path: str, source: Optional[str] = None):
        self.loader = HTMLLoaderByHead(file_path, source)
        self.degrade_loader = BSHTMLLoader(file_path)


class HTMLByHeadAndLengthLoader(
    BaseLoader,
    loader_name="章节定长分段HTML加载器",
    description="按标题划分文档块，并将标题信息融合到正文，可在一定程度上避免标题信息丢失，保持语义的完整性，同时将连续短的章节合并，避免片段太多",
    processable_types=[FileExtension.HTML],
):
    def __init__(self, file_path: str, chunk_size: int):
        self.loader = HTMLLoaderByHeadAndLength(file_path, chunk_size)
        self.degrade_loader = BSHTMLLoader(file_path)


class HTMLToDocxByHeadLoader(
    BaseLoader,
    loader_name="HTML转word章节分段加载器",
    description="先将html转换成docx文件，然后按标题划分文档块，并将标题信息融合到正文，可在一定程度上避免标题信息丢失，保持语义的完整性",
    processable_types=[FileExtension.HTML],
):
    def __init__(self, file_path: str):
        self.loader = HTMLToDocxLoaderByHead(file_path)
        self.degrade_loader = BSHTMLLoader(file_path)


class PptLoader(
    BaseLoader,
    loader_name="PPT加载器",
    description="LangChain Unstructured PPT加载器",
    processable_types=[FileExtension.PPT, FileExtension.PPTX],
):
    def __init__(self, file_path: str):
        self.loader = UnstructuredPowerPointLoader(file_path, mode="paged", include_page_breaks=False)


class TxtLoader(
    BaseLoader,
    loader_name="TXT加载器",
    description="LangChain Unstructured TXT加载器",
    processable_types=[FileExtension.TXT],
):
    def __init__(self, file_path: str):
        self.loader = TextLoader(file_path, autodetect_encoding=True, encoding="utf8")


class MarkdownToHTMLByHeadLoader(
    BaseLoader,
    loader_name="Markdown转HTML章节分段加载器",
    description="先将markdown转换成html文件，然后按标题划分文档块，并将标题信息融合到正文，可在一定程度上避免标题信息丢失，保持语义的完整性",
    processable_types=[FileExtension.MARKDOWN],
):
    def __init__(self, file_path: str):
        self.loader = MarkdownToHTMLLoaderByHead(file_path)
        self.degrade_loader = UnstructuredMarkdownLoader(file_path)


class MarkdownToHTMLByHeadAndLengthLoader(
    BaseLoader,
    loader_name="Markdown转HTML章节定长分段加载器",
    description="先将markdown转换成html文件，然后按标题划分文档块，并将标题信息融合到正文，可在一定程度上避免标题信息丢失，保持语义的完整性，同时将连续短的章节合并，避免片段太多",
    processable_types=[FileExtension.MARKDOWN],
):
    def __init__(self, file_path: str, chunk_size: int):
        self.loader = MarkdownToHTMLLoaderByHeadAndLength(file_path, chunk_size)
        self.degrade_loader = UnstructuredMarkdownLoader(file_path)


class MarkdownLoader(
    BaseLoader,
    loader_name="Markdown加载器",
    description="LangChain Unstructured Markdown加载器",
    processable_types=[FileExtension.MARKDOWN],
):
    def __init__(self, file_path: str):
        self.loader = UnstructuredMarkdownLoader(file_path)


class DocxByHeadLoader(
    BaseLoader,
    loader_name="章节分段Word加载器",
    description="按标题划分文档块，并将标题信息融合到正文，可在一定程度上避免标题信息丢失，保持语义的完整性",
    processable_types=[FileExtension.DOCX],
):
    def __init__(self, file_path: str):
        self.loader = DocxLoaderByHead(file_path)
        self.degrade_loader = Docx2txtLoader(file_path)


class DocxByHeadAndLengthLoader(
    BaseLoader,
    loader_name="章节定长分段Word加载器",
    description="按标题划分文档块，并将标题信息融合到正文，可在一定程度上避免标题信息丢失，保持语义的完整性，同时按照切分长度将连续短的章节合并，避免片段太多",
    processable_types=[FileExtension.DOCX],
):
    def __init__(self, file_path: str, chunk_size: int):
        self.loader = DocxLoaderByHeadAndLength(file_path, chunk_size)
        self.degrade_loader = Docx2txtLoader(file_path)


class StructuredDocxByBlockLoader(
    BaseLoader,
    loader_name="结构化DOCX加载器",
    description="DOCX结构化解析加载器，先输出ParsedBlock，再交由统一切片阶段处理",
    processable_types=[FileExtension.DOCX],
):
    def __init__(self, file_path: str):
        self.loader = StructuredDocxLoader(file_path)
        self.degrade_loader = Docx2txtLoader(file_path)

    def load(self) -> List[Document]:
        return blocks_to_documents(self.parse_blocks(), Document, None)

    def load_and_split(self, text_splitter: Optional[TextSplitter] = None) -> List[Document]:
        return blocks_to_documents(self.parse_blocks(), Document, text_splitter)


class PptHelperLoader(
    BaseLoader,
    loader_name="PPT助手加载器",
    description="PPT助手加载器，额外获取PPT缩略图等信息",
    processable_types=[FileExtension.PPT, FileExtension.PPTX],
):
    def __init__(self, file_path: str):
        self.loader = PowerPointHelperLoader(file_path, mode="paged", include_page_breaks=False)


class QaXlsxLoader(
    BaseLoader,
    loader_name="Xlsx问答对加载器",
    description="Xlsx格式的QA对及扩展QA对加载器",
    processable_types=[FileExtension.XLSX],
):
    def __init__(self, file_path: str):
        self.loader = XlsxForQaLoader(file_path)


_TYPE_TO_DEFAULT_LOADER: Dict[str, Type[BaseLoader]] = {
    FileExtension.PDF.value: PdfLoader,
    FileExtension.XLSX.value: XlsxMarkdownLoader,
    FileExtension.XLS.value: XlsMarkdownLoader,
    FileExtension.DOCX.value: StructuredDocxByBlockLoader,
    FileExtension.DOC.value: DocxByHeadAndLengthLoader,
    FileExtension.PPTX.value: PptLoader,
    FileExtension.PPT.value: PptLoader,
    FileExtension.TXT.value: TxtLoader,
    FileExtension.MARKDOWN.value: MarkdownToHTMLByHeadAndLengthLoader,
    FileExtension.HTML.value: HTMLByHeadAndLengthLoader,
}

_LOADERS_WITH_CHUNK_SIZE = [HTMLByHeadAndLengthLoader, MarkdownToHTMLByHeadAndLengthLoader, DocxByHeadAndLengthLoader]


def get_loaders() -> Dict[str, List[str]]:
    return {k.value: v for k, v in _TYPE_TO_LOADERS.items()}


def _get_default_loader(suffix: str) -> Type[BaseLoader]:
    loader = _TYPE_TO_DEFAULT_LOADER.get(suffix)
    if not loader:
        logger.warning("Unknown file type: %s", suffix)
        raise Exception(f"Unknown file type: {suffix}")
    return loader


def _get_loader_from_config(suffix: str, loader_config: LoaderConfig) -> Type[BaseLoader]:
    try:
        return _NAME_TO_LOADER[loader_config.loader]
    except Exception:
        return _get_default_loader(suffix)


def get_splitter_chunk_size(loader_config: LoaderConfig, vectorization_config: VectorizationConfig) -> int:
    chunk_size = None
    if loader_config and loader_config.splitter_config:
        chunk_size = loader_config.splitter_config.config.get("chunk_size")
    if not chunk_size and vectorization_config.default_splitter_config:
        chunk_size = vectorization_config.default_splitter_config.config.get("chunk_size")
    return chunk_size if chunk_size else SPLITTER_CHUNK_SIZE_DEFAULT


def get_loader(
    file_path: str,
    vectorization_config: VectorizationConfig,
    source: Optional[str] = None,
    asset_type: Optional[AssetType] = None,
) -> BaseLoader:
    loader_class = None
    loader_conf = None
    if vectorization_config.loader_configs:
        for loader_config in vectorization_config.loader_configs:
            if loader_config.process_type == Path(file_path).suffix:
                loader_class = _get_loader_from_config(Path(file_path).suffix, loader_config)
                loader_conf = loader_config
    if not loader_class:
        loader_class = _get_default_loader(Path(file_path).suffix)
    if loader_class in _LOADERS_WITH_CHUNK_SIZE:
        chunk_size = get_splitter_chunk_size(loader_conf, vectorization_config)
        return loader_class(file_path, chunk_size=chunk_size)
    if loader_class == HTMLByHeadLoader and asset_type in [
        AssetType.ASCEND_OFFICIAL_DOC,
        AssetType.KUNPENG_OFFICIAL_DOC,
    ]:
        return loader_class(file_path, source)
    return loader_class(file_path)


def normalize_loaded_document(
    doc: Document,
    original_document: OriginalDocument,
    asset_type: AssetType,
) -> Optional[Document]:
    if not doc.page_content:
        return None
    metadata = copy.deepcopy(doc.metadata)
    metadata["doc_id"] = original_document.doc_id
    source = metadata.pop("source", None)
    doc.metadata.clear()
    doc.metadata = original_document.dict()
    doc.metadata["extended_metadata"] = metadata
    if original_document.download_key:
        doc.metadata["extended_metadata"]["download_key"] = original_document.download_key
    if original_document.extended_metadata:
        doc.metadata["extended_metadata"].update(original_document.extended_metadata)
    if original_document.online_url:
        online_url, online_url_type, video_start_time = get_online_url_info(original_document.online_url)
        doc.metadata["extended_metadata"]["online_url"] = online_url if online_url else ''
        doc.metadata["extended_metadata"]["online_url_type"] = online_url_type if online_url_type else ''
        doc.metadata["extended_metadata"]["video_start_time"] = video_start_time if video_start_time else ''
    if asset_type == AssetType.QA_PAIR:
        doc.metadata["source"] = source
    return doc


def _handle_load_exception(original_document: OriginalDocument, error: Exception):
    if isinstance(error, FileTooLarge):
        logger.exception("Failed to load %s, FileTooLarge: %s", original_document, error)
        update_doc_error_info([original_document.doc_id],
                              DOC_IMPORT_FAIL_LOAD.format("文件超过规定大小"), DocumentLoadStatus.FAILURE)
        return
    if isinstance(error, UnicodeDecodeError):
        logger.exception("Failed to load %s, UnicodeDecodeError: %s", original_document, error)
        update_doc_error_info([original_document.doc_id],
                              DOC_IMPORT_FAIL_LOAD.format("文档非UTF-8编码"), DocumentLoadStatus.FAILURE)
        return
    if isinstance(error, (BadZipFile, ValueError)):
        logger.exception("Failed to load %s, BadZipFile: %s", original_document, error)
        update_doc_error_info([original_document.doc_id],
                              DOC_IMPORT_FAIL_LOAD.format("文档格式或数据异常"), DocumentLoadStatus.FAILURE)
        return
    logger.exception("Failed to load %s, exception: %s", original_document, error)
    update_doc_error_info([original_document.doc_id],
                          DOC_IMPORT_FAIL_LOAD.format(str(error)), DocumentLoadStatus.FAILURE)


def parse_file(
    original_document: OriginalDocument,
    vectorization_config: VectorizationConfig,
    asset_type: AssetType,
) -> List[ParsedBlock]:
    try:
        loader = get_loader(original_document.uri, vectorization_config, original_document.source, asset_type)
        return loader.parse_blocks()
    except Exception as e:
        _handle_load_exception(original_document, e)
    return []


def split_parsed_file(
    parsed_blocks: List[ParsedBlock],
    original_document: OriginalDocument,
    vectorization_config: VectorizationConfig,
    asset_type: AssetType,
) -> List[Document]:
    try:
        splitter = get_splitter(original_document.uri, vectorization_config)
        docs = blocks_to_documents(parsed_blocks, Document, splitter)
        processed_docs = []
        for doc in docs:
            normalized_doc = normalize_loaded_document(doc, original_document, asset_type)
            if normalized_doc:
                processed_docs.append(normalized_doc)
        return processed_docs
    except Exception as e:
        _handle_load_exception(original_document, e)
    return []


def load_file(
    original_document: OriginalDocument,
    vectorization_config: VectorizationConfig,
    asset_type: AssetType,
) -> List[Document]:
    parsed_blocks = parse_file(original_document, vectorization_config, asset_type)
    if not parsed_blocks:
        return []
    return split_parsed_file(parsed_blocks, original_document, vectorization_config, asset_type)
