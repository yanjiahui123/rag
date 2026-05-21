import importlib
import sys
import types
import unittest

from tests.pydantic_stub import install_pydantic_stub

install_pydantic_stub()


class FakeDocument:
    def __init__(self, page_content="", metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeEnumValue:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, FakeEnumValue) and self.value == other.value


class FakeOriginalDocument:
    def __init__(self):
        self.doc_id = "doc-1"
        self.uri = "demo.txt"
        self.source = "demo-source"
        self.mtime = 123
        self.download_key = "download-key"
        self.extended_metadata = {"owner": "team-a"}
        self.online_url = "https://example.test/video"

    def dict(self):
        return {
            "doc_id": self.doc_id,
            "uri": self.uri,
            "source": self.source,
            "mtime": self.mtime,
            "extended_metadata": {},
        }


class FakeAssetType:
    QA_PAIR = "qa_pair"
    ASCEND_OFFICIAL_DOC = "ascend"
    KUNPENG_OFFICIAL_DOC = "kunpeng"


def install_loader_import_stubs():
    sys.modules["langchain"] = types.ModuleType("langchain")
    sys.modules["langchain.docstore"] = types.ModuleType("langchain.docstore")
    langchain_doc = types.ModuleType("langchain.docstore.document")
    langchain_doc.Document = FakeDocument
    sys.modules["langchain.docstore.document"] = langchain_doc
    text_splitter = types.ModuleType("langchain.text_splitter")
    text_splitter.TextSplitter = object
    sys.modules["langchain.text_splitter"] = text_splitter

    community = types.ModuleType("langchain_community.document_loaders")
    for name in [
        "BSHTMLLoader",
        "Docx2txtLoader",
        "UnstructuredMarkdownLoader",
        "UnstructuredPDFLoader",
        "UnstructuredPowerPointLoader",
        "UnstructuredWordDocumentLoader",
    ]:
        setattr(community, name, type(name, (), {"__init__": lambda self, *args, **kwargs: None}))
    sys.modules["langchain_community"] = types.ModuleType("langchain_community")
    sys.modules["langchain_community.document_loaders"] = community
    community_base = types.ModuleType("langchain_community.document_loaders.base")
    community_base.BaseLoader = object
    sys.modules["langchain_community.document_loaders.base"] = community_base

    config = types.ModuleType("rag_service.config")
    config.SPLITTER_CHUNK_SIZE_DEFAULT = 1000
    sys.modules["rag_service.config"] = config

    constants = types.ModuleType("rag_service.constants")
    constants.DOC_IMPORT_FAIL_LOAD = "load failed: {}"
    sys.modules["rag_service.constants"] = constants

    logger = types.ModuleType("rag_service.logger")
    logger.Module = types.SimpleNamespace(VECTORIZATION="vectorization")
    logger.get_logger = lambda module=None: types.SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )
    sys.modules["rag_service.logger"] = logger

    enums = types.ModuleType("rag_service.models.enums")
    enums.AssetType = FakeAssetType
    enums.FileExtension = types.SimpleNamespace(
        PDF=FakeEnumValue(".pdf"),
        XLS=FakeEnumValue(".xls"),
        XLSX=FakeEnumValue(".xlsx"),
        DOC=FakeEnumValue(".doc"),
        DOCX=FakeEnumValue(".docx"),
        PPT=FakeEnumValue(".ppt"),
        PPTX=FakeEnumValue(".pptx"),
        TXT=FakeEnumValue(".txt"),
        MARKDOWN=FakeEnumValue(".md"),
        HTML=FakeEnumValue(".html"),
    )
    enums.DocumentLoadStatus = types.SimpleNamespace(FAILURE="failure")
    sys.modules["rag_service.models.enums"] = enums

    generic = types.ModuleType("rag_service.models.generic.models")
    generic.LoaderConfig = object
    generic.OriginalDocument = FakeOriginalDocument
    generic.VectorizationConfig = object
    sys.modules["rag_service.models.generic.models"] = generic

    api_exceptions = types.ModuleType("rag_service.rag_app.api_exceptions")
    api_exceptions.FileTooLarge = type("FileTooLarge", (Exception,), {})
    sys.modules["rag_service.rag_app.api_exceptions"] = api_exceptions

    splitter = types.ModuleType("rag_service.text_splitters.splitter")
    splitter.get_splitter = lambda uri, config: None
    sys.modules["rag_service.text_splitters.splitter"] = splitter

    db_util = types.ModuleType("rag_service.utils.db_util")
    db_util.update_doc_error_info = lambda *args, **kwargs: None
    sys.modules["rag_service.utils.db_util"] = db_util

    online_url = types.ModuleType("rag_service.utils.online_url_util")
    online_url.get_online_url_info = lambda url: ("https://example.test/video", "video", "10")
    sys.modules["rag_service.utils.online_url_util"] = online_url

    loader_class_modules = {
        "rag_service.document_loaders.docx_section_and_length_loader": "DocxLoaderByHeadAndLength",
        "rag_service.document_loaders.docx_section_loader": "DocxLoaderByHead",
        "rag_service.document_loaders.excel_loader": [
            "XlsLoader",
            "XlsToMarkdownLoader",
            "XlsxToMarkdownLoader",
            "XlsxLoader",
        ],
        "rag_service.document_loaders.html_section_and_length_loader": "HTMLLoaderByHeadAndLength",
        "rag_service.document_loaders.html_section_loader": "HTMLLoaderByHead",
        "rag_service.document_loaders.html_to_docx_section_loader": "HTMLToDocxLoaderByHead",
        "rag_service.document_loaders.mardown_to_html_section_loader": "MarkdownToHTMLLoaderByHead",
        "rag_service.document_loaders.markdown_to_html_section_and_length_loader": "MarkdownToHTMLLoaderByHeadAndLength",
        "rag_service.document_loaders.ppt_helper_loader": "PowerPointHelperLoader",
        "rag_service.document_loaders.qa_loader": "XlsxForQaLoader",
        "rag_service.document_loaders.txt_loader": "TextLoader",
    }
    for module_name, class_names in loader_class_modules.items():
        module = types.ModuleType(module_name)
        if isinstance(class_names, str):
            class_names = [class_names]
        for class_name in class_names:
            setattr(module, class_name, type(class_name, (), {"__init__": lambda self, *args, **kwargs: None}))
        sys.modules[module_name] = module


class LoaderPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_loader_import_stubs()
        cls.loader = importlib.import_module("rag_service.document_loaders.loader")

    def test_normalize_loaded_document_preserves_loader_metadata_in_extended_metadata(self):
        original_document = FakeOriginalDocument()
        doc = FakeDocument(
            page_content="chunk text",
            metadata={
                "source": "loader-source",
                "display": {"type": "table", "format": "html", "content": "<table></table>"},
            },
        )

        normalized = self.loader.normalize_loaded_document(doc, original_document, FakeAssetType.QA_PAIR)

        self.assertEqual(normalized.page_content, "chunk text")
        self.assertEqual(normalized.metadata["source"], "loader-source")
        extended_metadata = normalized.metadata["extended_metadata"]
        self.assertEqual(extended_metadata["doc_id"], "doc-1")
        self.assertEqual(extended_metadata["download_key"], "download-key")
        self.assertEqual(extended_metadata["owner"], "team-a")
        self.assertEqual(extended_metadata["online_url_type"], "video")
        self.assertEqual(extended_metadata["display"]["type"], "table")

    def test_parse_file_prefers_structured_document_over_legacy_load(self):
        original_document = FakeOriginalDocument()
        parsed_document_factory = self.loader.ParsedDocument

        class StructuredLoader:
            load_called = False

            def parse_to_document(self):
                return parsed_document_factory(text="structured docx", metadata={"source": "demo.docx"})

            def load(self):
                self.load_called = True
                return [FakeDocument(page_content="legacy docx", metadata={"source": "demo.docx"})]

        structured_loader = StructuredLoader()
        original_get_loader = self.loader.get_loader
        self.loader.get_loader = lambda *args, **kwargs: structured_loader
        try:
            parsed_document = self.loader.parse_file(original_document, types.SimpleNamespace(), FakeAssetType.QA_PAIR)
        finally:
            self.loader.get_loader = original_get_loader

        self.assertEqual(parsed_document.text, "structured docx")
        self.assertEqual(parsed_document.blocks, [])
        self.assertFalse(structured_loader.load_called)

    def test_parse_file_configures_structured_image_upload_prefix(self):
        original_document = FakeOriginalDocument()
        original_document.doc_id = "22222222-2222-2222-2222-222222222222"
        parsed_document_factory = self.loader.ParsedDocument

        class StructuredLoader:
            def __init__(self):
                self.loader = types.SimpleNamespace(
                    artifact_type="structured_docx",
                    image_upload_prefix="",
                )

            def parse_to_document(self):
                return parsed_document_factory(metadata={"image_upload_prefix": self.loader.image_upload_prefix})

        structured_loader = StructuredLoader()
        original_get_loader = self.loader.get_loader
        self.loader.get_loader = lambda *args, **kwargs: structured_loader
        try:
            parsed_document = self.loader.parse_file(
                original_document,
                types.SimpleNamespace(),
                FakeAssetType.QA_PAIR,
                knowledge_base_asset_id="11111111-1111-1111-1111-111111111111",
            )
        finally:
            self.loader.get_loader = original_get_loader

        self.assertEqual(
            parsed_document.metadata["image_upload_prefix"],
            "11111111-1111-1111-1111-111111111111/"
            "22222222-2222-2222-2222-222222222222/"
            "artifacts/structured_docx/images/",
        )

    def test_docx_default_loader_uses_structured_loader_for_new_documents(self):
        default_loader = self.loader._TYPE_TO_DEFAULT_LOADER[".docx"]

        self.assertIs(default_loader, self.loader.StructuredDocxByBlockLoader)

    def test_xlsx_default_loader_uses_structured_loader_for_new_documents(self):
        default_loader = self.loader._TYPE_TO_DEFAULT_LOADER[".xlsx"]

        self.assertIs(default_loader, self.loader.StructuredXlsxByBlockLoader)

    def test_html_default_loader_uses_structured_loader_for_new_documents(self):
        default_loader = self.loader._TYPE_TO_DEFAULT_LOADER[".html"]

        self.assertIs(default_loader, self.loader.StructuredHtmlByBlockLoader)

    def test_markdown_default_loader_uses_structured_loader_for_new_documents(self):
        default_loader = self.loader._TYPE_TO_DEFAULT_LOADER[".md"]

        self.assertIs(default_loader, self.loader.StructuredMarkdownByBlockLoader)

    def test_structured_default_loaders_receive_original_source(self):
        vectorization_config = types.SimpleNamespace(loader_configs=[])

        for suffix in [".docx", ".xlsx", ".html", ".md"]:
            with self.subTest(suffix=suffix):
                loader = self.loader.get_loader(
                    "tmp/internal-upload-id" + suffix,
                    vectorization_config,
                    "用户上传的原始文件名" + suffix,
                    FakeAssetType.QA_PAIR,
                )

                self.assertEqual(loader.loader.source, "用户上传的原始文件名" + suffix)


if __name__ == "__main__":
    unittest.main()
