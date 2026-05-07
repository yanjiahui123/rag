from typing import List, Optional

from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter

from rag_service.config import SPLITTER_CHUNK_SIZE_DEFAULT
from rag_service.document_loaders.loader_utils import (
    convert_and_split_head_content_to_documents,
    convert_head_content_to_documents,
    merge_head_content_by_length,
)
from rag_service.document_loaders.mardown_to_html_section_loader import MarkdownToHTMLLoaderByHead


class MarkdownToHTMLLoaderByHeadAndLength(MarkdownToHTMLLoaderByHead):
    def __init__(self, file_path: str, chunk_size=SPLITTER_CHUNK_SIZE_DEFAULT):
        super().__init__(file_path)
        self.chunk_size = chunk_size

    def load(self) -> List[Document]:
        all_content = self.get_head_contents()
        merge_contents = merge_head_content_by_length(all_content, self.chunk_size)
        return convert_head_content_to_documents(merge_contents, self.file_path)

    def load_and_split(self, text_splitter: Optional[TextSplitter] = None) -> List[Document]:
        all_content = self.get_head_contents()
        merge_contents = merge_head_content_by_length(all_content, self.chunk_size)
        return convert_and_split_head_content_to_documents(merge_contents, self.file_path, text_splitter)