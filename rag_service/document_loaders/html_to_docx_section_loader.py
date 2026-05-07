import tempfile
from pathlib import Path
from typing import List, Optional

import pypandoc
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.document_loaders.base import BaseLoader

from rag_service.document_loaders import html_table_phrase
from rag_service.document_loaders.docx_section_loader import DocxLoaderByHead


def _html_convert_dox_and_save(file_path: str, save_path_docx: str):
    with open(file_path, encoding="utf-8") as file:
        html_content = file.read()
    pypandoc.convert_text(html_content, format="html", to="docx", outputfile=save_path_docx, encoding="utf-8")


class HTMLToDocxLoaderByHead(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        original_path = Path(self.file_path)
        with tempfile.TemporaryDirectory(dir=original_path.parent, prefix=original_path.stem) as temp_dir:
            temp_file_name = f"{original_path.stem}.docx"
            save_path_docx = Path(temp_dir).joinpath(temp_file_name).as_posix()
            _html_convert_dox_and_save(self.file_path, save_path_docx)
            docx_loader = DocxLoaderByHead(save_path_docx)
            documents = docx_loader.load()
            tables = html_table_phrase.process_html_table(self.file_path)
            if tables:
                documents.extend(tables)
            return documents

    def load_and_split(self, text_splitter: Optional[TextSplitter] = None) -> List[Document]:
        original_path = Path(self.file_path)
        with tempfile.TemporaryDirectory(dir=original_path.parent, prefix=original_path.stem) as temp_dir:
            temp_file_name = f"{original_path.stem}.docx"
            save_path_docx = Path(temp_dir).joinpath(temp_file_name).as_posix()
            _html_convert_dox_and_save(self.file_path, save_path_docx)

            docx_loader = DocxLoaderByHead(save_path_docx)
            if text_splitter is None:
                _text_splitter: TextSplitter = RecursiveCharacterTextSplitter()
            else:
                _text_splitter = text_splitter

            documents = docx_loader.load_and_split(_text_splitter)
            tables = html_table_phrase.process_html_table(self.file_path)
            if tables:
                documents.extend(tables)
            return documents