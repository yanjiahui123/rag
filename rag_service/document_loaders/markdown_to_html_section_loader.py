import os
import stat
import tempfile
from pathlib import Path
from typing import List, Optional

import markdown
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders.base import BaseLoader

from rag_service.document_loaders.html_section_loader import HTMLLoaderByHead
from rag_service.document_loaders.loader_utils import (
    convert_and_split_head_content_to_documents,
    convert_head_content_to_documents,
)
from rag_service.models.generic.models import HeadAndContent


def _markdown_convert_html_and_save(file_path: str, save_path_docx: str):
    with open(file_path, encoding="utf-8") as file:
        md_content = file.read()
    html = markdown.markdown(md_content, extensions=["tables", "fenced_code"])
    with os.fdopen(
        os.open(os.path.join(save_path_docx), os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR),
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        f.write(html)


class MarkdownToHTMLLoaderByHead(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        all_content = self.get_head_contents()
        return convert_head_content_to_documents(all_content, self.file_path)

    def load_and_split(self, text_splitter: Optional[TextSplitter] = None) -> List[Document]:
        all_content = self.get_head_contents()
        return convert_and_split_head_content_to_documents(all_content, self.file_path, text_splitter)

    def get_head_contents(self) -> List[HeadAndContent]:
        original_path = Path(self.file_path)
        with tempfile.TemporaryDirectory(dir=original_path.parent, prefix=original_path.stem) as temp_dir:
            temp_file_name = f"{original_path.stem}.html"
            save_path_html = Path(temp_dir).joinpath(temp_file_name).as_posix()
            _markdown_convert_html_and_save(self.file_path, save_path_html)
            html_loader = HTMLLoaderByHead(save_path_html)
            return html_loader.get_head_contents()