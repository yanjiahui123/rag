from typing import List

import docx
from langchain.docstore.document import Document as LCDocument
from langchain_community.document_loaders.base import BaseLoader

from rag_service.document_loaders.loader_utils import build_markdown_table_string
from rag_service.logger import Module, get_logger

logger = get_logger(module=Module.VECTORIZATION)


class DocxLoader(BaseLoader):
    """Loading logic for loading documents from docx."""

    def __init__(self, file_path: str, image_inline=False):
        """Initialize with filepath and options."""
        self.doc_path = file_path
        self.do_ocr = image_inline
        self.doc = None
        self.table_index = 0

    def _handle_paragraph(self, element):
        """docx.oxml.text.paragraph.CT_P"""
        return element.text

    def _handle_table(self, element):
        """docx.oxml.table.CT_Tbl"""
        rows = list(element.rows)
        headers = [cell.text for cell in rows[0].cells]
        data = [[cell.text.replace("\n", " ").replace("|", "\|").strip() for cell in row.cells] for row in rows[1:]]
        return build_markdown_table_string(headers, data)

    def load(self) -> List[LCDocument]:
        """Load documents."""
        docs: List[LCDocument] = []
        all_text = []
        self.doc = docx.Document(self.doc_path)
        for element in self.doc.element.body:
            if element.tag.endswith("tbl"):
                # handle table
                table_text = self._handle_table(self.doc.tables[self.table_index])
                self.table_index += 1
                all_text.append(table_text)
            elif element.tag.endswith("p"):
                # handle paragraph
                xmlstr = str(element.xml)
                if "pic:pic" in xmlstr and self.do_ocr:
                    pic_texts = ""
                    all_text.extend(pic_texts)
                paragraph = docx.text.paragraph.Paragraph(element, self.doc)
                para_text = self._handle_paragraph(paragraph)
                all_text.append(para_text)
        onetext = " ".join(all_text)
        docs.append(LCDocument(page_content=onetext, metadata={"source": self.doc_path}))
        return docs
