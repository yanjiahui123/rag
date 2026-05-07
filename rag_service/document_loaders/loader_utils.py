from typing import List, Optional

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from rag_service.config import SPLITTER_CHUNK_SIZE_DEFAULT
from rag_service.models.generic.models import HeadAndContent


def merge_head_content_by_length(
    head_contents: List[HeadAndContent], chunk_size=SPLITTER_CHUNK_SIZE_DEFAULT
) -> List[HeadAndContent]:
    merge_head_contents = []
    merge_content = ""
    title = ""
    headers = []
    contents = []
    for item in head_contents:
        if not merge_content and not title:
            title = item.title
            merge_content = item.content
            contents.append(item.content)
            headers.append(title)
            continue
        add_content = f"{item.title} {item.content}"
        if len(title) + len(merge_content) + len(add_content) <= chunk_size:
            merge_content += "\n" + add_content
            contents.append(item.content)
            headers.append(item.title)
            continue
        merge_head_contents.append(HeadAndContent(title=title, content=merge_content, headers=headers, header_contents=contents))
        merge_content = item.content
        title = item.title
        headers = [title]
        contents = [item.content]
    if merge_content or title:
        merge_head_contents.append(HeadAndContent(title=title, content=merge_content, headers=headers, header_contents=contents))
    return merge_head_contents


def convert_and_split_head_content_to_documents(
    head_contents: List[HeadAndContent], source: str, text_splitter: Optional[TextSplitter] = None
) -> List[Document]:
    split_documents = []
    if text_splitter is None:
        _text_splitter: TextSplitter = RecursiveCharacterTextSplitter()
    else:
        _text_splitter = text_splitter
    for content in head_contents:
        splits = _text_splitter.split_text(content.content)
        if not splits:
            doc = Document(
                page_content=f"{content.title}",
                metadata={"source": source, "headers": content.headers, "header_contents": content.header_contents}
            )
            split_documents.append(doc)
            continue
        for split in splits:
            split_doc = Document(
                page_content=f"{content.title} {split}",
                metadata={"source": source, "headers": content.headers, "header_contents": content.header_contents}
            )
            split_documents.append(split_doc)
    return split_documents


def convert_head_content_to_documents(head_contents: List[HeadAndContent], source: str) -> List[Document]:
    documents = []
    for content in head_contents:
        doc = Document(page_content=f"{content.title} {content.content}", metadata={"source": source})
        documents.append(doc)
    return documents


def build_markdown_table_string(headers: List[str], rows: List[List[str]]) -> str:
    # 构建Markdown表格
    markdown = ['|' + '|'.join(headers) + '|', '|' + '|'.join(['-'] * len(headers)) + '|']
    # 表格内容
    for row in rows:
        markdown.append('|' + '|'.join(row) + '|')
    return '\n' + '\n'.join(markdown) + '\n'