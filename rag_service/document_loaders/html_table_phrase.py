import re

import pandas as pd
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

from rag_service.document_loaders.loader_utils import build_markdown_table_string
from rag_service.logger import Module, get_logger
from rag_service.text_splitters.markdown_table_splitter import MarkdownTableSplitter

logger = get_logger(module=Module.VECTORIZATION)


def extract_tables_from_html(html_content):
    """
    从HTML内容中提取所有表格
    """
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    return tables


def table_to_markdown(table):
    """
    将BeautifulSoup表格对象转换为Markdown格式
    """
    # 提取表头
    headers = []
    header_row = table.find("thead")
    if header_row:
        headers = [th.get_text().strip() for th in header_row.find_all(["th", "td"])]
    else:
        # 尝试从第一行获取表头
        first_row = table.find("tr")
        if first_row:
            headers = [th.get_text().strip() for th in first_row.find_all(["th", "td"])]

    # 如果没有找到表头，则创建默认表头
    if not headers:
        # 查找表格中最大的行长度
        max_cols = 0
        for row in table.find_all("tr"):
            cols = len(row.find_all(["td", "th"]))
            max_cols = max(max_cols, cols)
        headers = [f"Column {i + 1}" for i in range(max_cols)]

    # 提取表格数据行
    rows = []
    for tr in table.find_all("tr"):
        row = [td.get_text().strip() for td in tr.find_all(["td", "th"])]
        if row and row != headers:  # 避免重复添加表头
            rows.append(row)

    # 创建pandas DataFrame
    if rows:
        # 确保所有行具有相同的长度
        max_len = max(len(headers), max(len(row) for row in rows))
        headers = headers + [""] * (max_len - len(headers))
        rows = [row + [""] * (max_len - len(row)) for row in rows]
        df = pd.DataFrame(rows, columns=headers)
    else:
        df = pd.DataFrame(columns=headers)

    # 转换为Markdown
    markdown_table = df.to_markdown(index=False)
    return markdown_table


def extract_table_context(table):
    """
    提取表格周围的上下文信息，如标题、描述等
    """
    table_element = table

    # 寻找表格的标题（通常是前面的h1-h6标签）
    title = "Untitled Table"
    prev_element = table_element.find_previous(["h1", "h2", "h3", "h4", "h5", "h6", "caption"])
    if prev_element:
        title = prev_element.get_text().strip()

    # 寻找表格的描述（可能是表格前面的段落）
    description = ""
    prev_p = table_element.find_previous("p")
    if prev_p:
        description = prev_p.get_text().strip()

    return {"title": title, "description": description}


def process_html_table(file_path):
    """
    处理单个HTML文件，提取表格并转换为Markdown
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        tables = extract_tables_from_html(html_content)
        markdown_table_splitter = MarkdownTableSplitter()
        results = []
        for _, table in enumerate(tables):
            markdown_table = html_to_markdown_table(table)
            markdown_table = _simplify_markdown_table(markdown_table)
            context = extract_table_context(table)

            # 构建完整的Markdown文档，包含表格和上下文
            full_markdown = f"# {context['title']}\n\n"
            if context["description"]:
                full_markdown += f"{context['description']}\n\n"
            full_markdown += markdown_table
            markdown_table_texts = markdown_table_splitter.split_document_by_tables(full_markdown, file_path)
            for markdown_table_text in markdown_table_texts:
                results.append(Document(page_content=markdown_table_text, metadata={"source": file_path}))

        return results
    except Exception as e:
        logger.exception("Failed to process html table: %s, exception: %s", file_path, e)
        return []


def _simplify_markdown_table(text: str):
    # 将多个连续的 - 替换为单个 -
    text = re.sub(r'-{2,}', '-', text)
    # 将nbsp替换为空格
    text = text.replace("\u00a0", " ")
    # 将多个连续的空格替换为单个空格
    text = re.sub(r' {2,}', ' ', text)
    return text


def html_to_markdown_table(table):
    # 处理表头
    headers = []
    for th in table.find_all('th'):
        headers.append(th.get_text().replace('\n', ' ').strip())

    # 处理表格内容
    rows = []
    merge_rows_cache = []
    for tr in table.find_all('tr'):
        cells = []
        # 处理合并单元格属性
        merge_rows_cache = [merge_row for merge_row in merge_rows_cache if merge_row[1] > 0]
        for merge_row in merge_rows_cache:
            cells.append(merge_row[0])
            merge_row[1] -= 1

        for td in tr.find_all(['td', 'th']):
            colspan = int(td.get('colspan', 1))
            rowspan = int(td.get('rowspan', 1))

            cell_content = td.get_text().replace('\n', ' ').replace('|', '\|').strip()

            cells.append(cell_content)
            cells.extend([''] * (colspan - 1))
            if rowspan > 1:
                merge_rows_cache.append([cell_content, rowspan - 1])

        if cells and not all(cell == '' for cell in cells):
            rows.extend([cells])

    # 如果没有明确的表头，使用第一行作为表头
    if not headers and rows:
        headers = rows[0]
        rows = rows[1:]

    return build_markdown_table_string(headers, rows)