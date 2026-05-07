import re
from typing import List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString, PageElement, Tag
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders.base import BaseLoader

from rag_service.constants import PROGRAMMING_LANGUAGES
from rag_service.document_loaders import html_table_phrase
from rag_service.document_loaders.loader_utils import (
    convert_and_split_head_content_to_documents,
    convert_head_content_to_documents,
)
from rag_service.models.generic.models import HeadAndContent


def format_table_data(headers, data):
    result = []
    for row in data:
        row_data = [f"{header}: {value}" for header, value in zip(headers, row)]
        result.append("，".join(row_data))
    return "；".join(result) + "。"


def _process_a_tag(a_element: Tag, source: Optional[str] = None) -> str:
    href = a_element.get("href", "")
    text = a_element.get_text(strip=True)
    if not href:
        return text
    if source and not href.startswith("http"):
        parsed_url = urlparse(source)
        path = parsed_url.path
        path_parts = path.split("/")
        base_path = "/".join(path_parts[:-1]) + "/"
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{base_path}"
        href = base_url + href
    if text and text != href:
        return f"{text} {href}"
    return href


def _process_header_rows(head_rows: List[List[Tag]]) -> Optional[List[List[str]]]:
    head_row_spans = []
    max_cols = 0
    process_header_rows = []

    for head_row in head_rows:
        current_row_headers = []
        col_index = 0
        for cell_th in head_row:
            while col_index < len(head_row_spans) and head_row_spans[col_index] > 0:
                current_row_headers.append("")
                head_row_spans[col_index] -= 1
                col_index += 1
            colspan = int(cell_th.get("colspan", 1))
            rowspan = int(cell_th.get("rowspan", 1))
            text = cell_th.get_text(strip=True)

            current_row_headers.extend([text] * colspan)
            head_row_spans[col_index : col_index + colspan] = [rowspan - 1] * colspan
            col_index += colspan
            max_cols = max(max_cols, len(current_row_headers))
        # 处理最后一个未处理的rowspan
        while len(current_row_headers) < max_cols:
            current_row_headers.append("")
        if current_row_headers:
            process_header_rows.append(current_row_headers)

    return process_header_rows


def _get_table_header_rows(rows: List[Tag]) -> Optional[List[List[str]]]:
    header_rows = []
    first_row_td_count = len(rows[0].find_all("td"))
    if first_row_td_count > 0:
        return []
    for row in rows:
        header_row = row.find_all("th")
        if not header_row:
            break
        header_rows.append(header_row)
    return _process_header_rows(header_rows)


def _process_table_tag(table: Tag, source: Optional[str] = None) -> str:
    thead = table.find("thead")
    tbody = table.find("tbody")
    rows = table.find_all("tr")
    if not rows:
        return "".join([_process_element(element, source) for element in table.contents])
    if thead and tbody:
        unhandled_headers_rows = [row.find_all("td") or row.find_all("th") for row in thead.find_all("tr")]
        headers_rows = _process_header_rows(unhandled_headers_rows)
        data_rows = tbody.find_all("tr")
    else:
        # 一般情况获得表头
        headers_rows = _get_table_header_rows(rows)
        data_rows = rows[len(headers_rows) :]

    # 处理无表头或者竖向表头的情况
    if not headers_rows:
        return _process_no_head_or_vertical_head(rows, source)

    # 初始化 headers为headers_rows 的第一个行
    headers = headers_rows[0]
    # 可能存在多行表头的情况，用-连接作为最终的表头名
    for headers_row in headers_rows[1:]:
        if len(headers) != len(headers_row):
            return ""
        headers = [f"{header}-{col_head}" if col_head else header for header, col_head in zip(headers, headers_row)]

    return _process_horizontal_head_and_data(headers, data_rows, source)


def _process_horizontal_head_and_data(headers: List[str], data_rows: List[Tag], source: Optional[str] = None) -> str:
    data = []
    row_spans = [0] * len(headers)
    for row in data_rows:
        row_data = []
        col_index = 0
        for cell in row.find_all("td"):
            while row_spans[col_index] > 0:
                row_data.append(data[-1][col_index])
                row_spans[col_index] -= 1
                col_index += 1
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            cell_value = _process_element(cell, source).replace("\n", " ")

            row_data.extend([cell_value] * colspan)
            row_spans[col_index : col_index + colspan] = [rowspan - 1] * colspan
            col_index += colspan
        # 处理最后一个单元格之后的span row
        while len(row_data) < len(headers):
            row_data.append(data[-1][col_index])
            row_spans[col_index] -= 1
            col_index += 1
        data.append(row_data)
    return format_table_data(headers, data)


def _get_vertical_headers(table_rows: List[Tag]) -> List[str]:
    headers = []
    for row in table_rows:
        first_cell = row.find("th")
        if not first_cell:
            break
        headers.append(first_cell.get_text(strip=True))
    return headers


def _process_default_horizontal_table(table_rows: List[Tag], source: Optional[str] = None) -> str:
    header_rows = [table_rows[0].find_all("td")]
    for cell in header_rows[0]:
        rowspan = int(cell.get("rowspan", 1))
        if rowspan == 1:
            continue
        if rowspan == 2:
            header_rows.append(table_rows[1].find_all("td"))
            break
        if rowspan > 2:
            return get_table_data_text(table_rows, source)
    process_header_rows = _process_header_rows(header_rows)
    return _process_horizontal_head_and_data(process_header_rows[0], table_rows[len(header_rows) :], source)


def _append_vertical_row_data(data, row_spans, cells, source):
    col_index = 0
    for cell in cells[1:]:
        while row_spans[col_index] > 0:
            data[col_index].append(data[col_index][-1])
            row_spans[col_index] -= 1
            col_index += 1
        rowspan = int(cell.get("rowspan", 1))
        cell_value = _process_element(cell, source).replace("\n", " ")
        data[col_index].append(cell_value)
        row_spans[col_index] = rowspan - 1
        col_index += 1
    return col_index


def _process_vertical_head_table(headers: List[str], table_rows: List[Tag], source: Optional[str] = None) -> str:
    col_num = len(table_rows[0].find_all(["td", "th"])) - 1
    data = [[] for _ in range(col_num)]
    row_spans = [0] * col_num
    for row in table_rows:
        cells = row.find_all(["td", "th"])
        col_index = _append_vertical_row_data(data, row_spans, cells, source)
        while col_index < col_num:
            data.append(data[col_index][-1])
            row_spans[col_index] -= 1
            col_index += 1
    return format_table_data(headers, data)


def _process_no_head_or_vertical_head(table_rows: List[Tag], source: Optional[str] = None) -> str:
    if len(table_rows) == 1:
        return _process_element(table_rows[0], source)

    # 对于首行为单列的表格，直接逐行返回文本
    first_row_cells = table_rows[0].find_all(["td", "th"])
    if len(first_row_cells) <= 1:
        return get_table_data_text(table_rows, source)

    headers = _get_vertical_headers(table_rows)
    # 对于无横向且无竖向表头，选择默认表头
    if not headers or len(headers) != len(table_rows):
        return _process_default_horizontal_table(table_rows, source)

    return _process_vertical_head_table(headers, table_rows, source)


def get_table_data_text(table_rows: List[Tag], source: Optional[str] = None) -> str:
    result = []
    for row in table_rows:
        row_data = [_process_element(cell, source).replace("\n", " ") for cell in row.find_all(["td", "th"])]
        result.append("，".join(row_data))
    return "；".join(result) + "。"


def _process_head_tag(page_element: Tag, source: Optional[str] = None) -> HeadAndContent:
    title = ""
    content = ""
    for child in page_element.contents:
        if isinstance(child, str) and not title:
            title = child
        elif isinstance(child, str) and title:
            content += child
        elif not title:
            title = _process_element(child, source)
        else:
            content += _process_element(child, source)

    return HeadAndContent(title=title, content=content)


def _is_base64_img(text: str) -> bool:
    img_base64_pattern = re.compile(r'^data:image/([a-zA-Z0-9+\-]+);base64,', re.IGNORECASE)
    if img_base64_pattern.match(text):
        return True
    else:
        return False


def _process_img_tag(image_tag: Tag) -> str:
    src = image_tag.get("src", "").strip()
    if not src:
        return ""

    # 检查是否为base64 data url
    if _is_base64_img(src):
        return ""

    # 检查是否为完整url
    if src.startswith(('http://', 'https://', '//')):
        return f"![]({src})"
    return ""




def _process_element(page_element: Tag, source: Optional[str] = None) -> str:
    if not page_element:
        return ""

    if isinstance(page_element, NavigableString):
        return page_element.strip()

    res = ""
    tag_name = page_element.name
    if tag_name == "br":
        return "\n"

    if tag_name == "table":
        return "\n" + _process_table_tag(page_element, source)

    if tag_name == "a":
        return _process_a_tag(page_element, source)

    if tag_name == "img":
        return _process_img_tag(page_element)


    if tag_name == "span":
        return page_element.get_text()

    if isinstance(page_element, Tag):
        contents = page_element.contents
    else:
        return page_element.get_text(strip=True)

    for element in contents:
        if not element:
            continue
        res += _process_element(element, source)
    if tag_name in ["li", "p", "div"] and not res.endswith("\n"):
        res += "\n"
    return res


class HTMLLoaderByHead(BaseLoader):
    def __init__(self, file_path: str, source: Optional[str] = None):
        self.file_path = file_path
        self.current_title_path = [""] * 6
        self.source = source

    def load(self) -> list[Document]:
        all_content = self._parse_content()
        documents = convert_head_content_to_documents(all_content, self.file_path)
        tables = html_table_phrase.process_html_table(self.file_path)
        if tables:
            documents.extend(tables)
        return documents

    def load_and_split(self, text_splitter: Optional[TextSplitter] = None) -> List[Document]:
        all_content = self._parse_content()
        documents = convert_and_split_head_content_to_documents(all_content, self.file_path, text_splitter)
        tables = html_table_phrase.process_html_table(self.file_path)
        if tables:
            documents.extend(tables)
        return documents

    def get_head_contents(self) -> List[HeadAndContent]:
        return self._parse_content()

    def _get_merge_title(self) -> str:
        if not self.current_title_path:
            return ""
        title = self.current_title_path[0]
        for head in self.current_title_path[1:]:
            title += "-" + head if head else ""
        return title

    def _parse_content(self) -> List[HeadAndContent]:
        all_content = []
        with open(self.file_path, encoding="utf-8") as file:
            html_content = file.read()
        soup = html_codeblocks_to_markdown(html_content)
        contents = soup.body.contents if soup.body else soup.contents
        contents = self.convert_div_to_contents(contents)
        content = ""
        for element_tag in contents:
            if isinstance(element_tag, (NavigableString, str)):
                # 避免连续换行
                if element_tag == "\n" and content.endswith("\n"):
                    continue
                content += element_tag
                continue

            tag_name = element_tag.name
            if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                head_content = _process_head_tag(element_tag, self.source)

                # 遇到新的标题，先把前面的内容存储，然后重新开始
                if content and content.strip():
                    title = self._get_merge_title()
                    all_content.append(HeadAndContent(title=title, content=content))

                # 开始新的标题和内容
                content = head_content.content
                index = int(tag_name[1])
                self.current_title_path = self.current_title_path[: index - 1]
                self.current_title_path.append(head_content.title)
                self.current_title_path.extend([""] * (6 - index))
                continue
            content += _process_element(element_tag, self.source)

        # 结束，把最后一个标题和内容存储
        title = self._get_merge_title()
        all_content.append(HeadAndContent(title=title, content=content))
        return all_content

    def convert_div_to_contents(self, contents: List[PageElement]) -> List[PageElement]:
        result_contents = []
        for element_tag in contents:
            if isinstance(element_tag, NavigableString):
                new_content = element_tag.replace("\xa0", " ")
                result_contents.append(new_content)
                continue
            if isinstance(element_tag, Tag) and element_tag.name == "div":
                result_contents.extend(self.convert_div_to_contents(element_tag.contents))
                result_contents.append(NavigableString("\n"))
                continue
            result_contents.append(element_tag)
        return result_contents


def get_language(classes):
    language = ''
    for cls in classes:
        if cls.startswith('language-'):
            language = cls[9:]
        elif cls.startswith('lang-'):
            language = cls[5:]
        elif cls in PROGRAMMING_LANGUAGES:
            language = cls
    return language


def get_code_text(code):
    if code.find_all('li'):
        return '\n'.join(li.get_text() for li in code.find_all('li'))
    else:
        return code.get_text()


def html_codeblocks_to_markdown(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for pre in soup.find_all('pre'):
        codes = pre.find_all('code')
        if codes:
            language = get_language(codes[0].get('class', []))
            code_text = '\n'.join(get_code_text(code) for code in codes)
            code_block = f"```{language}\n{code_text}\n```"
            pre.replace_with(code_block)
    # 处理行内代码
    for code in soup.find_all('code'):
        if not any(p.name == 'pre' for p in code.parents):
            code_text = code.get_text()
            code.replace_with(f"`{code_text}`")
    return soup
