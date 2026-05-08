from __future__ import annotations

from html.parser import HTMLParser
from typing import Dict, Iterable, List, Union

from pydantic import BaseModel, Field


HtmlChild = Union["HtmlNode", str]
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}


class HtmlNode(BaseModel):
    tag: str
    attrs: Dict[str, str] = Field(default_factory=dict)
    children: List[HtmlChild] = Field(default_factory=list)

    def text(self) -> str:
        return _normalize_text(" ".join(_child_text(child) for child in self.children))


class _TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = HtmlNode(tag="document")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs):
        node = HtmlNode(tag=tag.lower(), attrs={key: value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if node.tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str):
        if data:
            self.stack[-1].children.append(data)


def parse_html(html: str) -> HtmlNode:
    parser = _TreeBuilder()
    parser.feed(html)
    parser.close()
    return parser.root


def find_first(node: HtmlNode, tag: str) -> HtmlNode | None:
    for child in iter_nodes(node):
        if child.tag == tag:
            return child
    return None


def iter_nodes(node: HtmlNode) -> Iterable[HtmlNode]:
    for child in node.children:
        if isinstance(child, HtmlNode):
            yield child
            yield from iter_nodes(child)


def _child_text(child: HtmlChild) -> str:
    if isinstance(child, HtmlNode):
        return child.text()
    return child


def _normalize_text(text: str) -> str:
    return " ".join(text.split())
