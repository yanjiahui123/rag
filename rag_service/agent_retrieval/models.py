from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SearchSlicesRequest(BaseModel):
    query: str
    kb_sn: Optional[str] = None
    kb_sn_list: List[str] = Field(default_factory=list)
    top_k: int = 20
    include_refs: bool = True
    include_artifact_handles: bool = True


class DocumentSummary(BaseModel):
    doc_id: Optional[str] = None
    title: Optional[str] = None
    source: str = ""
    asset_name: Optional[str] = None
    kb_sn: Optional[str] = None


class SliceLocation(BaseModel):
    block_id: Optional[str] = None
    block_index: Optional[int] = None
    block_type: Optional[str] = None
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    table_id: Optional[str] = None
    row_range: Optional[List[int]] = None


class SliceActions(BaseModel):
    can_get_section: bool = False
    can_get_table: bool = False
    can_get_original_text: bool = False


class SliceHandles(BaseModel):
    section_handle: Optional[str] = None
    table_handle: Optional[str] = None
    document_handle: Optional[str] = None


class SearchSlice(BaseModel):
    slice_id: str
    rank: int
    text: str
    score: float = 0.0
    doc: DocumentSummary
    location: SliceLocation
    actions: SliceActions = Field(default_factory=SliceActions)
    handles: SliceHandles = Field(default_factory=SliceHandles)
    es_index: Optional[str] = None
    es_doc_id: Optional[str] = None


class SearchSlicesResponse(BaseModel):
    query: str
    kb_sn_list: List[str]
    slices: List[SearchSlice] = Field(default_factory=list)


class DocumentOutlineRequest(BaseModel):
    document_handle: str


class OutlineSection(BaseModel):
    section_id: str
    title: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    block_count: int = 0
    section_handle: Optional[str] = None


class OutlineTable(BaseModel):
    table_id: str
    title: Optional[str] = None
    row_count: Optional[int] = None
    table_handle: Optional[str] = None


class DocumentOutlineResponse(BaseModel):
    doc_id: Optional[str] = None
    title: Optional[str] = None
    sections: List[OutlineSection] = Field(default_factory=list)
    tables: List[OutlineTable] = Field(default_factory=list)


class SectionRequest(BaseModel):
    section_handle: str
    max_chars: int = 12000


class SectionResponse(BaseModel):
    section_id: Optional[str] = None
    title: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    mode: str = "full_section"
    text: str = ""
    truncated: bool = False


TableMode = Literal["llm_text", "json", "html", "summary"]


class TableRequest(BaseModel):
    table_handle: str
    mode: TableMode = "llm_text"
    max_chars: int = 40000


class TableResponse(BaseModel):
    table_id: Optional[str] = None
    title: Optional[str] = None
    mode: str
    row_count: Optional[int] = None
    content: str = ""
    truncated: bool = False


class OriginalTextRequest(BaseModel):
    document_handle: str
    center_block_id: Optional[str] = None
    before: int = 2
    after: int = 2
    max_chars: int = 12000


class OriginalTextBlock(BaseModel):
    block_id: Optional[str] = None
    block_index: Optional[int] = None
    text: str = ""


class OriginalTextResponse(BaseModel):
    doc_id: Optional[str] = None
    mode: str = "neighbor_blocks"
    blocks: List[OriginalTextBlock] = Field(default_factory=list)
    truncated: bool = False


class SkillPackageResponse(BaseModel):
    name: str = "kb-retrieval"
    files: Dict[str, str] = Field(default_factory=dict)

