import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, Field

from rag_service.config import DEFAULT_DATASET_GENERATION_SIZE, DEFAULT_TOP_K, DOCUMENT_THRESHOLD_SCORE
from rag_service.models.database.models import KnowledgeBase
from rag_service.models.enums import (
    EvalDataType,
    GitcodeIssueType,
    KnowledgeBaseType,
    LayerKnowledgeBaseType,
    MemberType,
    QueryStrategy,
    RerankModel,
    Domain,
    ServiceConfigType,
    SecurityLevel,
    AssetType,
)


class OriginalDocument(BaseModel):
    doc_id: UUID
    uri: str
    source: str
    mtime: datetime.datetime
    display_name: Optional[str] = None
    download_key: Optional[str] = None
    extended_metadata: Optional[Dict[str, Any]] = None
    online_url: Optional[str] = None
    security_level: Optional[SecurityLevel] = None


class SplitterConfig(BaseModel):
    splitter: str
    config: Optional[Dict[str, Any]] = {}


class LoaderConfig(BaseModel):
    process_type: str
    loader: str
    splitter_config: Optional[SplitterConfig] = None


class VectorizationConfig(BaseModel):
    loader_configs: Optional[List[LoaderConfig]] = None
    default_splitter_config: Optional[SplitterConfig] = None
    attachment_vectorize: Optional[bool] = False
    local_file_online_dic: Optional[Dict[str, str]] = None
    ebook_asset_config: Optional[Dict[str, str]] = None
    max_documents_num: Optional[int] = None
    gitcode_issue_asset_config: Optional[Dict[str, Any]] = None
    gitcode_file_asset_config: Optional[Dict[str, Any]] = None


class Version(BaseModel):
    label: str
    value: str


class ProductInfo(BaseModel):
    label: str
    value: str
    lang: str
    versions: Optional[List[Version]] = None


class AssetDomain(BaseModel):
    label: str
    value: str


class DocNodeInfo(BaseModel):
    doc_node: str
    updated_at: int
    updated_type: Optional[str] = None


class DocDirInfo(BaseModel):
    url: str
    ancestor: List[str]


class StreamLlmDataToDatabase(BaseModel):
    request_id: str
    load_llm_start_time: Optional[datetime.datetime] = None
    load_llm_first_token_time: datetime.datetime
    load_llm_end_time: datetime.datetime
    llm_return_answer: str


class WikiArticleAndAttachmentDetail(BaseModel):
    source: str
    mtime: float
    uri: str
    sn: str
    is_attachment: bool
    display_name: str
    attachment_path: Optional[str] = None
    wiki_content_html: Optional[dict] = None
    title: str
    author: str
    wiki_url: Optional[str] = None


class WikiDocumentInfo(BaseModel, frozen=True):
    wiki_content: str
    wiki_sn: str
    wiki_title: str
    folder_uri: str
    wiki_author: str
    wiki_url: str


class JiaXianUrlInfo(BaseModel):
    sort_rule: Optional[str] = None
    content_type: str
    zone_id: Optional[str] = None
    daily_briefing: Optional[bool] = None
    article_id: Optional[str] = None


class JiaXianArticleInfo(BaseModel):
    id: str
    content: str
    mtime: datetime.datetime
    title: str
    source: str
    content_source: str


class ThreeMSDataDetail(BaseModel):
    content: str
    source: str
    uri: str
    mtime: int
    display_name: str
    author: Optional[str] = None
    security_level: Optional[SecurityLevel] = None


class ThreeMSUrlParams(BaseModel):
    group_id: str
    category_id: Optional[str] = None
    module_id: Optional[str] = None
    blog_id: Optional[str] = None
    wiki_id: Optional[str] = None


class EvalDataConfig(BaseModel):
    dataset_size: int = DEFAULT_DATASET_GENERATION_SIZE
    kba_name: Optional[str] = None
    kba_id: Optional[str] = None
    top_k: Optional[int] = DEFAULT_TOP_K
    document_source_list: Optional[List[str]] = None
    description: Optional[str] = Field(None, max_length=500)
    rerank: Optional[str] = RerankModel.BASIC.value
    document_score_threshold: Optional[float] = Field(DOCUMENT_THRESHOLD_SCORE, ge=0.0, lt=1.0)
    query_strategy: Optional[str] = QueryStrategy.HYBRID_QUERY.value


class QuestionAnswer(BaseModel):
    question: str
    ground_truth: str
    context: Optional[str] = None
    original_document_id: Optional[UUID] = None
    kba_id: Optional[UUID] = None
    document_name: Optional[str] = None
    source: Optional[str] = None


class EvaluateData(BaseModel):
    data_id: UUID
    question: str
    answer: str
    retrieved_context: List[str]
    ground_truth_context: Optional[str] = None
    ground_truth: str


class EvaluateDataSet(BaseModel):
    question: List[str]
    answer: List[str]
    contexts: List[List[str]]
    ground_truth: List[str]


class EvaluateDatasetMetric(BaseModel):
    hit: Optional[float] = None
    mrr: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_correctness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    answer_similarity: Optional[float] = None


class GeneratedEvaluateData(BaseModel):
    id: UUID
    data_type: EvalDataType
    question: str
    ground_truth: str
    context: Optional[str] = ""
    create_at: datetime.datetime
    update_at: datetime.datetime
    name: str
    source: str
    valid: bool


class UploadedEvaluateData(BaseModel):
    id: UUID
    data_type: EvalDataType
    question: str
    ground_truth: str
    context: Optional[str] = ""
    create_at: datetime.datetime
    update_at: datetime.datetime
    kb_type: KnowledgeBaseType
    layer_kb_type: Optional[LayerKnowledgeBaseType]
    valid: bool


class DetailDatasetMetric(BaseModel):
    question: str
    ground_truth: Optional[str] = None
    context: Optional[str] = None
    answer: str
    contexts: List[str]
    hit: Optional[float] = None
    mrr: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_correctness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    answer_similarity: Optional[float] = None
    kb_sn: str
    kba_name: Optional[str] = None
    document_name: Optional[str] = None
    document_source: Optional[str] = None


class RetrievedDocumentMetadata(BaseModel):
    source: Optional[str] = None
    link: Optional[str] = None
    online_link: Optional[str] = None


class HeadAndContent(BaseModel):
    title: str
    content: str
    headers: Optional[List[str]] = None # 用于检测word和html文档类型切切片被归并至一个切片中的标题层级深度
    header_contents: Optional[List[str]] = None # 用于检测word和html文档类型切切片被归并至一个切片中且正文长度超过阈值的内容


class IDPFileInfo(BaseModel):
    project_id: str
    user_name: str
    number: str
    file_type: str
    url: str
    name: str


class IDPFileContentInfo(BaseModel):
    content_type: str
    user_name: str
    project_id: str
    number: str


class IDPFileChapterTree(BaseModel):
    title: Optional[str]
    chapters: List["IDPFileChapterTree"] = []
    level: int
    number: str


class QaPair(BaseModel):
    question: str
    answer: str


class UploadedQaPair(BaseModel):
    question: str
    answer: str
    extended_questions: Optional[List[str]] = None
    document_url: Optional[str] = None


class CodeHubIssueInfo(BaseModel):
    project_id: str
    issue_id: str
    title: str
    description: str
    url: str
    author: str
    updated_at: datetime.datetime


class VectorStoreElasticSearchQueryInfo(BaseModel):
    vs_indexes: List[str]
    document_close_sources: List[str]


class KnowledgeBaseAndMemberTypeInfo(BaseModel):
    knowledge_base: KnowledgeBase
    member_type: MemberType


class DatasetMetricAndDataInfo(BaseModel):
    hit: Optional[float]
    answer_correctness: Optional[float]
    question: str
    context: Optional[str]
    ground_truth: Optional[str]
    answer: str
    contexts: Optional[List[str]]
    note: Optional[str]
    manual_score: Optional[float]
    valid: bool
    metric_id: UUID
    data_id: UUID

    @classmethod
    def from_list(cls, data_list: List[Tuple]):
        return [cls(**{key: data[i] for i, key in enumerate(cls.__fields__.keys())}) for data in data_list]


class DeptInfo(BaseModel):
    l1_dept_code: Optional[str] = None
    l1_name: Optional[str] = None
    l2_dept_code: Optional[str] = None
    l2_name: Optional[str] = None
    l3_dept_code: Optional[str] = None
    l3_name: Optional[str] = None
    l4_dept_code: Optional[str] = None
    l4_name: Optional[str] = None
    l5_dept_code: Optional[str] = None
    l5_name: Optional[str] = None
    l6_dept_code: Optional[str] = None
    l6_name: Optional[str] = None
    l7_dept_code: Optional[str] = None
    l7_name: Optional[str] = None
    l8_dept_code: Optional[str] = None
    l8_name: Optional[str] = None
    l9_dept_code: Optional[str] = None
    l9_name: Optional[str] = None


class SegmentRawInfo(BaseModel):
    content: str
    kb_sn: Optional[str] = None
    asset_name: Optional[str] = None
    source: Optional[str] = None
    doc_id: Optional[str] = None


class SegmentComparisonRawInfo(BaseModel):
    content: str
    doc_id: Optional[str] = None
    kb_sn: Optional[str] = None
    asset_name: Optional[str] = None
    source: Optional[str] = None
    judge_reason: Optional[str] = None
    es_doc_id: Optional[str] = None


class PerDuplicateSegment(BaseModel):
    duplicate_sentences: Optional[List[str]] = []
    duplicate_segment_content: Optional[str] = None


class DuplicateSegmentInfo(BaseModel):
    duplicate_segment: Optional[Dict[int, PerDuplicateSegment]] = {}
    kb_sn: Optional[str] = None
    asset_name: Optional[str] = None
    source: Optional[str] = None
    doc_id: Optional[str] = None


class OutdatedDocInfo(BaseModel):
    doc_id: Optional[str] = None
    kba_id: Optional[str] = None
    kb_id: Optional[str] = None


class DetectionMetadata(BaseModel):
    operation: Optional[str] = None
    segment_row_info: Optional[SegmentRawInfo] = None
    segment_comparison_raw_info: Optional[SegmentComparisonRawInfo] = None
    duplicate_segment_info: Optional[DuplicateSegmentInfo] = None
    outdated_doc_info: Optional[OutdatedDocInfo] = None


class DetectionInfo(BaseModel):
    rule_code: int
    metadata: Optional[DetectionMetadata] = None


class DetectionShowInfo(DetectionInfo):
    kb_sn: Optional[str] = None
    asset_name: Optional[str] = None
    doc_name: Optional[str] = None
    operation: Optional[str] = None


class KnowledgeBaseMigrateInfo(BaseModel):
    kb_sn: str
    owner: str
    dept_name: Optional[str] = None
    is_migrated: bool
    allow_synchronous_update: bool


class UrlAccessStatusInfo(BaseModel):
    url: Optional[str] = None
    accessible: Optional[bool] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    error: Optional[str] = None


class DeptTreeNode(BaseModel):
    dept_code: str
    dept_name: str
    has_doc_flag: Optional[bool] = True
    children: Optional[List["DeptTreeNode"]] = None


class HuaweiCaseInfo(BaseModel):
    case_info_id: str
    name: str
    url: str
    uri: str
    content: str
    author: str
    updated_at: datetime.datetime


class VersionAllNodeInfo(BaseModel):
    name: str
    path: Optional[str] = None
    children: Optional[List["VersionAllNodeInfo"]] = None


class GitcodeIssueInfo(BaseModel):
    issue_num: str
    title: str
    description: str
    url: str
    author: str
    updated_at: datetime.datetime


class GitcodeIssueCommentInfo(BaseModel):
    issue_num: str
    comment: str


class GitcodeAssetConfig(BaseModel):
    issue_type: GitcodeIssueType
    issue_created_after: Optional[datetime.datetime] = None
    filter_accounts: Optional[List[str]] = []
    include_open_issue: Optional[bool] = False
    issue_count_limit: Optional[int] = None
    last_update_time: Optional[datetime.datetime] = None


class KbaCopyConfig(BaseModel):
    source_kb_sn: str
    source_asset_name: str
    target_kb_sn: str
    target_asset_name: str


class PdmPartDocument(BaseModel):
    uri: str
    source: str
    mtime: datetime.datetime
    download_key: str


class SupportProductInfo(BaseModel):
    # 产品id
    pid: str
    # 产品名称
    name: str
    # 领域id
    domain_id: Optional[str] = None


class SupportDocInfo(BaseModel):
    # 文档id
    doc_id: str
    # 文档名称
    doc_name: str
    # 文档发布时间
    release_date: datetime.datetime
    # 文档预览地址
    view_url: str


class TslNode(BaseModel):
    """
    技术标准规范库节点
    """
    # 节点id
    node_id: int
    # 节点名称
    node_name: str
    # 是否有子节点
    has_children: bool
    # 子节点
    children: List['TslNode'] = []


class TslDoc(BaseModel):
    """
    技术标准规范库文档
    """
    # 标准规范id
    standard_id: int
    # 文档id
    doc_id: str
    # 文档名称
    doc_name: str
    # 更新时间
    update_time: datetime.datetime


class ResourceNumberConfigureShowInfo(BaseModel):
    # 配置id
    config_id: UUID
    # 部门编码
    dept_code: Optional[str] = None
    # 部门名称
    dept_name: Optional[str] = None
    # 领域
    domain: Optional[Domain] = None
    # 知识库序列号
    kb_sn: Optional[str] = None
    # 资产名称
    asset_name: Optional[str] = None
    # 配置的资源数量
    number: int
    # 配置类型
    config_type: ServiceConfigType


class AddResourceNumberConfigureReq(BaseModel):
    # 部门编码
    dept_code: Optional[str] = None
    # 领域
    domain: Optional[Domain] = None
    # 知识库序列号
    kb_sn: Optional[str] = None
    # 资产名称
    asset_name: Optional[str] = None
    # 配置的资源数量
    number: int = Field(ge=1)
    # 配置类型
    config_type: ServiceConfigType


class UpdateResourceNumberConfigureReq(BaseModel):
    # 配置id
    config_id: UUID
    # 配置的资源数量
    number: int = Field(ge=1)


class SmallSpiritAttach(BaseModel):
    """
    2012小巧灵突击队文档附件
    """
    # 附件名称
    file_name: str
    # 附件id
    file_id: str


class SmallSpiritDoc(BaseModel):
    """
    2012小巧灵突击队文档
    """
    # 发布时间，时间戳，单位秒
    publish_time: int
    # 修改时间，时间戳，单位秒
    edit_time: int
    # 文档id
    content_id: str
    # 文档作者
    author: str
    # 文档标题
    title: str
    # 文档内容
    content: str
    # 文档附件
    attachments: List[SmallSpiritAttach] = []


class EsDocumentChunk(BaseModel):
    index: str
    doc_id: str
    content: Dict[Any, Any]


class EsDocumentEmbeddingResult(BaseModel):
    index: str
    doc_id: str
    embedding: List[float]


class SpilloverShowInfo(BaseModel):
    """
    知识外溢管理页面展示vo
    """
    id: int
    kb_sn: str
    user_id: str
    url_list: List[str]
    created_at: datetime.datetime


class SpilloverDocShowInfo(BaseModel):
    """
    知识外溢管理页面文档清单vo
    """
    id: UUID
    asset_name: str
    asset_type: AssetType
    doc_name: str
    doc_source: str
    security_level: Optional[SecurityLevel] = None


class IpdRAGKnowledgeBaseInfo(BaseModel):
    """
    ipd rag的知识库
    """
    # ipd rag知识库id
    id: str
    # ipd rag知识库名称
    name: str
    # 雅典娜知识集id
    athena_id: Optional[str] = None
    # 雅典娜知识集跳转链接
    athena_url: Optional[str] = None