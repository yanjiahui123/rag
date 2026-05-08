import datetime
import uuid
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import UniqueConstraint, SmallInteger, BigInteger
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlmodel import JSON, Column, DateTime, Field, Relationship, SQLModel

from rag_service.constants import DEFAULT_UPDATE_TIME_INTERVAL_SECOND
from rag_service.models.enums import (
    Analyzer,
    AssetStatus,
    AssetType,
    CommonKnowledgeBaseType,
    DefaultKbStatus,
    DocumentLoadStatus,
    Domain,
    EmbeddingModel,
    EvalDataType,
    GenerationEvaluationJobType,
    JobStatus,
    KnowledgeBaseType,
    LayerKnowledgeBaseType,
    MemberType,
    QaPairStatus,
    QaPairType,
    RetrieveStatus,
    StopWordOrSynonymType,
    UpdateOriginalDocumentType,
    VectorizationJobType,
    ServiceConfigType,
    SecurityLevel,
)
from rag_service.utils.time_util import default_next_update_at, now_with_time_zone


class ServiceConfig(SQLModel, table=True):
    __tablename__ = "service_config"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True)
    value: str
    config_type: Optional[ServiceConfigType]


class KnowledgeBase(SQLModel, table=True):
    __tablename__ = "knowledge_base"
    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str
    sn: str = Field(unique=True)  # 知识库名唯一标识
    owner: str
    created_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    updated_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )
    kb_type: KnowledgeBaseType
    layer_kb_type: Optional[LayerKnowledgeBaseType]
    common_dept_code: Optional[str]
    common_kb_type: Optional[CommonKnowledgeBaseType]
    domain: Optional[Domain]
    pbi_project_id: Optional[str]
    analyzer: Optional[Analyzer]
    config: Optional[Dict[Any, Any]] = Field(default={}, sa_column=Column(JSON))
    ipd_rag_kb_id: Optional[str]  # 知识库迁移到IPD_RAG后对应的知识库id
    is_default: Optional[bool]
    default_status: Optional[DefaultKbStatus]
    allow_synchronous_update: Optional[bool] = False
    is_migrated: Optional[bool] = False
    dept_no_tree: Optional[str] = None
    ipd_rag_kb_sn: Optional[str] = None  # 知识库迁移到IPD_RAG后对应的知识库集合序列号
    athena_kb_id: Optional[str] = None  # 对应的雅典娜知识集id
    knowledge_base_assets: List["KnowledgeBaseAsset"] = Relationship(
        back_populates="knowledge_base", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    knowledge_base_members: List["KnowledgeBaseMember"] = Relationship(
        back_populates="member_knowledge_base", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    eval_dataset_jobs: List["EvalDatasetJob"] = Relationship(
        back_populates="knowledge_base", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    eval_datasets: List["EvalDataset"] = Relationship(
        back_populates="knowledge_base", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    knowledge_base_stopword_and_synonyms: List["KnowledgeBaseStopwordAndSynonym"] = Relationship(
        back_populates="knowledge_base", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    knowledge_base_favorites: List["KnowledgeBaseFavorites"] = Relationship(
        back_populates="knowledge_base", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    knowledge_base_blacklist_members: List["KnowledgeBaseBlacklistMember"] = Relationship(
        back_populates="knowledge_base", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    document_defects: List["DocumentDefect"] = Relationship(
        back_populates="knowledge_base", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class KnowledgeBaseMember(SQLModel, table=True):
    __tablename__ = "knowledge_base_member"
    __table_args__ = (UniqueConstraint("kb_id", "employee_number"),)

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    employee_number: str
    employee_name: Optional[str]
    member_type: MemberType
    kb_id: UUID = Field(foreign_key="knowledge_base.id")
    member_knowledge_base: KnowledgeBase = Relationship(
        back_populates="knowledge_base_members",
    )


class KnowledgeBaseAsset(SQLModel, table=True):
    __tablename__ = "knowledge_base_asset"
    __table_args__ = (UniqueConstraint("kb_id", "name", name="knowledge_base_asset_name_uk"),)

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str
    label: str
    asset_type: AssetType
    asset_status: AssetStatus = Field(default=AssetStatus.OPEN)
    asset_uri: Optional[str]
    allow_synchronous_update: Optional[bool] = False
    embedding_model: Optional[EmbeddingModel]
    vectorization_config: Optional[Dict[Any, Any]] = Field(default={}, sa_column=Column(JSON))
    save_file: Optional[bool]
    created_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    updated_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )

    kb_id: UUID = Field(foreign_key="knowledge_base.id")
    knowledge_base: KnowledgeBase = Relationship(
        back_populates="knowledge_base_assets",
    )

    vector_stores: List["VectorStore"] = Relationship(
        back_populates="knowledge_base_asset", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    automated_job_schedule: Optional["AutomatedJobSchedule"] = Relationship(
        sa_relationship_kwargs={"uselist": False, "cascade": "all, delete-orphan"},
        back_populates="knowledge_base_asset",
    )
    auto_job_instances: List["AutoJobInstances"] = Relationship(
        back_populates="knowledge_base_asset", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    online_qa_pairs: List["OnlineQaPair"] = Relationship(
        back_populates="knowledge_base_asset", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    document_defects: List["DocumentDefect"] = Relationship(
        back_populates="knowledge_base_asset", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class VectorStore(SQLModel, table=True):
    __tablename__ = "vector_store"
    __table_args__ = (UniqueConstraint("kba_id", "name", name="vector_store_name_uk"),)

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str
    created_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    updated_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )

    original_documents: List["OriginalDocument"] = Relationship(
        back_populates="vector_store", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )

    kba_id: UUID = Field(foreign_key="knowledge_base_asset.id")
    knowledge_base_asset: KnowledgeBaseAsset = Relationship(
        back_populates="vector_stores",
    )


class OriginalDocument(SQLModel, table=True):
    __tablename__ = "original_document"
    __table_args__ = (UniqueConstraint("vs_id", "uri", name="kb_doc_uk"),)

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    uri: str = Field(index=True)
    source: str
    online_url: Optional[str]
    mtime: datetime.datetime = Field(sa_column=Column(DateTime(timezone=True), index=True))
    download_key: Optional[str]
    display_name: Optional[str]
    retrieve_status: RetrieveStatus = Field(default=RetrieveStatus.OPEN)
    document_load_status: DocumentLoadStatus = Field(default=DocumentLoadStatus.LOADING)
    error_info: Optional[str]
    extended_metadata: Optional[Dict[str, Any]] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    updated_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )
    ipd_rag_document_list: Optional[List[Dict[str, str]]] = Field(
        default=None, sa_column=Column(ARRAY(JSONB()), nullable=True, default=None)
    )
    has_defect: Optional[bool] = Field(default=False)
    security_level: Optional[SecurityLevel]
    vs_id: UUID = Field(foreign_key="vector_store.id")
    allow_spill_over: Optional[bool] = Field(default=False)
    is_detect: Optional[bool] = Field(default=True)
    vector_store: VectorStore = Relationship(
        back_populates="original_documents",
    )
    document_defects: List["DocumentDefect"] = Relationship(
        back_populates="original_document",
        sa_relationship_kwargs={"foreign_keys": "DocumentDefect.doc_id", "cascade": "all, delete-orphan"},
    )
    compare_document_defects: List["DocumentDefect"] = Relationship(
        back_populates="compare_original_document",
        sa_relationship_kwargs={"foreign_keys": "DocumentDefect.compare_doc_id", "cascade": "all, delete-orphan"},
    )


class AutomatedJobSchedule(SQLModel, table=True):
    __tablename__ = "automated_job_schedule"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    updated_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )
    last_updated_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True))
    )
    updated_cycle: datetime.timedelta = Field(
        default=datetime.timedelta(seconds=DEFAULT_UPDATE_TIME_INTERVAL_SECOND),
    )
    next_updated_at: datetime.datetime = Field(
        default_factory=default_next_update_at, sa_column=Column(DateTime(timezone=True))
    )

    kba_id: UUID = Field(foreign_key="knowledge_base_asset.id")
    knowledge_base_asset: KnowledgeBaseAsset = Relationship(back_populates="automated_job_schedule")

    extra_info: Optional[Dict[Any, Any]] = Field(default={}, sa_column=Column(MutableDict.as_mutable(JSON)))


class AutoJobInstances(SQLModel, table=True):
    __tablename__ = "auto_job_instances"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    status: JobStatus
    job_type: VectorizationJobType
    created_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    updated_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )
    corpus_access_interception: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    knowledge_id: UUID = Field(foreign_key="knowledge_base_asset.id")
    knowledge_base_asset: KnowledgeBaseAsset = Relationship(back_populates="auto_job_instances")

    updated_original_documents: List["UpdatedOriginalDocument"] = Relationship(
        back_populates="auto_job_instances", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )

    extra_info: Optional[Dict[Any, Any]] = Field(default={}, sa_column=Column(MutableDict.as_mutable(JSON)))


class UpdatedOriginalDocument(SQLModel, table=True):
    __tablename__ = "updated_original_document"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    update_type: UpdateOriginalDocumentType
    source: str
    display_name: Optional[str]
    created_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))

    job_id: UUID = Field(foreign_key="auto_job_instances.id")
    auto_job_instances: AutoJobInstances = Relationship(back_populates="updated_original_documents")


class RequestResponseLog(SQLModel, table=True):
    __tablename__ = "request_response_log"

    request_id: str = Field(primary_key=True)
    aigc_record_id: Optional[str]
    user_id: str
    method_name: str
    kb_sn: Optional[str]
    question: str
    request_start_time: Optional[datetime.datetime] = Field(sa_column=Column(DateTime(timezone=True)))
    request_end_time: Optional[datetime.datetime] = Field(sa_column=Column(DateTime(timezone=True)))
    request_time_use: Optional[float]
    retrieve_start_time: Optional[datetime.datetime] = Field(sa_column=Column(DateTime(timezone=True)))
    retrieve_end_time: Optional[datetime.datetime] = Field(sa_column=Column(DateTime(timezone=True)))
    retrieve_time_use: Optional[float]
    retrieve_result: Optional[str]
    load_non_stream_llm_start_time: Optional[datetime.datetime] = Field(sa_column=Column(DateTime(timezone=True)))
    load_non_stream_llm_end_time: Optional[datetime.datetime] = Field(sa_column=Column(DateTime(timezone=True)))
    load_non_stream_llm_time_use: Optional[float]
    load_non_stream_llm_prompt: Optional[str]
    load_non_stream_llm_result: Optional[str]
    load_stream_llm_start_time: Optional[datetime.datetime] = Field(sa_column=Column(DateTime(timezone=True)))
    load_stream_llm_end_time: Optional[datetime.datetime] = Field(sa_column=Column(DateTime(timezone=True)))
    load_stream_llm_first_token_time: Optional[datetime.datetime] = Field(sa_column=Column(DateTime(timezone=True)))
    load_stream_llm_first_token_time_use: Optional[float]
    load_stream_llm_time_use: Optional[float]
    load_stream_llm_prompt: Optional[str]
    load_stream_llm_result: Optional[str]
    answer_user_want: Optional[str]
    answer_source: Optional[str]
    error_reason: Optional[str]
    acceptance: Optional[bool]
    score: Optional[float]
    extra_info: Optional[Dict[Any, Any]] = Field(default=None, sa_column=Column(JSONB))
    rewrite_question: Optional[str]
    question_id: Optional[str]
    dislike_tag: Optional[int] = None
    ungoverned_corpus_links: Optional[str] = None
    process_status: Optional[bool] = None
    dept_no: Optional[str] = None
    dept_no_tree: Optional[str] = None


class RequestDocumentHit(SQLModel, table=True):
    """
    请求命中文档信息记录
    """
    __tablename__ = "request_document_hit"

    id: Optional[int] = Field(sa_column=Column(BigInteger, primary_key=True, autoincrement=True))
    request_id: str = Field(max_length=255, nullable=False, foreign_key="request_response_log.request_id",
                            description='请求id，request_response_log表中的request_id')
    doc_id: Optional[UUID] = Field(foreign_key="original_document.id")
    sort: int = Field(sa_column=Column(SmallInteger, nullable=False), description='命中的排序')
    kb_id: UUID = Field(foreign_key="knowledge_base.id", description='冗余归属知识库id')
    kba_id: UUID = Field(foreign_key="knowledge_base_asset.id", description='冗余归属资产id')
    create_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)),
                                         description='创建时间')
    query_strategy: Optional[str] = Field(max_length=255, description='检索策略')
    rerank_model: Optional[str] = Field(max_length=255, description='重排序模型')
    es_doc_id: Optional[str] = Field(max_length=255, description='elasticsearch的分片id')
    content: Optional[str] = Field(description="切片内容")


class TblDeptCompute(SQLModel, table=True):
    __tablename__ = "tbl_dept_compute"

    id: Optional[int] = Field(nullable=False, primary_key=True)
    dept_code: str
    name: str
    upper_dept_code: Optional[str]
    dept_level: str
    l1_dept_code: Optional[str]
    l1_name: Optional[str]
    l2_dept_code: Optional[str]
    l2_name: Optional[str]
    l3_dept_code: Optional[str]
    l3_name: Optional[str]
    l4_dept_code: Optional[str]
    l4_name: Optional[str]
    l5_dept_code: Optional[str]
    l5_name: Optional[str]
    l6_dept_code: Optional[str]
    l6_name: Optional[str]
    l7_dept_code: Optional[str]
    l7_name: Optional[str]
    l8_dept_code: Optional[str]
    l8_name: Optional[str]
    l9_dept_code: Optional[str]
    l9_name: Optional[str]


class EvalDatasetJob(SQLModel, table=True):
    __tablename__ = "eval_dataset_job"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    uid: str
    job_status: JobStatus
    job_type: GenerationEvaluationJobType
    eval_dataset_id: UUID

    created_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    updated_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )

    extra_config: Optional[Dict[Any, Any]] = Field(sa_column=Column(JSON))

    kb_id: UUID = Field(foreign_key="knowledge_base.id")
    knowledge_base: KnowledgeBase = Relationship(back_populates="eval_dataset_jobs")

    eval_data_metrics: List["EvalDataMetric"] = Relationship(
        back_populates="eval_dataset_job", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )

    eval_dataset_comprehensive_metrics: Optional["EvalDatasetComprehensiveMetric"] = Relationship(
        back_populates="eval_dataset_job", sa_relationship_kwargs={"uselist": False, "cascade": "all, delete-orphan"}
    )


class EvalDataset(SQLModel, table=True):
    __tablename__ = "eval_dataset"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str
    description: Optional[str]
    create_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    update_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )
    valid: bool
    kb_id: UUID = Field(foreign_key="knowledge_base.id")
    extra_config: Optional[Dict[Any, Any]] = Field(sa_column=Column(JSON))
    knowledge_base: KnowledgeBase = Relationship(back_populates="eval_datasets")

    eval_datas: List["EvalData"] = Relationship(
        back_populates="eval_dataset", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class EvalData(SQLModel, table=True):
    __tablename__ = "eval_data"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    data_type: EvalDataType
    question: str
    ground_truth: str
    context: Optional[str]
    valid: bool
    original_document_id: Optional[UUID]
    kba_id: Optional[UUID]
    document_name: Optional[str]
    source: Optional[str]
    create_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    update_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )

    eval_dataset_id: UUID = Field(foreign_key="eval_dataset.id")
    eval_dataset: EvalDataset = Relationship(back_populates="eval_datas")

    eval_data_metrics: List["EvalDataMetric"] = Relationship(
        back_populates="eval_data", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class EvalDatasetComprehensiveMetric(SQLModel, table=True):
    __tablename__ = "eval_dataset_comprehensive_metric"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    hit: Optional[float]
    mrr: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]
    faithfulness: Optional[float]
    answer_correctness: Optional[float]
    answer_relevancy: Optional[float]
    answer_similarity: Optional[float]
    score: Optional[float]
    create_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    update_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )
    eval_dataset_job_id: UUID = Field(foreign_key="eval_dataset_job.id")
    eval_dataset_job: EvalDatasetJob = Relationship(back_populates="eval_dataset_comprehensive_metrics")


class EvalDataMetric(SQLModel, table=True):
    __tablename__ = "eval_data_metric"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    answer: str
    hit: Optional[float]
    mrr: Optional[float]
    contexts: Optional[Dict[Any, Any]] = Field(default={}, sa_column=Column(JSON))
    context_precision: Optional[float]
    context_recall: Optional[float]
    faithfulness: Optional[float]
    answer_correctness: Optional[float]
    answer_relevancy: Optional[float]
    answer_similarity: Optional[float]
    valid: bool
    create_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    update_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )
    is_baseline: Optional[bool]
    note: Optional[str]
    manual_score: Optional[float]
    eval_dataset_job_id: UUID = Field(foreign_key="eval_dataset_job.id")
    eval_dataset_job: EvalDatasetJob = Relationship(back_populates="eval_data_metrics")

    eval_data_id: UUID = Field(foreign_key="eval_data.id")
    eval_data: EvalData = Relationship(back_populates="eval_data_metrics")


class IntentAssets(SQLModel, table=True):
    __tablename__ = "intent_assets"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    creator: str
    tag: Optional[str]
    intent_name: str
    intent_sn: str = Field(unique=True)
    intent_es_index: str


class KnowledgeBaseStopwordAndSynonym(SQLModel, table=True):
    __tablename__ = "knowledge_base_stopword_and_synonym"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    word_type: StopWordOrSynonymType
    word: str
    create_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    update_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )
    kb_id: UUID = Field(foreign_key="knowledge_base.id")
    knowledge_base: KnowledgeBase = Relationship(
        back_populates="knowledge_base_stopword_and_synonyms",
    )


class KnowledgeBaseFavorites(SQLModel, table=True):
    __tablename__ = "knowledge_base_favorites"
    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    employee_number: str
    kb_id: UUID = Field(foreign_key="knowledge_base.id")
    knowledge_base: KnowledgeBase = Relationship(back_populates="knowledge_base_favorites")  # 确保名称匹配
    # 添加唯一约束，防止重复收藏
    __table_args__ = (UniqueConstraint("employee_number", "kb_id", name="_employee_kb_uc"),)


class OnlineQaPair(SQLModel, table=True):
    __tablename__ = "online_qa_pair"
    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    question: str
    answer: str
    document_url: Optional[str] = Field(default=None, nullable=True)
    extended_questions: str
    kb_sn: str
    asset_name: str
    qa_pair_type: QaPairType
    status: QaPairStatus
    kba_id: UUID = Field(foreign_key="knowledge_base_asset.id")
    create_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    update_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )
    knowledge_base_asset: KnowledgeBaseAsset = Relationship(back_populates="online_qa_pairs")


class KnowledgeBaseBlacklistMember(SQLModel, table=True):
    __tablename__ = "knowledge_base_blacklist_member"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    employee_number: str
    employee_name: Optional[str]
    department: Optional[str]
    kb_id: UUID = Field(foreign_key="knowledge_base.id")
    knowledge_base: KnowledgeBase = Relationship(
        back_populates="knowledge_base_blacklist_members",
    )


class WhitelistMember(SQLModel, table=True):
    __tablename__ = "whitelist_member"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    employee_number: str
    employee_name: Optional[str]
    department: Optional[str]


class UserInfo(SQLModel, table=True):
    __tablename__ = "user_info"

    user_id: str = Field(primary_key=True)
    user_name: Optional[str]
    l1_dept_code: Optional[str]
    l1_name: Optional[str]
    l2_dept_code: Optional[str]
    l2_name: Optional[str]
    l3_dept_code: Optional[str]
    l3_name: Optional[str]
    l4_dept_code: Optional[str]
    l4_name: Optional[str]
    l5_dept_code: Optional[str]
    l5_name: Optional[str]
    l6_dept_code: Optional[str]
    l6_name: Optional[str]
    l7_dept_code: Optional[str]
    l7_name: Optional[str]
    l8_dept_code: Optional[str]
    l8_name: Optional[str]
    l9_dept_code: Optional[str]
    l9_name: Optional[str]


class CorpusDefect(SQLModel, table=True):
    __tablename__ = "corpus_defect"

    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    rule_code: int
    job_id: UUID
    extended_metadata: Optional[Dict[str, Any]] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    updated_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )


class ComputeWhitelist(SQLModel, table=True):
    __tablename__ = "compute_member_white_list"

    id: Optional[int] = Field(primary_key=True)
    w3Account: str = Field(unique=True)
    redirect_dept_code: Optional[str]


class DocumentDefect(SQLModel, table=True):
    __tablename__ = "document_defect"
    id: Optional[UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    job_id: UUID = Field(index=True)
    rule_code: int
    detection_context: Optional[Dict[str, Any]] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    updated_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )
    kb_id: Optional[UUID] = Field(foreign_key="knowledge_base.id", index=True)
    knowledge_base: KnowledgeBase = Relationship(
        back_populates="document_defects",
    )

    kba_id: Optional[UUID] = Field(foreign_key="knowledge_base_asset.id", index=True)
    knowledge_base_asset: KnowledgeBaseAsset = Relationship(
        back_populates="document_defects",
    )

    doc_id: Optional[UUID] = Field(foreign_key="original_document.id")
    compare_doc_id: Optional[UUID] = Field(foreign_key="original_document.id")

    original_document: OriginalDocument = Relationship(
        back_populates="document_defects", sa_relationship_kwargs={"foreign_keys": "DocumentDefect.doc_id"}
    )
    compare_original_document: OriginalDocument = Relationship(
        back_populates="compare_document_defects",
        sa_relationship_kwargs={"foreign_keys": "DocumentDefect.compare_doc_id"},
    )
    operation: str


class SupportProduct(SQLModel, table=True):
    __tablename__ = "support_product"

    id: Optional[int] = Field(nullable=False, primary_key=True)
    domain_id: str
    product_id: str
    product_name: str


class SpilloverManage(SQLModel, table=True):
    __tablename__ = "knowledge_spillover_manage"

    id: Optional[int] = Field(default=None, primary_key=True)
    kb_sn: str
    user_id: str
    url_list: List[str] = Field(default_factory=list, sa_column=Column(JSONB))
    doc_id_list: List[str] = Field(default_factory=list, sa_column=Column(JSONB))
    download_key: str
    created_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    updated_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )


class DocDetectInfo(SQLModel, table=True):
    __tablename__ = "document_detect_info"

    id: Optional[int] = Field(default=None, primary_key=True)
    date: datetime.date
    dept_code: str
    dept_name: str
    kb_sn: str
    asset_name: str
    asset_label: str
    doc_id: uuid.UUID
    doc_name: str
    doc_url: str
    detect_info: str
    is_show: bool = True
    created_at: datetime.datetime = Field(default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True)))
    updated_at: datetime.datetime = Field(
        default_factory=now_with_time_zone, sa_column=Column(DateTime(timezone=True), onupdate=now_with_time_zone)
    )