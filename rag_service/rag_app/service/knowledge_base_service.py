import contextlib
import json
import re
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote

import fastapi_pagination.iterables
import pandas as pd
import requests
from cachetools import cached, TTLCache
from fastapi import BackgroundTasks, UploadFile
from fastapi_pagination import Page, Params
from fastapi_pagination.ext.sqlmodel import paginate
from more_itertools import collapse
from sqlalchemy import (
    String,
    and_,
    between,
    delete,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, cast, desc, func, or_, select, text
from starlette_context import context

from rag_service.config import (
    APP_ID,
    DEFAULT_TOP_K,
    DOCUMENT_THRESHOLD_SCORE,
    HW_APP_KEY,
    LLM_MODEL,
    MAX_LAYER_KNOWLEDGE_BASE_ASSET_SIZE,
    MAX_LAYER_KNOWLEDGE_BASE_NUMBER,
    MAX_PRIVATE_KNOWLEDGE_BASE_ASSET_SIZE,
    MAX_PRIVATE_KNOWLEDGE_BASE_NUMBER,
    MAX_SYNONYM_SIZE,
    MULTI_TURN_CONVERSATIONS_QUESTION_REWRITE_PROMPT_ID,
    POSSIBLE_SIMILAR_KNOWLEDGE_BASE_COUNT,
    QA_QUESTION_SIMILARITY_THRESHOLD,
    QUESTION_GENERATION_PROMPT_ID,
    QWEN_KNOWLEDGE_BASE_LIST,
    QWEN_LLM_MODEL,
    SYNONYM_PROMPT_ID,
    TRY_IT_OUT_URL,
)
from rag_service.config_center import get_w3_token
from rag_service.constants import (
    DEFAULT_PROMPT_TEMPLATE,
    EXCEL_HEIGHT,
    EXCEL_WIDTH,
    IPD_RAG_RETRIEVE_KB_URL,
    IPD_RAG_RETRIEVE_METHOD_MAP,
    IS_OLD_USER,
    IS_WHITELIST_USER,
    KNOWLEDGE_BASE_ASSETS_METADATA_FILE,
    KNOWLEDGE_BASE_DOCUMENTS_METADATA_FILE,
    KNOWLEDGE_BASE_OWNER_DISLIKE_TAGS,
    MAPPING_ASSET_TYPE_TO_LABEL,
    MULTI_KB_RETRIEVE_DOCUMENT_SCORE_THRESHOLD,
    MULTI_KB_RETRIEVE_LLM_MODEL,
    MULTI_KB_RETRIEVE_QUERY_STRATEGY,
    MULTI_KB_RETRIEVE_RERANK_MODEL,
    MULTI_TURN_CONVERSATIONS_QUESTION_REWRITE_PROMPT,
    NOT_OLD_USER,
    NOT_WHITELIST_USER,
    PUBLIC_ACCOUNT_NAME,
    QUESTION_GENERATION_PROMPT,
    SUPER_ROOT_NAME,
    SYNONYM_PROMPT,
    XIAOLUBAN_NAME,
    ENABLE_RETRIEVE_FROM_IPD_RAG,
    ENABLE_RETRIEVE_FROM_IPD_RAG_CONFIG_KEY, PBI_ATHENA_KNOWLEDGE_NAME_PREFIX,
    IPD_RAG_RETRIEVE_URL, ATHENA_KNOWLEDGE_URL_PREFIX,
)
from rag_service.database import engine, engine_3ms
from rag_service.exceptions import (
    InvalidParamException,
    KnowledgeBaseMemberNotExistsException,
    KnowledgeBaseNotExistsException,
    OperationNotPermittedException,
    RerankConnectException,
    RerankResponseException,
    UserInformationSearchException,
)
from rag_service.llms.llm import (
    RagLLM,
    llm_answer,
    llm_stream_answer,
    select_llm_and_get_answer,
)
from rag_service.logger import Module, get_logger
from rag_service.logger.wrapper_trace import safe_trace
from rag_service.middleware.context import request_val
from rag_service.models.api.models import (
    CancelKbPermissionsReq,
    CreateKnowledgeBaseReq,
    CreateUpdateBlacklistMemberReq,
    CreateUpdateBlacklistMultipleMemberReq,
    CreateUpdateMemberReq,
    CreateUpdateMultipleMemberReq,
    DeleteMemberReq,
    DeleteStopwordOrSynonymReq,
    DepartmentKnowledgeBaseInformationAnswer,
    DepartmentRequest,
    DeptManagerResponse,
    DomainManagerResponse,
    ExportKnowledgeBaseInfoRequest,
    FeedbackAnswer,
    FuzzyMatchKnowledgeBaseResponse,
    GetKbMigrateInfoReq,
    Highlight,
    KnowledgeBaseBlacklistMemberShowInfo,
    KnowledgeBaseConfig,
    KnowledgeBaseInfo,
    KnowledgeBaseMemberShowInfo,
    KnowledgeBaseOwnerRequest,
    KnowledgeBaseShowInfo,
    KnowledgeBaseTypeInfo,
    LayerKnowledgeBaseReq,
    LlmAnswer,
    OpenKbPermissionsReq,
    OperateDefaultKnowledgeBaseReq,
    PptHelperRequest,
    PptHelperResponse,
    PrivaterKnowledgeBaseListReq,
    PublicKnowledgeBaseShowInfo,
    QueryRequest,
    QueryStopwordOrSynonymReq,
    QuitKbMemberReq,
    RandomDocRequest,
    ReferenceAnswerReq,
    ReferenceAnswerResp,
    RerankSearchDataReq,
    RetrievedDocument,
    RetrievedDocumentMetadata,
    StopwordOrSynonymInfo,
    UpdateDepartmentKnowledgeBaseShareScope,
    UpdateFeedbackReq,
    UpdateStopwordOrSynonymReq,
    UploadStopwordOrSynonymReq,
    VectorizationJobStatus, WhitelistResp, WhitelistReq, WhitelistQueryReq, ChangeOwnerReq, SpilloverMarkRequest,
    PbiVersionRequest,
)
from rag_service.models.database.models import (
    AutoJobInstances,
    AutomatedJobSchedule,
    EvalDataset,
    EvalDatasetComprehensiveMetric,
    EvalDatasetJob,
    KnowledgeBase,
    KnowledgeBaseBlacklistMember,
    KnowledgeBaseFavorites,
    KnowledgeBaseMember,
    KnowledgeBaseStopwordAndSynonym,
    OnlineQaPair,
    RequestResponseLog,
    ServiceConfig,
    UserInfo,
    WhitelistMember, ComputeWhitelist, DocumentDefect, SpilloverManage,
)
from rag_service.models.enums import (
    Analyzer,
    CommonKnowledgeBaseType,
    DefaultKbStatus,
    DocumentContentType,
    Domain,
    JobStatus,
    KnowledgeBaseInfoType,
    KnowledgeBasePermission,
    KnowledgeBaseType,
    LayerKnowledgeBaseType,
    LlmModel,
    MemberType,
    OperationType,
    QueryStrategy,
    RerankModel,
    StopWordOrSynonymType,
    StreamAnswerFormat,
    VectorizationJobType,
    ServiceConfigType,
    ThreemsUriCategory, SecurityLevel,
)
from rag_service.models.generic.models import DeptInfo, KnowledgeBaseAndMemberTypeInfo, KnowledgeBaseMigrateInfo, \
    QaPair, IpdRAGKnowledgeBaseInfo
from rag_service.postprocess.format_answer import (
    format_answer,
    format_stream_answer,
    get_answer_with_citations,
    get_quote_reference,
)
from rag_service.postprocess.qa import extract_answer_from_qa_document
from rag_service.postprocess.synonym import get_question_synonyms
from rag_service.rag_app.api_exceptions import (
    AssetsAreAvailableInKnowledgeBase,
    BatchCancelFavoriteKnowledgeBaseParamFormatError,
    CustomPromptFormatError,
    DeptNotExistException,
    DuplicateKnowledgeBaseMemberException,
    ExportKnowledgeBaseInfoTypeException,
    ExternalApiResponseException,
    FeedbackRequestParamException,
    InvalidParamError,
    KnowledgeBaseAssetJobIsRunning,
    KnowledgeBaseAssetNotInitializedException,
    KnowledgeBaseMemberNotExistsError,
    KnowledgeBaseMemberValidationException,
    KnowledgeBaseNotExistsError,
    MaxKnowledgeBaseSizeReached,
    NotDepartmentSharedLayerKnowledgeBaseException,
    NotFoundRelevantKnowledge,
    OperationNotPermittedError,
    ReindexKnowledgeBaseError,
    RequestIdNotExistsException,
    StopwordNotExistException,
    SynchronousKnowledgeBaseSynonymAndStopwordError,
    SynonymFormatException,
    SynonymNotExistException,
    TimeIntervalException,
)
from rag_service.rag_app.dao.knowledge_base_dao import update_owner, query_by_owner
from rag_service.rag_app.dao.knowledge_base_member_dao import update_kn_base_owner
from rag_service.rag_app.dao.service_config_dao import query_by_config_type_and_name, query_value_by_name
from rag_service.rag_app.third_part.cloud_dragon_service import get_user_token_from_cookie
from rag_service.utils import portal_util
from rag_service.utils.context_lock import CTXThread
from rag_service.utils.db_3ms_util import get_group_blog_info, get_group_wiki_info
from rag_service.utils.db_util import (
    get_all_dept_shared_kbs,
    get_default_kb_list_by_dept_list,
    get_department_code_and_names,
    get_dept_info_by_code,
    get_dept_knowledge_base,
    get_domain_kbs,
    get_embedding_model_and_vector_stores,
    get_embedding_model_and_vector_stores_by_kb_kba_map,
    get_favorite_member_knowledge_base,
    get_grouped_vector_stores_by_knowledge_base_and_asset,
    get_init_and_not_running_knowledge_base_asset_list,
    get_kb_list_by_kb_sn_list,
    get_knowledge_base_and_member_type,
    get_knowledge_base_asset_info,
    get_knowledge_base_by_kb_sn,
    get_knowledge_base_document_info,
    get_knowledge_base_members,
    get_knowledge_base_owner,
    get_knowledge_base_running_knowledge_base_asset,
    get_knowledge_bases,
    get_knowledge_bases_vector_names,
    get_layer_dept_shared_kbs,
    get_member_knowledge_base,
    get_online_qa_pairs_by_kb_sn_list,
    get_personal_kbs,
    get_stopword_or_synonym,
    get_stopword_or_synonym_by_id,
    get_user_knowledge_base_asset_numbers,
    get_user_knowledge_base_numbers,
    get_user_not_in_devuc_layer_permission_kbs,
    get_user_record_from_all_knowledge_base_members,
    get_user_record_from_knowledge_base_member,
    get_user_record_from_whitelist_members,
    get_user_record_list_from_knowledge_base_member,
    get_user_vector_stores,
    validate_knowledge_base,
    validate_knowledge_base_name, get_whitelist_list_by_account_list,
    delete_whitelist_list_by_account_list, update_spill_over_status, get_doc_id_list_by_doc_source,
    get_doc_id_list_by_asset_uri, get_doc_id_list_like_asset_uri,
)
from rag_service.utils.his_util.get_user_info import get_user_name, get_user_name_without_exception
from rag_service.utils.his_util.idata_util import get_dept_employee_list
from rag_service.utils.his_util.member_infomation_search import search_member_information
from rag_service.utils.his_util.obs_util import create_signed_url, download_file_as_bytes, unify_object_key, upload_file_as_bytes
from rag_service.utils.his_util.threems_fetch_util import ThreeMSAssetValidator, ThreeMSUriType, get_uri_params, \
    get_threems_source, get_threems_community_doc_list, identify_uri_type
from rag_service.utils.his_util.threems_personal_blog_fetch_util import get_personal_blogs_by_asset_uri, \
    get_source_by_id
from rag_service.utils.ipd_rag_util import create_kb_and_asset_in_ipd_rag, thread_exec_operation_in_ipd_rag, \
    query_asset_by_name_in_ipd_rag
from rag_service.utils.jiaxian_community_fetch_util import get_article_infos
from rag_service.utils.portal_util import (
    get_dept_name,
    get_upper_dept_codes,
    is_compute_user,
    is_department_manager,
    is_department_user,
    is_domain_manager,
    is_pbi_manager,
    is_pbi_user,
)
from rag_service.utils.prompt_util import get_prompt
from rag_service.utils.redis_util import (
    old_user_keyspace,
    rc,
    whitelist_user_keyspace,
)
from rag_service.utils.request_answer_log_to_db import (
    thread_exec_kb_sn_list_to_database,
    insert_rewrite_question_to_db, insert_retrieve_result_to_db,
    send_log_to_aigc_record,
)
from rag_service.utils.time_util import (
    get_current_passed_seconds,
    get_next_cycle_time_from_current,
    now_with_time_zone,
)
from rag_service.vectorize.rerank import rerank_embedding
from rag_service.vectorstore import get_vector_store_manager
from rag_service.vectorstore.elasticsearch.es_model import es_query_k_documents
from rag_service.retrieval.evidence_packages import (
    EvidencePackageOptions,
    EvidencePackageResponse,
    build_evidence_packages,
    collect_evidence_candidate_documents,
    expanded_candidate_top_k,
    run_parallel_retrievers,
)

logger = get_logger(module=Module.APP)


JIAXIAN_URL_PRE = "https://jx.huawei.com"
THREEMS_URL_PRE = "https://3ms.huawei.com"
DBOX_URL_PRE = "https://dbox.huawei.com/"

# ipd rag检索线程池
executor = ThreadPoolExecutor(max_workers=20)

EVIDENCE_PACKAGE_DEFAULT_TOP_K = 5
EVIDENCE_PACKAGE_DEFAULT_EVIDENCE_PER_PACKAGE = 6
EVIDENCE_PACKAGE_DEFAULT_CANDIDATE_MULTIPLIER = 3
EVIDENCE_PACKAGE_MAX_CANDIDATE_K = 100
EVIDENCE_PACKAGE_DEFAULT_TABLE_EXPAND_RATIO_THRESHOLD = 0.7
EVIDENCE_PACKAGE_DEFAULT_MAX_FULL_TABLE_ROWS = 500
EVIDENCE_PACKAGE_DEFAULT_MAX_INLINE_TABLE_CHARS = 40000


def _validate_create_knowledge_base_param(req: CreateKnowledgeBaseReq):
    if req.is_external_knowledge and (not req.athena_kb_id or not req.ipd_rag_kb_id):
        raise InvalidParamException('请输入正确的雅典娜知识集ID和RAG知识库ID')


def _is_need_sync_ipd_rag(kb: KnowledgeBase) -> bool:
    # 与雅典娜知识集绑定的知识库无需同步到ipd rag
    if kb.athena_kb_id:
        return False
    # 个人知识库和领域知识库需要同步到ipd rag
    return kb.kb_type == KnowledgeBaseType.PRIVATE or kb.layer_kb_type == LayerKnowledgeBaseType.DOMAIN


def create_knowledge_base(req: CreateKnowledgeBaseReq, session: Session) -> str:
    _validate_create_knowledge_base_param(req)
    verify_knowledge_base_number(req, session)
    new_knowledge_base = get_knowledge_base_entity(req, session)
    new_member = KnowledgeBaseMember(
        employee_number=req.owner, member_type=MemberType.OWNER, kb_id=new_knowledge_base.id
    )
    new_knowledge_base.knowledge_base_members.append(new_member)
    # 个人知识库自动同步创建ipd rag知识库，迁移状态和自动更新状态设置为True
    if _is_need_sync_ipd_rag(new_knowledge_base):
        new_knowledge_base.is_migrated = True
        new_knowledge_base.allow_synchronous_update = True
    session.add(new_knowledge_base)
    session.commit()
    # 如果是个人知识库或者领域知识库，在ipd_rag异步创建知识库集合和知识库
    if _is_need_sync_ipd_rag(new_knowledge_base):
        thread_exec_operation_in_ipd_rag(
            session, "", new_knowledge_base.name, new_knowledge_base.sn, new_knowledge_base.owner, OperationType.CREATE
        )
    return new_knowledge_base.sn


def get_ipd_rag_knowledge_base_set_sn(knowledge_base: KnowledgeBase):
    if knowledge_base.common_dept_code:
        return "通用知识库_85e3569b"
    if knowledge_base.domain:
        return "领域知识库_0eaacd13"
    if knowledge_base.pbi_project_id:
        return "产品知识库_9834fdb8"
    return "个人知识库_051e0487"


def get_knowledge_base_entity(req: CreateKnowledgeBaseReq, session: Session) -> KnowledgeBase:
    serial_number = unify_object_key(f"{req.name}_{uuid.uuid4().hex[:8]}")
    layer_kb_type = None
    common_dept_code = None
    dept_no_tree = None
    common_kb_type = None
    domain = None
    pbi_project_id = None
    config = None
    # 分层知识库
    if req.kb_type != KnowledgeBaseType.PRIVATE:
        layer_kb_type = req.layer_kb_type
        # 通用知识
        if req.layer_kb_type == LayerKnowledgeBaseType.COMMON:
            common_dept_code = req.common_dept_code
            dept_no_tree = concat_dept_no_tree(session, req.common_dept_code)
            common_kb_type = req.common_kb_type
        # 领域知识
        elif req.layer_kb_type == LayerKnowledgeBaseType.DOMAIN:
            domain = req.domain
        # 产品知识
        elif req.layer_kb_type == LayerKnowledgeBaseType.PRODUCT:
            pbi_project_id = req.pbi_project_id
        config = KnowledgeBaseConfig(
            top_k=DEFAULT_TOP_K,
            document_score_threshold=DOCUMENT_THRESHOLD_SCORE,
            rerank_model=RerankModel.QWEN3_RERANKER_4B,
            query_strategy=QueryStrategy.TEXT_VECTOR_HYBRID_QUERY,
            prompt=SYNONYM_PROMPT if req.analyzer == Analyzer.IK_ANALYZER else DEFAULT_PROMPT_TEMPLATE,
            llm_model=LlmModel.DeepSeek_V3_1,
        )
    if req.kb_type != KnowledgeBaseType.PRIVATE:
        # 不允许创建同名知识库
        validate_knowledge_base_name(common_dept_code, domain, pbi_project_id, req.name, session)
    return KnowledgeBase(
        name=req.name,
        sn=serial_number,
        owner=req.owner,
        kb_type=req.kb_type,
        layer_kb_type=layer_kb_type,
        common_dept_code=common_dept_code,
        dept_no_tree=dept_no_tree,
        common_kb_type=common_kb_type,
        domain=domain,
        pbi_project_id=pbi_project_id,
        analyzer=req.analyzer,
        config=json.loads(config.model_dump_json()) if config else {},
        ipd_rag_kb_id=req.ipd_rag_kb_id if req.is_external_knowledge else None,
        athena_kb_id=req.athena_kb_id if req.is_external_knowledge else None,

    )


def concat_dept_no_tree(session: Session, dept_code: str):
    dept_info = get_dept_info_by_code(session, dept_code)
    dept_code_list = []
    for level in range(2, 10):
        dept_code_attr = f"l{level}_dept_code"
        dept_code = getattr(dept_info, dept_code_attr, None)
        if not dept_code:
            break
        dept_code_list.append(dept_code)
    return "|".join(dept_code_list)


def _get_max_kb_number(req: CreateKnowledgeBaseReq, session: Session):
    if req.kb_type == KnowledgeBaseType.PRIVATE:
        return MAX_PRIVATE_KNOWLEDGE_BASE_NUMBER

    config_value = None
    if req.layer_kb_type == LayerKnowledgeBaseType.COMMON:
        config_value = query_by_config_type_and_name(session, req.common_dept_code, ServiceConfigType.DEPT_KB_COUNT)
    elif req.layer_kb_type == LayerKnowledgeBaseType.DOMAIN:
        config_value = query_by_config_type_and_name(session, req.domain.name, ServiceConfigType.DOMAIN_KB_COUNT)

    if config_value:
        return max(int(config_value.value), MAX_LAYER_KNOWLEDGE_BASE_NUMBER)

    return MAX_LAYER_KNOWLEDGE_BASE_NUMBER


def verify_knowledge_base_number(req: CreateKnowledgeBaseReq, session: Session):
    query = select(func.count(KnowledgeBase.id))
    if req.kb_type == KnowledgeBaseType.LAYER:
        query = query.where(KnowledgeBase.kb_type == KnowledgeBaseType.LAYER)
        if req.layer_kb_type == LayerKnowledgeBaseType.COMMON:
            query = query.where(
                KnowledgeBase.layer_kb_type == LayerKnowledgeBaseType.COMMON,
                KnowledgeBase.common_dept_code == req.common_dept_code,
            )
        elif req.layer_kb_type == LayerKnowledgeBaseType.DOMAIN:
            query = query.where(
                KnowledgeBase.layer_kb_type == LayerKnowledgeBaseType.DOMAIN,
                KnowledgeBase.domain == req.domain,
            )
        elif req.layer_kb_type == LayerKnowledgeBaseType.PRODUCT:
            query = query.where(
                KnowledgeBase.layer_kb_type == LayerKnowledgeBaseType.PRODUCT,
                KnowledgeBase.pbi_project_id == req.pbi_project_id,
            )
    else:
        query = query.where(
            KnowledgeBase.kb_type == KnowledgeBaseType.PRIVATE,
            KnowledgeBase.owner == req.owner,
        )
    kb_number = session.execute(query).scalar()
    max_kb_number = _get_max_kb_number(req, session)
    if kb_number >= max_kb_number:
        raise MaxKnowledgeBaseSizeReached(f"知识库数量到达上限<{max_kb_number}>个")


@safe_trace(log_args=True, include_args=["req"])
def get_llm_answer(req: QueryRequest, background_tasks: BackgroundTasks, session: Session) -> LlmAnswer:
    validate_user_operation_permission(session, req.uid)
    kb_sn_list = req.kb_sn_list if req.kb_sn_list else [req.kb_sn]
    knowledge_base_list = list(batch_validate_knowledge_base(session, kb_sn_list, req.uid))
    documents = retrieve_documents(req, knowledge_base_list, session, background_tasks=background_tasks)
    validate_retrieved_documents(documents, req)
    qa_answer, document = extract_answer_from_qa_document(req.question, documents)
    if qa_answer:
        return format_answer([document], qa_answer, req.fetch_source)

    synonyms = get_question_synonyms(session, req.question, kb_sn_list)
    kb_prompt = (
        req.user_custom_retrieve_config.prompt
        if req.user_custom_retrieve_config and req.user_custom_retrieve_config.prompt
        else get_kb_prompt(kb_sn_list, session)
    )
    llm_model_name = get_llm_model_name(knowledge_base_list, req)
    llm_config = req.user_custom_retrieve_config.dict() if req.user_custom_retrieve_config else {}
    return llm_answer(
        req.question,
        kb_sn_list,
        req.fetch_source,
        documents,
        synonyms,
        temperature=llm_config.get("temperature"),
        llm_top_p=llm_config.get("top_p"),
        llm_top_k=llm_config.get("top_k") if llm_config.get("top_k") else get_multi_kb_max_top_k(knowledge_base_list),
        max_tokens=llm_config.get("max_tokens"),
        repetition_penalty=llm_config.get("repetition_penalty"),
        frequency_penalty=llm_config.get("frequency_penalty"),
        knowledge_base_prompt=kb_prompt,
        llm_model_name=llm_model_name,
        request_id=req.request_id,
        background_tasks=background_tasks
    )


def permission_judge(session: Session, kb_sn_list: List[str], uid: str):
    validate_user_operation_permission(session, uid)
    knowledge_base_list = list(batch_validate_knowledge_base(session, kb_sn_list, uid))
    return knowledge_base_list


def get_llm_model_name(knowledge_base_list: List[KnowledgeBase], req: QueryRequest):
    if req.user_custom_retrieve_config and req.user_custom_retrieve_config.llm_model:
        return req.user_custom_retrieve_config.llm_model
    if len(knowledge_base_list) > 1:
        return MULTI_KB_RETRIEVE_LLM_MODEL
    if not knowledge_base_list:
        return LLM_MODEL
    kb_sn_list = [kb.sn for kb in knowledge_base_list]
    if set(kb_sn_list) & set(QWEN_KNOWLEDGE_BASE_LIST):
        logger.info("Knowledge base list %s use qwen model.", knowledge_base_list)
        return QWEN_LLM_MODEL
    kb_config = KnowledgeBaseConfig.model_validate(knowledge_base_list[0].config)
    return kb_config.llm_model.value if kb_config.llm_model else LLM_MODEL


def get_kb_prompt(kb_sn_list: List[str], session: Session):
    if len(kb_sn_list) == 1:
        knowledge_base = validate_knowledge_base(session, kb_sn_list[0])
        kb_config = KnowledgeBaseConfig.model_validate(knowledge_base.config)
        return kb_config.prompt if kb_config.prompt else None
    return None


def judge_knowledge_base_rerank_retrieve(kb_list: List[KnowledgeBase], req: QueryRequest):
    if len(kb_list) > 1 and not req.user_custom_retrieve_config:
        return True
    kb_config = KnowledgeBaseConfig.model_validate(kb_list[0].config)
    if len(kb_list) == 1 and kb_config.rerank_model and not req.user_custom_retrieve_config:
        return kb_config.rerank_model != RerankModel.BASIC
    return req.user_custom_retrieve_config and req.user_custom_retrieve_config.rerank != RerankModel.BASIC


def get_online_qa_pairs(session: Session, knowledge_base_list: List[KnowledgeBase], req: QueryRequest):
    query_kb_sn_list = []
    if req.query_database_first is None:
        for knowledge_base in knowledge_base_list:
            kb_config = KnowledgeBaseConfig.model_validate(knowledge_base.config)
            if kb_config.query_database_first:
                query_kb_sn_list.append(knowledge_base.sn)
    if req.query_database_first:
        query_kb_sn_list = [knowledge_base.sn for knowledge_base in knowledge_base_list]
    if not query_kb_sn_list:
        return None
    return get_online_qa_pairs_by_kb_sn_list(session, query_kb_sn_list)


def get_similarity_values(online_qa_pairs: OnlineQaPair, req: QueryRequest):
    qa_list: List[Tuple[str, str]] = []
    index_list = []  # 存储每个问题对应的在线qa对下标，方便后续获取对应的在线qa对
    for index, online_qa_pair in enumerate(online_qa_pairs):
        qa_list.append((req.question, online_qa_pair.question))
        extended_questions = json.loads(online_qa_pair.extended_questions)
        index_list.append(index)
        for extend_question in extended_questions:
            qa_list.append((req.question, extend_question))
            index_list.append(index)
    return rerank_embedding(qa_list), index_list


@safe_trace()
def query_online_qa_pairs(req: QueryRequest, knowledge_base_list: List[KnowledgeBase], session: Session):
    documents = []
    online_qa_pairs = get_online_qa_pairs(session, knowledge_base_list, req)
    # 获取到在线qa对则优先处理
    if online_qa_pairs:
        similarity_value_list, index_list = get_similarity_values(online_qa_pairs, req)
        # 获取相似值列表中的最大相似值max_similarity_value和对应的索引下标max_similarity_value_index
        max_similarity_value_index, max_similarity_value = max(enumerate(similarity_value_list), key=lambda x: x[1])
        # 最大相似值大于相似性阈值时直接返回qa对的回答
        if max_similarity_value >= QA_QUESTION_SIMILARITY_THRESHOLD:
            target_online_qa_pair = online_qa_pairs[index_list[max_similarity_value_index]]
            qa_answer = target_online_qa_pair.answer
            document = RetrievedDocument(
                text=QaPair(question=req.question, answer=qa_answer).json(),
                metadata=RetrievedDocumentMetadata(
                    source=target_online_qa_pair.document_url or "",
                    mtime=target_online_qa_pair.update_at,
                    extended_metadata={
                        "title": target_online_qa_pair.asset_name,
                        "kb_sn": target_online_qa_pair.kb_sn,
                        "asset_name": target_online_qa_pair.asset_name,
                        "content_type": DocumentContentType.ONLINE_QA.value,
                    },
                ),
            )
            documents = [document]
    return documents


@safe_trace()
def retrieve_documents(
    req: QueryRequest, knowledge_base_list: List[KnowledgeBase], session: Session, collect_info=True, background_tasks: Optional[BackgroundTasks] = None
) -> List[RetrievedDocument]:
    verify_retrieve_kb_sn_or_kb_sn_list(req.kb_sn, req.kb_sn_list)
    rewrite_question = question_rewrite(
        req.question,
        MULTI_TURN_CONVERSATIONS_QUESTION_REWRITE_PROMPT,
        MULTI_TURN_CONVERSATIONS_QUESTION_REWRITE_PROMPT_ID,
        req.historical_questions,
        session,
    )
    req.question = rewrite_question

    # 异步记录改写后的问题
    background_tasks.add_task(insert_rewrite_question_to_db, req.request_id, rewrite_question)
    # 根据用户请求的配置或知识库的配置查询在线qa对
    documents = query_online_qa_pairs(req, knowledge_base_list, session)
    # 不优先查询在线qa对或在线qa对中没有满足要求的回答，则继续检索切片
    if not documents:
        if judge_knowledge_base_rerank_retrieve(knowledge_base_list, req):
            documents = process_rerank_retrieval(session, req, background_tasks, knowledge_base_list, collect_info)
        else:
            documents = process_basic_retrieval(session, req, background_tasks, knowledge_base_list, collect_info)

    # 异步记录检索结果
    background_tasks.add_task(insert_retrieve_result_to_db, req.request_id, documents)
    return documents


@safe_trace()
def get_evidence_packages(
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    session: Session,
) -> EvidencePackageResponse:
    original_question = req.question
    kb_sn_list = req.kb_sn_list if req.kb_sn_list else [req.kb_sn]
    knowledge_base_list = permission_judge(session, kb_sn_list, req.uid)
    options = _get_evidence_package_options(req)
    candidate_top_k = expanded_candidate_top_k(
        options.package_top_k,
        options.max_evidence_per_package,
        _get_evidence_candidate_multiplier(req),
        max_candidate_k=EVIDENCE_PACKAGE_MAX_CANDIDATE_K,
    )

    rewrite_question = question_rewrite(
        req.question,
        MULTI_TURN_CONVERSATIONS_QUESTION_REWRITE_PROMPT,
        MULTI_TURN_CONVERSATIONS_QUESTION_REWRITE_PROMPT_ID,
        req.historical_questions,
        session,
    )
    req.question = rewrite_question
    if background_tasks and req.request_id:
        background_tasks.add_task(insert_rewrite_question_to_db, req.request_id, rewrite_question)

    documents = query_online_qa_pairs(req, knowledge_base_list, session) if _include_online_qa(req) else []
    if not documents:
        documents = _retrieve_evidence_candidate_documents(
            session,
            req,
            background_tasks,
            knowledge_base_list,
            candidate_top_k,
        )

    if background_tasks and req.request_id:
        background_tasks.add_task(insert_retrieve_result_to_db, req.request_id, documents)

    artifact_url_resolver = create_signed_url if options.artifact_mode == "signed_url" else None
    return build_evidence_packages(
        query=original_question,
        rewrite_query=rewrite_question,
        documents=documents,
        options=options,
        artifact_url_resolver=artifact_url_resolver,
        artifact_text_resolver=_download_artifact_text,
    )


def _retrieve_evidence_candidate_documents(
    session: Session,
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    knowledge_base_list: List[KnowledgeBase],
    candidate_top_k: int,
) -> List[RetrievedDocument]:
    original_top_k = req.top_k
    req.top_k = candidate_top_k
    try:
        if req.user_custom_retrieve_config and req.user_custom_retrieve_config.sequential_retrieval:
            return _retrieve_evidence_candidates_sequential(session, req, background_tasks, knowledge_base_list)
        retrieve_config = _get_evidence_retrieve_config(knowledge_base_list, req)
        retrievers = _build_evidence_candidate_retrievers(
            session,
            req,
            background_tasks,
            knowledge_base_list,
            retrieve_config,
        )
        documents, has_ipd_documents = collect_evidence_candidate_documents(
            run_parallel_retrievers(retrievers),
            max_documents=candidate_top_k,
        )
        if judge_knowledge_base_rerank_retrieve(knowledge_base_list, req):
            return rerank_retrieved_documents(
                req.question,
                documents,
                retrieve_config.top_k,
                retrieve_config.document_score_threshold,
                retrieve_config.rerank_model,
            )
        if has_ipd_documents:
            return rerank_retrieved_documents(
                req.question,
                documents,
                retrieve_config.top_k,
                retrieve_config.document_score_threshold,
                MULTI_KB_RETRIEVE_RERANK_MODEL,
            )
        return sorted(documents, key=lambda x: x.score, reverse=True)[:retrieve_config.top_k]
    finally:
        req.top_k = original_top_k


def _retrieve_evidence_candidates_sequential(
    session: Session,
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    knowledge_base_list: List[KnowledgeBase],
) -> List[RetrievedDocument]:
    if judge_knowledge_base_rerank_retrieve(knowledge_base_list, req):
        return process_rerank_retrieval(session, req, background_tasks, knowledge_base_list)
    return process_basic_retrieval(session, req, background_tasks, knowledge_base_list)


def _get_evidence_retrieve_config(kb_list: List[KnowledgeBase], req: QueryRequest) -> KnowledgeBaseConfig:
    if len(kb_list) == 1:
        return get_retrieve_param_by_kb_config_and_request(kb_list[0], req)
    return get_multi_kb_retrieve_param(kb_list, req)


def _build_evidence_candidate_retrievers(
    session: Session,
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    knowledge_base_list: List[KnowledgeBase],
    retrieve_config: KnowledgeBaseConfig,
):
    if len(knowledge_base_list) == 1:
        return _single_kb_evidence_retrievers(
            session, req, background_tasks, knowledge_base_list[0], retrieve_config
        )
    return _multi_kb_evidence_retrievers(session, req, background_tasks, knowledge_base_list, retrieve_config)


def _single_kb_evidence_retrievers(
    session: Session,
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    knowledge_base: KnowledgeBase,
    retrieve_config: KnowledgeBaseConfig,
):
    ipd_kbs = [knowledge_base] if _should_retrieve_kb_from_ipd(knowledge_base, retrieve_config) else []
    retrievers = [_source_tagged_retriever("ipd", _ipd_evidence_retriever(req, retrieve_config, ipd_kbs))]
    if ipd_kbs:
        return retrievers
    vector_stores = (
        get_embedding_model_and_vector_stores_by_kb_kba_map(session, req.user_custom_retrieve_config.kb_kba_map)
        if req.user_custom_retrieve_config and req.user_custom_retrieve_config.kb_kba_map
        else get_embedding_model_and_vector_stores(session, knowledge_base.sn)
    )
    retrievers.append(
        _source_tagged_retriever(
            "libing",
            lambda: get_vector_store_manager().retrieve(
                req.question,
                retrieve_config.top_k,
                vector_stores,
                retrieve_config.document_score_threshold,
                analyzer=knowledge_base.analyzer,
                query_strategy=retrieve_config.query_strategy,
                request_id=req.request_id,
                background_tasks=background_tasks,
            ),
        )
    )
    return retrievers


def _multi_kb_evidence_retrievers(
    session: Session,
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    kb_list: List[KnowledgeBase],
    retrieve_config: KnowledgeBaseConfig,
):
    ipd_kbs = [kb for kb in kb_list if _is_enable_ipd_rag_retrieve() and _is_ipd_rag_retrieve_kb(kb)]
    libing_kb_sns = _libing_retrieve_kb_sns(kb_list)
    retrievers = [_source_tagged_retriever("ipd", _ipd_evidence_retriever(req, retrieve_config, ipd_kbs))]
    if not libing_kb_sns:
        return retrievers
    grouped_vector_stores = get_grouped_vector_stores_by_knowledge_base_and_asset(
        session, {kb_sn: [] for kb_sn in libing_kb_sns}
    )
    for analyzer, vector_stores in grouped_vector_stores.items():
        retrievers.append(
            _source_tagged_retriever(
                "libing",
                _libing_evidence_retriever(req, background_tasks, retrieve_config, analyzer, vector_stores),
            )
        )
    return retrievers


def _libing_retrieve_kb_sns(kb_list: List[KnowledgeBase]) -> List[str]:
    if not _is_enable_ipd_rag_retrieve():
        return [kb.sn for kb in kb_list]
    return [kb.sn for kb in kb_list if not _is_ipd_rag_retrieve_kb(kb)]


def _libing_evidence_retriever(req, background_tasks, retrieve_config, analyzer, vector_stores):
    return lambda: get_vector_store_manager().retrieve(
        req.question,
        retrieve_config.top_k,
        vector_stores,
        retrieve_config.document_score_threshold,
        analyzer=analyzer,
        query_strategy=retrieve_config.query_strategy,
        request_id=req.request_id,
        background_tasks=background_tasks,
    )


def _ipd_evidence_retriever(req: QueryRequest, retrieve_config: KnowledgeBaseConfig, ipd_kbs: List[KnowledgeBase]):
    return lambda: retrieve_documents_from_ipd_rag(
        req.question,
        retrieve_config.top_k,
        retrieve_config.query_strategy,
        ipd_kbs,
        get_all_pbi_version_knowledge_id_list(),
    )


def _source_tagged_retriever(source: str, retriever):
    return lambda: [(source, retriever())]


def _should_retrieve_kb_from_ipd(kb: KnowledgeBase, retrieve_config: KnowledgeBaseConfig) -> bool:
    return retrieve_config.retrieve_from_ipd_rag or (_is_ipd_rag_retrieve_kb(kb) and _is_enable_ipd_rag_retrieve())


def _get_evidence_package_options(req: QueryRequest) -> EvidencePackageOptions:
    package_top_k = _positive_int(
        getattr(req, "package_top_k", None),
        _positive_int(getattr(req, "top_k", None), EVIDENCE_PACKAGE_DEFAULT_TOP_K),
    )
    max_evidence_per_package = _positive_int(
        getattr(req, "max_evidence_per_package", None),
        EVIDENCE_PACKAGE_DEFAULT_EVIDENCE_PER_PACKAGE,
    )
    artifact_mode = getattr(req, "artifact_mode", None) or "key"
    if artifact_mode not in {"key", "signed_url"}:
        artifact_mode = "key"
    return EvidencePackageOptions(
        package_top_k=package_top_k,
        max_evidence_per_package=max_evidence_per_package,
        artifact_mode=artifact_mode,
        enable_table_expansion=_bool_value(getattr(req, "enable_table_expansion", None), True),
        table_expand_ratio_threshold=_ratio_float(
            getattr(req, "table_expand_ratio_threshold", None),
            EVIDENCE_PACKAGE_DEFAULT_TABLE_EXPAND_RATIO_THRESHOLD,
        ),
        max_full_table_rows=_non_negative_int(
            getattr(req, "max_full_table_rows", None),
            EVIDENCE_PACKAGE_DEFAULT_MAX_FULL_TABLE_ROWS,
        ),
        max_inline_table_chars=_non_negative_int(
            getattr(req, "max_inline_table_chars", None),
            EVIDENCE_PACKAGE_DEFAULT_MAX_INLINE_TABLE_CHARS,
        ),
    )


def _get_evidence_candidate_multiplier(req: QueryRequest) -> int:
    return _positive_int(
        getattr(req, "candidate_multiplier", None),
        EVIDENCE_PACKAGE_DEFAULT_CANDIDATE_MULTIPLIER,
    )


def _include_online_qa(req: QueryRequest) -> bool:
    return bool(getattr(req, "include_online_qa", False))


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _ratio_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < 0 or parsed > 1:
        return default
    return parsed


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _download_artifact_text(object_key: str) -> Optional[str]:
    if not object_key:
        return None
    content = download_file_as_bytes(object_key)
    return content.decode("utf-8")


@safe_trace()
def question_rewrite(
    question: str,
    prompt: str,
    prompt_id: str,
    historical_questions: List[Dict[str, str]] = None,
    session: Session = None,
) -> str:
    try:
        query = get_prompt(default_prompt=prompt, prompt_id=prompt_id).replace("{{ question }}", "{" + question + "}")
        if historical_questions:
            questions_string = (
                "{\n"
                + "\n\n".join([f"问题{i + 1}：{item['question']}" for i, item in enumerate(historical_questions)])
                + "\n}"
            )
            query = query.replace("{{ historical_questions }}", questions_string)
            messages = []
            question_ids = []
            for item in historical_questions:
                question_ids.append(item["question_id"])
            request_response_log = session.exec(
                select(RequestResponseLog)
                .where(RequestResponseLog.question_id.in_(question_ids))
                .order_by(RequestResponseLog.request_start_time)
            ).all()
            for log_item, question_item in zip(request_response_log, historical_questions):
                if not log_item.rewrite_question:
                    log_item.rewrite_question = log_item.question
                old_question_info = {"role": "user", "content": question_item["question"]}
                messages.append(old_question_info)
                rewrite_question_info = {"role": "assistant", "content": log_item.rewrite_question}
                messages.append(rewrite_question_info)
            return RagLLM().invoke(query, messages=messages)
        return question
    except Exception as e:
        logger.exception("An error occurred when rewrite question, error is %s", e)
        return question


def verify_retrieve_kb_sn_or_kb_sn_list(kb_sn: Optional[str], kb_sn_list: Optional[List[str]]) -> List[str]:
    if not kb_sn and not kb_sn_list:
        raise InvalidParamException('"kb_sn"和"kb_sn_list"需传其中一个')
    if kb_sn_list and kb_sn:
        raise InvalidParamException('"kb_sn"和"kb_sn_list"只能使用其中一个')
    return kb_sn_list if kb_sn_list else [kb_sn]


def _need_rerank(kb_list: List[KnowledgeBase]) -> bool:
    # 多知识库下，并且启用了ipd rag检索，并且既有个人知识库，也有分层知识库的场景下，需要rerank下
    if not kb_list or len(kb_list) == 1:
        return False
    if not _is_enable_ipd_rag_retrieve():
        return False
    has_private_kb = False
    has_layer_kb = False
    kb_index = 0
    while kb_index < len(kb_list):
        if kb_list[kb_index].kb_type == KnowledgeBaseType.PRIVATE:
            has_private_kb = True
        if kb_list[kb_index].kb_type == KnowledgeBaseType.LAYER:
            has_layer_kb = True
        if has_private_kb and has_layer_kb:
            return True
        kb_index += 1
    return False


@cached(cache=TTLCache(maxsize=1, ttl=3600))
def _is_enable_ipd_rag_retrieve() -> bool:
    # 数据库配置为1为开启，配置为0为关闭。默认开启，开关状态缓存一个小时
    with Session(engine) as session:
        is_enable_ipd_rag_retrieve = query_value_by_name(session, ENABLE_RETRIEVE_FROM_IPD_RAG_CONFIG_KEY)
    if is_enable_ipd_rag_retrieve:
        return int(is_enable_ipd_rag_retrieve) == 1
    return ENABLE_RETRIEVE_FROM_IPD_RAG


@safe_trace()
def process_basic_retrieval(
    session: Session, req: QueryRequest, background_tasks: BackgroundTasks, valid_kb_list: List[KnowledgeBase], collect_info: bool = True
):
    if req.user_custom_retrieve_config and req.user_custom_retrieve_config.sequential_retrieval:
        for knowledge_base in valid_kb_list:
            retrieve_config = get_retrieve_param_by_kb_config_and_request(knowledge_base, req)
            documents = process_single_knowledge_base_retrieval(session, req, knowledge_base, retrieve_config,
                                                                collect_info, background_tasks=background_tasks)
            if documents:
                return documents
        return []

    # 记录需要走ipd rag检索的知识库，后续统一调用接口
    ipd_rag_knowledge_list = []
    if len(valid_kb_list) == 1:
        retrieve_config = get_retrieve_param_by_kb_config_and_request(valid_kb_list[0], req)
        retrieved_documents = process_single_knowledge_base_retrieval(
            session, req, valid_kb_list[0], retrieve_config, collect_info, background_tasks=background_tasks,
            ipd_rag_knowledge_list=ipd_rag_knowledge_list
        )
        top_k = retrieve_config.top_k
    else:
        retrieve_config = get_multi_kb_retrieve_param(valid_kb_list, req)
        retrieved_documents = process_kb_sn_list_retrieval(
            session, req, valid_kb_list, retrieve_config, collect_info, background_tasks=background_tasks,
            ipd_rag_knowledge_list=ipd_rag_knowledge_list
        )
        top_k = retrieve_config.top_k

    # ipd rag检索，默认加入版本知识库
    ipd_rag_retrieved_documents = retrieve_documents_from_ipd_rag(req.question, retrieve_config.top_k,
                                                                  retrieve_config.query_strategy,
                                                                  ipd_rag_knowledge_list,
                                                                  get_all_pbi_version_knowledge_id_list())
    retrieved_documents.extend(ipd_rag_retrieved_documents)

    retrieved_documents = filter_same_document(retrieved_documents)

    # 虽然process_basic_retrieval方法判断了不需要rerank，但是如果ipd rag检索有结果，为保证检索结果的质量，使用默认的rerank模型进行rerank
    if ipd_rag_retrieved_documents:
        return rerank_retrieved_documents(
            req.question,
            retrieved_documents,
            top_k,
            retrieve_config.document_score_threshold,
            MULTI_KB_RETRIEVE_RERANK_MODEL,
        )
    return sorted(retrieved_documents, key=lambda x: x.score, reverse=True)[:top_k]


@safe_trace()
def process_single_knowledge_base_retrieval(
        session: Session, req: QueryRequest, knowledge_base: KnowledgeBase, retrieve_config: KnowledgeBaseConfig,
        collect_info: bool = True, background_tasks: Optional[BackgroundTasks] = None,
        ipd_rag_knowledge_list: Optional[List[KnowledgeBase]] = None
):
    # 已迁移完成的个人知识库默认走ipd rag的检索
    if retrieve_config.retrieve_from_ipd_rag or (
            _is_ipd_rag_retrieve_kb(knowledge_base)
            and _is_enable_ipd_rag_retrieve()):
        if ipd_rag_knowledge_list is not None:
            # 此处只添加记录，后续统一调用ipd rag检索接口
            ipd_rag_knowledge_list.append(knowledge_base)
            return []
        return retrieve_documents_from_ipd_rag(
            req.question, retrieve_config.top_k, retrieve_config.query_strategy, [knowledge_base]
        )
    return get_vector_store_manager().retrieve(
        req.question,
        retrieve_config.top_k,
        get_embedding_model_and_vector_stores_by_kb_kba_map(session, req.user_custom_retrieve_config.kb_kba_map)
        if req.user_custom_retrieve_config and req.user_custom_retrieve_config.kb_kba_map
        else get_embedding_model_and_vector_stores(session, knowledge_base.sn),
        retrieve_config.document_score_threshold,
        collect_info=collect_info,
        analyzer=knowledge_base.analyzer,
        query_strategy=retrieve_config.query_strategy,
        request_id=req.request_id,
        background_tasks=background_tasks
    )


def get_value(user_val, kb_val, default_val):
    return user_val if user_val is not None else (kb_val if kb_val is not None else default_val)


@safe_trace()
def get_retrieve_param_by_kb_config_and_request(
    knowledge_base: KnowledgeBase, req: QueryRequest
) -> KnowledgeBaseConfig:
    kb_config = KnowledgeBaseConfig.model_validate(knowledge_base.config)
    user_custom_config = req.user_custom_retrieve_config

    return KnowledgeBaseConfig(
        top_k=get_value(req.top_k if req.top_k else None, kb_config.top_k, DEFAULT_TOP_K),
        document_score_threshold=get_value(
            user_custom_config.document_score_threshold if user_custom_config else None,
            kb_config.document_score_threshold,
            DOCUMENT_THRESHOLD_SCORE,
        ),
        query_strategy=get_value(
            user_custom_config.query_strategy if user_custom_config else None,
            kb_config.query_strategy,
            QueryStrategy.HYBRID_QUERY,
        ),
        rerank_model=get_value(
            user_custom_config.rerank if user_custom_config else None, kb_config.rerank_model, RerankModel.QWEN3_RERANKER_4B
        ),
        retrieve_from_ipd_rag=get_value(
            user_custom_config.retrieve_from_ipd_rag if user_custom_config else None,
            kb_config.retrieve_from_ipd_rag,
            False,
        ),
    )


def filter_same_document(documents: List[RetrievedDocument]):
    document_set = set()
    unique_documents = []
    for document in documents:
        if document.text not in document_set:
            unique_documents.append(document)
            document_set.add(document.text)
    return unique_documents


def _is_ipd_rag_retrieve_kb(kb: KnowledgeBase) -> bool:
    # 与雅典娜知识集绑定，走ipd rag检索
    if kb.athena_kb_id:
        return True
    # 已经迁移完成的个人知识库走ipd rag检索
    return (kb.kb_type == KnowledgeBaseType.PRIVATE or kb.layer_kb_type == LayerKnowledgeBaseType.DOMAIN) and kb.is_migrated


@safe_trace()
def process_kb_sn_list_retrieval(
    session: Session,
    req: QueryRequest,
    kb_list: List[KnowledgeBase],
    retrieve_config: KnowledgeBaseConfig,
    collect_info: bool = True,
    background_tasks: Optional[BackgroundTasks] = None,
    ipd_rag_knowledge_list: List[KnowledgeBase] = None
):
    retrieved_documents = []
    # 已迁移完的知识库检索走ipd rag
    if _is_enable_ipd_rag_retrieve():
        # 筛选出已迁移完成的知识库，走ipd rag检索
        ipd_retrieve_kb_list = [kb for kb in kb_list if _is_ipd_rag_retrieve_kb(kb)]
        if ipd_retrieve_kb_list:
            if ipd_rag_knowledge_list is not None:
                # 此处只添加记录，后续统一调用ipd rag检索接口
                ipd_rag_knowledge_list.extend(ipd_retrieve_kb_list)
            else:
                retrieved_documents.extend(retrieve_documents_from_ipd_rag(
                    req.question, retrieve_config.top_k, retrieve_config.query_strategy, ipd_retrieve_kb_list
                ))

    # 未迁移完成的知识库走libing rag检索
    libing_retrieve_kb_sn_list = ([kb.sn for kb in kb_list if not _is_ipd_rag_retrieve_kb(kb)]
                                  if _is_enable_ipd_rag_retrieve()
                                  else [kb.sn for kb in kb_list])
    if libing_retrieve_kb_sn_list:
        analyzer_to_embedding_model_to_vector_stores = get_grouped_vector_stores_by_knowledge_base_and_asset(
            session, {kb_sn: [] for kb_sn in libing_retrieve_kb_sn_list}
        )
        for analyzer, embedding_model_to_vector_stores in analyzer_to_embedding_model_to_vector_stores.items():
            documents = get_vector_store_manager().retrieve(
                req.question,
                retrieve_config.top_k,
                embedding_model_to_vector_stores,
                collect_info=collect_info,
                analyzer=analyzer,
                query_strategy=retrieve_config.query_strategy,
                request_id=req.request_id,
                background_tasks=background_tasks
            )
            retrieved_documents.extend(documents)

    return retrieved_documents


@safe_trace()
def process_rerank_retrieval(
    session: Session, req: QueryRequest, background_tasks: BackgroundTasks, valid_kb_list: List[KnowledgeBase], collect_info: bool = True
):
    if req.user_custom_retrieve_config and req.user_custom_retrieve_config.sequential_retrieval:
        for knowledge_base in valid_kb_list:
            retrieve_config = get_retrieve_param_by_kb_config_and_request(knowledge_base, req)
            documents = process_single_knowledge_base_retrieval(
                session, req, knowledge_base, retrieve_config, collect_info, background_tasks=background_tasks
            )
            rerank_documents = rerank_retrieved_documents(
                req.question, documents, retrieve_config.top_k, retrieve_config.document_score_threshold,
                retrieve_config.rerank_model,
            )
            if rerank_documents:
                return rerank_documents
        return []

    # 记录需要走ipd rag检索的知识库，后续统一调用接口
    ipd_rag_knowledge_list = []
    if len(valid_kb_list) == 1:
        retrieve_config = get_retrieve_param_by_kb_config_and_request(valid_kb_list[0], req)
        retrieved_documents = process_single_knowledge_base_retrieval(
            session, req, valid_kb_list[0], retrieve_config, collect_info, background_tasks=background_tasks,
            ipd_rag_knowledge_list=ipd_rag_knowledge_list
        )
    else:
        retrieve_config = get_multi_kb_retrieve_param(valid_kb_list, req)
        retrieved_documents = process_kb_sn_list_retrieval(
            session, req, valid_kb_list, retrieve_config, collect_info, background_tasks,
            ipd_rag_knowledge_list=ipd_rag_knowledge_list
        )

    # ipd rag检索，默认加入版本知识库
    ipd_rag_retrieved_documents = retrieve_documents_from_ipd_rag(
        req.question, retrieve_config.top_k, retrieve_config.query_strategy, ipd_rag_knowledge_list,
        get_all_pbi_version_knowledge_id_list()
    )
    # 合并libing rag检索和ipd rag检索
    retrieved_documents.extend(ipd_rag_retrieved_documents)
    retrieved_documents = filter_same_document(retrieved_documents)

    return rerank_retrieved_documents(
        req.question,
        retrieved_documents,
        retrieve_config.top_k,
        retrieve_config.document_score_threshold,
        retrieve_config.rerank_model,
    )


def get_multi_kb_retrieve_param(kb_list: List[KnowledgeBase], req: QueryRequest) -> KnowledgeBaseConfig:
    top_k = req.top_k if req.top_k else get_multi_kb_max_top_k(kb_list)
    document_score_threshold = (
        req.user_custom_retrieve_config.document_score_threshold
        if req.user_custom_retrieve_config
        else MULTI_KB_RETRIEVE_DOCUMENT_SCORE_THRESHOLD
    )
    query_strategy = (
        req.user_custom_retrieve_config.query_strategy
        if req.user_custom_retrieve_config
        else MULTI_KB_RETRIEVE_QUERY_STRATEGY
    )
    rerank_model = (
        req.user_custom_retrieve_config.rerank if req.user_custom_retrieve_config else MULTI_KB_RETRIEVE_RERANK_MODEL
    )
    return KnowledgeBaseConfig(
        top_k=top_k,
        document_score_threshold=document_score_threshold,
        query_strategy=query_strategy,
        rerank_model=rerank_model,
    )


def get_multi_kb_max_top_k(kb_list: List[KnowledgeBase]):
    max_top_k_in_config = 0
    for kb in kb_list:
        kb_config = KnowledgeBaseConfig.model_validate(kb.config)
        max_top_k_in_config = max(max_top_k_in_config, kb_config.top_k if kb_config.top_k else DEFAULT_TOP_K)
    return max_top_k_in_config


def get_rerank_format(doc: RetrievedDocument):
    title = doc.metadata.retrieve_metadata.title if doc.metadata.retrieve_metadata else ""
    file_address = doc.metadata.retrieve_metadata.file_address if doc.metadata.retrieve_metadata else ""
    return f"标题: {title}\n文档路径: {file_address}\n片段内容: {doc.text}"


@safe_trace()
def rerank_retrieved_documents(
    question: str,
    retrieved_documents: List[RetrievedDocument],
    top_k: int,
    document_score_threshold: float = DOCUMENT_THRESHOLD_SCORE,
    rerank_model: RerankModel = RerankModel.QWEN3_RERANKER_4B,
) -> List[RetrievedDocument]:
    question_and_document_pairs = [(question, get_rerank_format(document)) for document in retrieved_documents]
    start = time.time()
    try:
        score_resp = rerank_embedding(question_and_document_pairs, rerank_model)
    except Exception as e:
        logger.error("rerank 服务请求失败: %s", e)
        # 直接返回 top_k 个 retrieved_documents
        return retrieved_documents[:top_k]
    end = time.time()
    logger.info("%s sentence pairs rerank 耗时: %s秒...", len(question_and_document_pairs), end - start)
    filter_documents = []
    for score, document in sorted(zip(score_resp, retrieved_documents), reverse=True, key=lambda x: x[0]):
        if score > document_score_threshold and len(filter_documents) < top_k:
            document.score = score
            filter_documents.append(document)
    return filter_documents


@safe_trace()
def validate_retrieved_documents(documents: List[RetrievedDocument], req: QueryRequest):
    if (
        not documents
        and req.user_custom_retrieve_config
        and not req.user_custom_retrieve_config.return_answer_when_document_empty
    ):
        raise NotFoundRelevantKnowledge("没有检索到相关知识")


def get_retrieve_method(query_strategy: QueryStrategy):
    if query_strategy == QueryStrategy.HYBRID_QUERY or query_strategy == QueryStrategy.TEXT_VECTOR_HYBRID_QUERY:
        return [
            IPD_RAG_RETRIEVE_METHOD_MAP[QueryStrategy.FULL_TEXT_QUERY],
            IPD_RAG_RETRIEVE_METHOD_MAP[QueryStrategy.VECTOR_QUERY],
        ]
    return [IPD_RAG_RETRIEVE_METHOD_MAP[query_strategy]]


def _get_user_token():
    try:
        r = request_val.get()
        if r:
            user_token = r.headers.get("x-user-token", "")
            if user_token:
                return user_token
            user_token = get_user_token_from_cookie(r.cookies)
            return user_token
        else:
            return ""
    except Exception:
        return ""


@safe_trace()
def retrieve_documents_from_ipd_rag(
        question: str, top_k: int, query_strategy: QueryStrategy, knowledge_bases: List[KnowledgeBase] = None,
        ipd_rag_kb_ids: List[str] = None
) -> List[RetrievedDocument]:
    logger.info("Retrieve from ipd rag, kb:%s", knowledge_bases)
    logger.info("Retrieve from ipd rag, ipd rag kb id:%s", ipd_rag_kb_ids)
    ipd_rag_kb_ids = [] if ipd_rag_kb_ids is None else ipd_rag_kb_ids.copy()
    if knowledge_bases:
        ipd_rag_kb_ids.extend([knowledge_base.ipd_rag_kb_id for knowledge_base in knowledge_bases])

    if not ipd_rag_kb_ids:
        return []

    # paas检索一次最多支持最多20个知识库id，需要进行分批
    chunk = 20
    ipd_rag_kb_ids_chunked = [ipd_rag_kb_ids[i:i + chunk] for i in range(0, len(ipd_rag_kb_ids), chunk)]

    results = []
    # 多线程请求，提高效率
    futures = [executor.submit(_request_ipd_rag_retrieve, question, top_k, query_strategy, id_list)
               for id_list in ipd_rag_kb_ids_chunked]

    for future in as_completed(futures, 3):
        try:
            result = future.result()
            results.extend(result)
        except Exception:
            logger.exception("request ipd rag retrieve error")
    return results


def _request_ipd_rag_retrieve(
        question: str,
        top_k: int,
        query_strategy: QueryStrategy,
        ipd_rag_kb_ids: List[str]
) -> List[RetrievedDocument]:
    headers = {
        "X-HW-ID": APP_ID,
        "X-HW-APPKEY": HW_APP_KEY,
        "Content-Type": "application/json",
        "x-auth-token": get_w3_token(),
        "x-user-token": _get_user_token()
    }
    json_entry = {
        "data": json.dumps({"question": question}),
        "retrieve_method": get_retrieve_method(query_strategy),
        "result_type": "org_data",
        "knowledge_repo_ids": ipd_rag_kb_ids,
        "top_k": top_k,
    }
    response = requests.post(IPD_RAG_RETRIEVE_KB_URL, headers=headers, data=json.dumps(json_entry), verify=False)
    logger.info(f"request ipd rag retrieve, param:{json_entry}, response:{response.json()}")
    if not response.ok:
        return []
    return get_search_result_from_ipd(response.json())


def get_search_result_from_ipd(data):
    """
    组装从idp-rag检索得内容
    :param data: idp-rag检索得内容
    :return: 拼接后得内容
    """
    results = []
    for retrieve_data in data.get("query_res"):
        org_data = json.loads(retrieve_data.get("org_data", "{}"))
        extended_metadata = org_data.get("def", {}).get("meta_data", {}).get("extended_metadata", {})
        # 在灵小冰信息来源要展示title
        if "title" not in extended_metadata:
            extended_metadata["title"] = org_data.get("title", '')
        results.append(
            RetrievedDocument(
                text=org_data.get("text", ''),
                metadata=RetrievedDocumentMetadata(
                    source=org_data.get("uri", ''),
                    mtime=org_data.get("meta_data", {}).get("mtime", ''),
                    extended_metadata=extended_metadata,
                ),
                score=float(retrieve_data.get("similarity", 0)),
            )
        )
    return results


def get_llm_stream_answer(
    req: QueryRequest,
    knowledge_base_list: List[KnowledgeBase],
    documents: List[RetrievedDocument],
    synonyms: List[str],
    background_tasks: BackgroundTasks,
    kb_prompt: str = None,
):
    llm_model_name = get_llm_model_name(knowledge_base_list, req)
    yield from llm_stream_answer(req, llm_model_name, documents, synonyms, background_tasks, kb_prompt)


def get_qa_stream_answer(req: QueryRequest, document: RetrievedDocument, answer: str):
    yield "data: " + json.dumps({"content": answer}) + "\n\n"
    yield from format_stream_answer(req, [document], answer)


def paginate_member_knowledge_base_info(session: Session, params, query, member=None):
    return paginate(
        session,
        query,
        params,
        unique=True,
        transformer=lambda kbs: [
            KnowledgeBaseShowInfo(
                name=kb.KnowledgeBase.name,
                sn=kb.KnowledgeBase.sn,
                owner=kb.KnowledgeBase.owner,
                created_at=kb.KnowledgeBase.created_at,
                updated_at=kb.KnowledgeBase.updated_at,
                member_type=kb.member_type
                if kb.member_type
                else get_layer_knowledge_base_permission(kb.KnowledgeBase, member),
                analyzer=kb.KnowledgeBase.analyzer,
                kb_type=kb.KnowledgeBase.kb_type,
                layer_kb_type=kb.KnowledgeBase.layer_kb_type,
                common_dept_code=kb.KnowledgeBase.common_dept_code,
                common_kb_type=kb.KnowledgeBase.common_kb_type,
                domain=kb.KnowledgeBase.domain,
                pbi_project_id=kb.KnowledgeBase.pbi_project_id,
                is_favorite=getattr(kb, "is_favorite", True),
                members=[item.employee_number for item in kb.KnowledgeBase.knowledge_base_members],
                defect_count=kb.defect_count,
                athena_kb_url=ATHENA_KNOWLEDGE_URL_PREFIX + kb.KnowledgeBase.athena_kb_id if kb.KnowledgeBase.athena_kb_id else "",
            )
            for kb in kbs
        ],
    )


@safe_trace()
def paginate_layer_member_knowledge_base_info(session: Session, params, query, is_manager, member):
    return paginate(
        session,
        query,
        params,
        unique=True,
        transformer=lambda kbs: [
            KnowledgeBaseShowInfo(
                name=kb.KnowledgeBase.name,
                sn=kb.KnowledgeBase.sn,
                owner=kb.KnowledgeBase.owner,
                created_at=kb.KnowledgeBase.created_at,
                updated_at=kb.KnowledgeBase.updated_at,
                member_type=MemberType.OWNER
                if is_manager
                else get_layer_knowledge_base_permission(kb.KnowledgeBase, member),
                analyzer=kb.KnowledgeBase.analyzer,
                kb_type=kb.KnowledgeBase.kb_type,
                layer_kb_type=kb.KnowledgeBase.layer_kb_type,
                common_dept_code=kb.KnowledgeBase.common_dept_code,
                common_kb_type=kb.KnowledgeBase.common_kb_type,
                domain=kb.KnowledgeBase.domain,
                pbi_project_id=kb.KnowledgeBase.pbi_project_id,
                is_favorite=kb.is_favorite,
                is_default=kb.KnowledgeBase.is_default,
                is_in_member_list=is_user_in_knowledge_base_member_list(member, kb.KnowledgeBase),
                default_status=kb.KnowledgeBase.default_status,
                defect_count=kb.defect_count,
                athena_kb_url=ATHENA_KNOWLEDGE_URL_PREFIX + kb.KnowledgeBase.athena_kb_id if kb.KnowledgeBase.athena_kb_id else "",
            )
            for kb in kbs
        ],
    )


@safe_trace()
def get_knowledge_base_list(session: Session, req_param: PrivaterKnowledgeBaseListReq):
    query = get_member_knowledge_base(req_param)
    params = Params(page=req_param.page, size=req_param.size)
    return paginate_member_knowledge_base_info(session, params, query)


def set_real_member_type(item, dept_codes, member):
    """
    收藏知识库设置用户在知识库中的角色类型
    用户所在知识库为普通用户且不在知识库成员表中时：
        1、个人知识库：设置用户角色类型为空
        2、分层知识库，通用知识库且为部门公开时：
            1）知识库所在部门不属于用户所在部门时：设置角色类型为空
    :param item: 收藏实体信息
    :param dept_codes: 用户所在部门编号
    :param member: 用户工号
    :return: 设置完成以后的收藏实体信息
    """
    if item.member_type == MemberType.SIMPLE_MEMBER and member not in item.members:
        if item.kb_type == KnowledgeBaseType.PRIVATE:
            item.member_type = item.member_type if member in item.members else None
        if (item.kb_type == KnowledgeBaseType.LAYER
                and item.layer_kb_type == LayerKnowledgeBaseType.COMMON
                and item.common_kb_type == CommonKnowledgeBaseType.DEPARTMENT_SHARED):
            item.member_type = item.member_type if item.common_dept_code in dept_codes else None


def get_favorite_knowledge_base_list(session: Session, req_param: PrivaterKnowledgeBaseListReq):
    member = req_param.member
    validate_user_operation_permission(session, member)
    query = get_favorite_member_knowledge_base(req_param.member, req_param.kb_sn, req_param.name)
    params = Params(page=req_param.page, size=req_param.size)
    res = paginate_member_knowledge_base_info(session, params, query, member)
    dept_codes = portal_util.get_upper_dept_codes(member)
    for item in res.items:
        set_real_member_type(item, dept_codes, member)
    return res


@safe_trace(log_args=True, include_args=["kb_sn", "member"])
def get_knowledge_base_detail(session: Session, kb_sn: str, member: str) -> KnowledgeBaseShowInfo:
    permission_knowledge_base_and_member_type = valid_batch_knowledge_base_member_read_permission(
        session, [kb_sn], member
    )
    if not permission_knowledge_base_and_member_type:
        raise OperationNotPermittedException(f"用户 {member} 无权读取知识库 {kb_sn}")
    knowledge_base, permission_type = (
        permission_knowledge_base_and_member_type[0].knowledge_base,
        permission_knowledge_base_and_member_type[0].member_type,
    )
    if validate_knowledge_base_blacklist_member_exist(member, knowledge_base):
        raise OperationNotPermittedException(f"<{member}>在知识库{kb_sn}黑名单中，无法访问该知识库")
    knowledge_base_favorite = session.exec(
        select(KnowledgeBaseFavorites).where(
            KnowledgeBaseFavorites.kb_id == knowledge_base.id, KnowledgeBaseFavorites.employee_number == member
        )
    ).one_or_none()

    is_favorite = bool(knowledge_base_favorite)

    return KnowledgeBaseShowInfo(
        name=knowledge_base.name,
        sn=knowledge_base.sn,
        owner=knowledge_base.owner,
        created_at=knowledge_base.created_at,
        updated_at=knowledge_base.updated_at,
        member_type=permission_type,
        analyzer=knowledge_base.analyzer,
        kb_type=knowledge_base.kb_type,
        layer_kb_type=knowledge_base.layer_kb_type,
        common_dept_code=knowledge_base.common_dept_code,
        common_kb_type=knowledge_base.common_kb_type,
        domain=knowledge_base.domain,
        pbi_project_id=knowledge_base.pbi_project_id,
        is_favorite=is_favorite,
        max_asset_num=get_max_asset_num(knowledge_base, session),
        id=knowledge_base.id,
        athena_kb_url=ATHENA_KNOWLEDGE_URL_PREFIX + knowledge_base.athena_kb_id if knowledge_base.athena_kb_id else None,
    )


def get_max_asset_num(knowledge_base: KnowledgeBase, session: Session) -> int:
    """
    返回知识库可创建的最大资产数。取知识库配置。全局默认默认、后台管理配置中的最大值
    @param knowledge_base: 知识库
    @param session:  数据库会话
    @return: 最大可创建资产数
    """
    # 知识库配置中的值
    max_asset_num = KnowledgeBaseConfig.model_validate(knowledge_base.config).max_asset_num

    # 全局默认值
    default_num = (
            MAX_PRIVATE_KNOWLEDGE_BASE_ASSET_SIZE
            if knowledge_base.kb_type == KnowledgeBaseType.PRIVATE
            else MAX_LAYER_KNOWLEDGE_BASE_ASSET_SIZE
        )

    # 后台管理配置的值
    config = query_by_config_type_and_name(session, knowledge_base.sn, ServiceConfigType.ASSET_COUNT)
    config_num = int(config.value) if config else None

    num_values = [value for value in [max_asset_num, config_num, default_num] if value is not None]

    return max(num_values)


@safe_trace(log_args=True, include_args=["req"])
def get_layer_knowledge_base_list(session: Session, req: LayerKnowledgeBaseReq):
    query = None
    is_manager = None
    # 通用知识
    if req.common_dept_code is not None:
        # 如果是部门的人,则显示DEPARTMENT_SHARED和ALL_SHARED，如果不是则显示ALL_SHARED
        is_dept_manager = is_department_manager(req.common_dept_code, req.member)
        is_dept_user = is_department_user(req.common_dept_code, req.member)
        query = get_dept_knowledge_base(req, is_dept_manager, is_dept_user)
    # 领域知识
    elif req.domain is not None:
        is_manager = is_domain_manager(req.domain.value, req.member)
        query = get_knowledge_bases(req)
    elif req.pbi_project_id is not None:
        is_manager = is_pbi_manager(req.pbi_project_id, req.member)
        query = get_knowledge_bases(req)
    if req.has_defect and len(req.has_defect) == 1:
        # 查询有质量告警的
        if True in req.has_defect:
            query = query.where(DocumentDefect.id.isnot(None))
        # 查询无质量告警的
        else:
            query = query.where(DocumentDefect.id.is_(None))
    if is_super_root(req.member):
        is_manager = True
    params = Params(page=req.page, size=req.size)
    result = paginate_layer_member_knowledge_base_info(session, params, query, is_manager, req.member)
    return result


def can_create_member(session: Session, kb_sn: str):
    knowledge_base = validate_knowledge_base(session, kb_sn)
    return not (
        knowledge_base.kb_type == KnowledgeBaseType.LAYER
        and (knowledge_base.layer_kb_type == LayerKnowledgeBaseType.DOMAIN)
        | (
            knowledge_base.layer_kb_type == LayerKnowledgeBaseType.COMMON
            and knowledge_base.common_kb_type == CommonKnowledgeBaseType.ALL_SHARED
        )
    )


def get_knowledge_base_list_no_page(
    uid: str, permission: KnowledgeBasePermission, session: Session, keyword: str = None, filter_athena_kb: bool = False
) -> List[KnowledgeBaseInfo]:
    validate_user_operation_permission(session, uid)
    permission_knowledge_bases = get_read_or_operate_permission_knowledge_base(session, uid, permission, keyword)
    if filter_athena_kb:
        permission_knowledge_bases = [kb for kb in permission_knowledge_bases if not kb.athena_kb_id]
    return get_knowledge_bases_info(permission_knowledge_bases)


def get_knowledge_base_page(
        member: str, page: int, size: int, permission: KnowledgeBasePermission, session: Session, keyword: str = None,
        filter_athena_kb: bool = False
) -> Page[KnowledgeBaseInfo]:
    knowledge_bases_info = get_knowledge_base_list_no_page(member, permission, session, keyword, filter_athena_kb)
    params = Params(page=page, size=size)
    return fastapi_pagination.iterables.paginate(knowledge_bases_info, params=params, total=len(knowledge_bases_info))


def get_knowledge_bases_info(knowledge_bases):
    knowledge_bases_info = []
    for knowledge_base in knowledge_bases:
        knowledge_base_info = KnowledgeBaseInfo(
            name=knowledge_base.name,
            sn=knowledge_base.sn,
            owner=knowledge_base.owner,
            created_at=knowledge_base.created_at,
            updated_at=knowledge_base.updated_at,
        )
        knowledge_bases_info.append(knowledge_base_info)
    return knowledge_bases_info


@safe_trace(log_args=True, include_args=["req"])
def get_related_docs(req: QueryRequest, background_tasks: BackgroundTasks, session: Session) -> List[RetrievedDocument]:
    kb_sn_list = req.kb_sn_list if req.kb_sn_list else [req.kb_sn]
    knowledge_base_list = list(batch_validate_knowledge_base(session, kb_sn_list, req.uid))
    return retrieve_documents(req, knowledge_base_list, session, background_tasks=background_tasks)


def delete_knowledge_base(kb_sn: str, operator: str, session: Session):
    # 只有owner才能删除知识库
    knowledge_base = validate_knowledge_base_member_owner_permission(session, kb_sn, operator)
    if knowledge_base.knowledge_base_assets:
        raise AssetsAreAvailableInKnowledgeBase(f"知识库<{kb_sn}>下存在资产，请先删除所有资产")
    session.delete(knowledge_base)
    session.commit()
    if _is_need_sync_ipd_rag(knowledge_base):
        thread_exec_operation_in_ipd_rag(
            session,
            knowledge_base.ipd_rag_kb_sn,
            knowledge_base.name,
            knowledge_base.sn,
            knowledge_base.owner,
            OperationType.DELETE,
        )


def favorite_knowledge_base(kb_sn: str, operator: str, is_favorite: bool, session: Session):
    if is_favorite:
        knowledge_base = valid_batch_knowledge_base_member_read_permission(session, [kb_sn], operator)[0].knowledge_base
        knowledge_base_favorite = KnowledgeBaseFavorites(
            employee_number=operator,
            kb_id=knowledge_base.id,
        )
        session.add(knowledge_base_favorite)
        try:
            session.commit()
            message = f"已收藏知识库 <{knowledge_base.name}> 。"
        except IntegrityError:
            session.rollback()
            message = f"知识库 <{knowledge_base.name}> 已收藏过。"
    else:
        knowledge_base = validate_knowledge_base(session, kb_sn)
        knowledge_base_favorite = session.exec(
            select(KnowledgeBaseFavorites).where(
                KnowledgeBaseFavorites.kb_id == knowledge_base.id, KnowledgeBaseFavorites.employee_number == operator
            )
        ).one_or_none()
        if knowledge_base_favorite:
            session.delete(knowledge_base_favorite)
            session.commit()
            message = f"已取消收藏知识库 <{knowledge_base.name}> 。"
        else:
            message = f"知识库 <{knowledge_base.name}>未被收藏，无需取消 。"

    return message


def batch_cancel_favorite_knowledge_base(kb_sn_list: List[str], operator: str, session: Session):
    if not kb_sn_list:
        raise BatchCancelFavoriteKnowledgeBaseParamFormatError("输入的知识库列表不能为空")

    # 合并查询，获取知识库及其收藏信息
    kb_query = (
        select(KnowledgeBase, KnowledgeBaseFavorites)
        .outerjoin(
            KnowledgeBaseFavorites,
            and_(KnowledgeBase.id == KnowledgeBaseFavorites.kb_id, KnowledgeBaseFavorites.employee_number == operator),
        )
        .where(KnowledgeBase.sn.in_(kb_sn_list))
    )

    results = session.exec(kb_query).all()

    # 将结果分为知识库和收藏信息
    favorite_knowledge_bases = [result[0] for result in results]
    favorites = [result[1] for result in results if result[1] is not None]

    favorite_knowledge_base_map = {kb.sn: kb for kb in favorite_knowledge_bases}
    favorites_map = {fav.kb_id: fav for fav in favorites}

    canceled_knowledge_bases = []
    unfavorable_knowledge_bases = []

    for kb_sn in kb_sn_list:
        knowledge_base = favorite_knowledge_base_map.get(kb_sn)
        if knowledge_base:
            knowledge_base_favorite = favorites_map.get(knowledge_base.id)
            if knowledge_base_favorite:
                session.delete(knowledge_base_favorite)
                canceled_knowledge_bases.append(knowledge_base.name)
            else:
                unfavorable_knowledge_bases.append(knowledge_base.name)
        else:
            unfavorable_knowledge_bases.append(kb_sn)

    session.commit()

    message = ""
    if canceled_knowledge_bases:
        message += f"已取消收藏{len(canceled_knowledge_bases)}个知识库, 知识库名如下：\n"
        message += str(canceled_knowledge_bases) + "\n"
    if unfavorable_knowledge_bases:
        message += "以下知识库未被收藏，无需取消：\n"
        message += str(unfavorable_knowledge_bases) + "\n"

    return message


def update_knowledge_base_name(session: Session, kb_sn: str, name: str, operator: str):
    knowledge_base = validate_knowledge_base_member_operate_permission(session, kb_sn, operator)
    if knowledge_base.kb_type != KnowledgeBaseType.PRIVATE:
        validate_knowledge_base_name(
            knowledge_base.common_dept_code, knowledge_base.domain, knowledge_base.pbi_project_id, name, session
        )
    knowledge_base.name = name
    session.add(knowledge_base)
    session.commit()


# 模糊人员信息查询
def member_information_search(lang: str, search_value: str, search_type: str, page_size: str, page: str):
    return search_member_information(lang, search_value, search_type, page_size, page)


# 新增知识库member
def create_knowledge_base_member(req: CreateUpdateMemberReq, session: Session):
    # 对工号进行检查
    user_name = get_user_name(req.employee_number)
    if req.member_type == MemberType.OWNER:
        raise KnowledgeBaseMemberValidationException("新增知识库的成员类型不能为owner")
    # 验证操作者权限
    knowledge_base = validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.operator)
    _validate_knowledge_base_member_exist(req.employee_number, knowledge_base)
    new_knowledge_base_member = KnowledgeBaseMember(
        employee_number=req.employee_number,
        employee_name=user_name,
        member_type=req.member_type,
        kb_id=knowledge_base.id,
    )
    knowledge_base.knowledge_base_members.append(new_knowledge_base_member)
    session.add(knowledge_base)
    session.commit()
    rc.set(
        old_user_keyspace.resolve(req.employee_number),
        IS_OLD_USER,
        px=24 * 60 * 60 * 1000,
    )


# 批量新增知识库member
def create_batch_knowledge_base_member(
    req: CreateUpdateMultipleMemberReq, session: Session
) -> Tuple[Set[str], Set[str]]:
    failed_member_set = set()
    succeed_member_set = set()
    # 验证操作者权限
    knowledge_base = validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.operator)
    # 验证管理员和普通成员参数列表
    if not req.simple_employee_number_list and not req.control_employee_number_list:
        raise InvalidParamError("普通成员列表和管理员成员列表为空")
    # 过滤不合法的工号
    pattern = r"^([A-Za-z]\d+|[A-Za-z]{3}\d+)$"
    simple_employee_number_set = {
        simple_employee_number
        for simple_employee_number in req.simple_employee_number_list
        if re.fullmatch(pattern, simple_employee_number)
    }
    control_employee_number_set = {
        control_employee_number
        for control_employee_number in req.control_employee_number_list
        if re.fullmatch(pattern, control_employee_number)
    }

    for simple_employee_number in simple_employee_number_set:
        try:
            get_user_name(simple_employee_number)
            knowledge_base_add_member(session, MemberType.SIMPLE_MEMBER, simple_employee_number, knowledge_base)
            succeed_member_set.add(simple_employee_number)
        except Exception:  # noqa
            logger.warning("添加普通用户%s失败，原因是用户不存在", simple_employee_number)
            failed_member_set.add(simple_employee_number)

    for control_employee_number in control_employee_number_set:
        try:
            get_user_name(control_employee_number)
            knowledge_base_add_member(session, MemberType.CONTROL_MEMBER, control_employee_number, knowledge_base)
            succeed_member_set.add(control_employee_number)
        except Exception:  # noqa
            logger.warning("添加管理员%s失败，原因是用户不存在", control_employee_number)
            failed_member_set.add(control_employee_number)

    session.commit()
    return failed_member_set, succeed_member_set


def knowledge_base_add_member(
    session: Session, member_type: MemberType, employee_id: str, knowledge_base: KnowledgeBase
):
    knowledge_base_members = get_knowledge_base_members(session, knowledge_base.sn)
    knowledge_base_members_ids = [member.employee_number.lower() for member in knowledge_base_members]
    if employee_id.lower() not in knowledge_base_members_ids:
        new_knowledge_base_member = KnowledgeBaseMember(
            employee_number=employee_id, member_type=member_type, kb_id=knowledge_base.id
        )
        knowledge_base.knowledge_base_members.append(new_knowledge_base_member)
        session.add(knowledge_base)
        rc.set(
            old_user_keyspace.resolve(employee_id.lower()),
            IS_OLD_USER,
            px=24 * 60 * 60 * 1000,
        )
    else:
        user_record = get_user_record_from_knowledge_base_member(session, knowledge_base.sn, employee_id)
        if user_record.member_type != MemberType.OWNER:
            user_record.member_type = member_type
            session.add(knowledge_base)


def _validate_knowledge_base_member_exist(employee_number: str, knowledge_base: KnowledgeBase):
    for knowledge_base_member in knowledge_base.knowledge_base_members:
        if employee_number.lower() == knowledge_base_member.employee_number.lower():
            raise DuplicateKnowledgeBaseMemberException(f"知识库已存在成员 <{employee_number}> .")


def get_knowledge_base_member_list(kb_sn: str, search_txt: str, session: Session):
    query = (
        select(KnowledgeBaseMember)
        .join(KnowledgeBase, KnowledgeBaseMember.kb_id == KnowledgeBase.id, isouter=True)
        .where(KnowledgeBase.sn == kb_sn)
        .order_by(KnowledgeBaseMember.member_type, KnowledgeBaseMember.employee_number)
    )
    if search_txt:
        search_txt = search_txt.strip()
        query = query.where(
            or_(
                KnowledgeBaseMember.employee_number.ilike(f"%{search_txt}%"),
                KnowledgeBaseMember.employee_name.ilike(f"%{search_txt}%"),
            )
        )
    member_infos = paginate(
        session,
        query,
        transformer=lambda members: [
            KnowledgeBaseMemberShowInfo(
                employee_number=member.employee_number,
                name=member.employee_name if member.employee_name else "",
                member_type=member.member_type,
            )
            for member in members
        ],
    )
    for member_info in member_infos.items:
        if member_info.name:
            continue
        try:
            user_name = get_user_name(member_info.employee_number)
        except UserInformationSearchException as uise:
            logger.warning("知识库<%s>成员<%s>不存在，原因是%s。", kb_sn, member_info.employee_number, uise)
            continue
        if user_name is None:
            user_name = "公共账号"
        member_info.name = user_name

        knowledge_base_members = session.exec(
            select(KnowledgeBaseMember).where(KnowledgeBaseMember.employee_number == member_info.employee_number)
        ).all()
        for knowledge_base_members in knowledge_base_members:
            knowledge_base_members.employee_name = user_name
        session.add(knowledge_base_members)
        session.commit()
    return member_infos


def get_self_member_type(kb_sn: str, employee_number: str, session: Session):
    return valid_batch_knowledge_base_member_read_permission(session, [kb_sn], employee_number)[0].member_type


def get_fuzzy_match_knowledge_base(kb_info: str, employee_number: str, session: Session = Session):
    validate_user_operation_permission(session, employee_number)
    # 匹配结尾是 "_ + 8 个十六进制字符" 的正则表达式
    kb_sn_pattern = r"_\w{8}$"
    # kb_info是sn号
    if re.search(kb_sn_pattern, kb_info):
        # 查询用户有没有权限访问, 如果用户不为知识库成员则无访问权限
        valid_batch_knowledge_base_member_read_permission(session, [kb_info], employee_number)
        return FuzzyMatchKnowledgeBaseResponse(most_similar_knowledge_base=kb_info)
    # 获取到所有的知识库
    permission_knowledge_bases = get_read_or_operate_permission_knowledge_base(
        session, employee_number, KnowledgeBasePermission.READ
    )
    return get_most_and_possible_match_knowledge_base(permission_knowledge_bases, kb_info)


def get_most_and_possible_match_knowledge_base(permission_knowledge_bases, kb_info):
    most_similar_knowledge_base = None
    possible_similar_knowledge_bases = []
    # 取前10个模糊查询的其他知识库
    possible_similar_knowledge_base_count = 0
    for permission_knowledge_base in permission_knowledge_bases:
        sn = permission_knowledge_base.sn
        name = permission_knowledge_base.name
        # 模糊匹配失败
        if not any([re.search(kb_info, sn, re.IGNORECASE), re.search(kb_info, name, re.IGNORECASE)]):
            continue
        if most_similar_knowledge_base is None:
            most_similar_knowledge_base = sn
        else:
            if possible_similar_knowledge_base_count >= POSSIBLE_SIMILAR_KNOWLEDGE_BASE_COUNT:
                break
            possible_similar_knowledge_base_count += 1
            possible_similar_knowledge_bases.append(_get_possible_similar_knowledge_base(permission_knowledge_base))
    if most_similar_knowledge_base is None:
        raise KnowledgeBaseNotExistsError(f"没有知识库匹配<{kb_info}>")
    return FuzzyMatchKnowledgeBaseResponse(
        most_similar_knowledge_base=most_similar_knowledge_base,
        possible_similar_knowledge_bases=possible_similar_knowledge_bases,
    )


def _get_possible_similar_knowledge_base(permission_knowledge_base):
    kb_sn = permission_knowledge_base.sn
    kb_name = permission_knowledge_base.name
    kb_type = permission_knowledge_base.kb_type
    layer_kb_type = permission_knowledge_base.layer_kb_type
    domain = permission_knowledge_base.domain
    pbi_project_id = permission_knowledge_base.pbi_project_id
    dept_name = None
    if permission_knowledge_base.common_dept_code is not None:
        dept_name = get_dept_name(permission_knowledge_base.common_dept_code)
    return KnowledgeBaseTypeInfo(
        kb_name=kb_name,
        kb_sn=kb_sn,
        kb_type=kb_type,
        layer_kb_type=layer_kb_type,
        dept_name=dept_name,
        domain=domain,
        pbi_project_id=pbi_project_id,
    )


def batch_delete_knowledge_base_member(req: DeleteMemberReq, session: Session):
    """删除member(不能删除owner)"""
    # 验证操作者权限
    validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.operator)
    employee_members = [employee.lower() for employee in req.employee_numbers]
    # 子查询
    subq = session.query(KnowledgeBase).filter(KnowledgeBase.sn == req.kb_sn).subquery()
    session.execute(
        delete(KnowledgeBaseMember).where(
            func.lower(KnowledgeBaseMember.employee_number).in_(employee_members),
            KnowledgeBaseMember.kb_id == subq.c.id,
            KnowledgeBaseMember.member_type != MemberType.OWNER,
        )
    )
    session.commit()
    for employee_member in employee_members:
        rc.delete(old_user_keyspace.resolve(employee_member))


# 更新member
def update_knowledge_base_member(req: CreateUpdateMemberReq, session: Session):
    # 对工号进行检查
    get_user_name(req.employee_number)
    if req.member_type.value == MemberType.OWNER:
        raise KnowledgeBaseMemberValidationException("更新知识库的成员类型不能为owner.")
    # 验证操作者权限
    validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.operator)
    knowledge_base_member = session.exec(
        select(KnowledgeBaseMember)
        .join(KnowledgeBase, KnowledgeBase.id == KnowledgeBaseMember.kb_id)
        .where(
            KnowledgeBase.sn == req.kb_sn,
            func.lower(KnowledgeBaseMember.employee_number) == req.employee_number.lower(),
        )
    ).one_or_none()
    if not knowledge_base_member:
        raise KnowledgeBaseMemberNotExistsError(f"<{req.operator}>不是知识库<{req.kb_sn}>成员")
    knowledge_base_member.member_type = req.member_type
    session.add(knowledge_base_member)
    session.commit()


def try_it_out(kb_sn: str, uid: str):
    data = {
        "sender": uid.lower(),
        "msgBody": f"rag {kb_sn}",
        "msgType": "text",
        "timestamp": str(int(time.time())),
        "botUser": XIAOLUBAN_NAME,
    }
    res = requests.post(TRY_IT_OUT_URL, json=data, verify=False)
    if not res.ok:
        raise Exception(f"Failed to try it out for <{kb_sn}>, error: {res.text}")


def ppt_helper(ppthelper_request: PptHelperRequest, session: Session, cookie: str) -> List[PptHelperResponse]:
    # 根据入参cookie识别并寻找有权限的知识库，文件夹
    kb_kba_map = get_kbsn_asset(cookie)
    kb_sn_list = [str(kb) for kb in kb_kba_map]
    thread_exec_kb_sn_list_to_database(kb_sn_list, context.data.get("X-Request-ID"))
    analyzer_to_embedding_model_to_vector_stores = get_grouped_vector_stores_by_knowledge_base_and_asset(
        session, kb_kba_map
    )

    retrieved_documents = []
    for analyzer, embedding_model_to_vector_stores in analyzer_to_embedding_model_to_vector_stores.items():
        documents = get_vector_store_manager().retrieve(
            ppthelper_request.search_txt,
            60,
            embedding_model_to_vector_stores,
            analyzer=analyzer,
        )
        retrieved_documents.extend(documents)
    retrieved_documents = filter_same_document(retrieved_documents)
    retrieved_documents = sorted(retrieved_documents, key=lambda x: x.score, reverse=True)[:60]

    # 将搜索结果转换为ppt助手插件需要的格式
    ppthelper_data = []
    valid_folder_list = list(collapse(kb_kba_map.values()))
    if ppthelper_request.custom_params[0].custom_param_value.strip() == "":
        # 搜索全量数据
        ppt_document = set()  # 搜索页面:相同文件只显示一个
        for retrieved_document in retrieved_documents:
            ppt_source = retrieved_document.metadata.source
            folder = retrieved_document.metadata.extended_metadata["folder_uuid"]
            if ppt_source in ppt_document or folder not in valid_folder_list:
                continue
            # 转换格式
            ppt_response = get_ppt_response(retrieved_document, ppthelper_request)
            ppthelper_data.append(ppt_response)
            ppt_document.add(ppt_source)
    else:
        # 指定搜索某一个文件
        page_num_set = set()  # 一个文件一页只返回一次
        for retrieved_document in retrieved_documents:
            ppt_id = retrieved_document.metadata.extended_metadata["doc_attach_id"]
            page_num = retrieved_document.metadata.extended_metadata["page_number"]
            if ppt_id != ppthelper_request.custom_params[0].custom_param_value or page_num in page_num_set:
                continue
            if retrieved_document.metadata.extended_metadata["folder_uuid"] not in valid_folder_list:
                raise Exception("越权查看文件详情")
            # 转换格式
            ppt_response = get_ppt_response(retrieved_document, ppthelper_request)
            ppthelper_data.append(ppt_response)
            page_num_set.add(page_num)
    return ppthelper_data


def get_kbsn_asset(cookie: str) -> dict[str, List[str]]:
    response = requests.get("http://ai.libing.huawei.com/export-desktop/user/queryForRag", headers={"Cookie": cookie})
    if not response.ok:
        raise ExternalApiResponseException

    response_json = response.json()
    kb_kba_map = defaultdict(list)
    for folders in response_json["result"]:
        for folder in folders["readFolders"] + folders["writeFolders"]:
            kb_kba_map[folders["ragKbSn"]].append(folder["folderUuid"])

    if not kb_kba_map:
        raise ExternalApiResponseException("无可搜索的文件，请先上传文件")
    return kb_kba_map


def get_ppt_response(retrieved_document, pptreq):
    attach_dir_title = retrieved_document.metadata.extended_metadata["title"]
    # 页码
    attach_page_no = retrieved_document.metadata.extended_metadata["page_number"]
    doc_attach_id = retrieved_document.metadata.extended_metadata["doc_attach_id"]
    # 文件名称(含后缀) DOC_ATTACH_NAME
    doc_attach_name = retrieved_document.metadata.extended_metadata["file_name"]
    # 文件名称(不含后缀) + 高亮 DOC_ATTACH_TEXT
    doc_attach_text = doc_attach_name[: doc_attach_name.rfind(".")].replace(
        pptreq.search_txt, "<em>" + pptreq.search_txt + "</em>"
    )
    download_url = retrieved_document.metadata.extended_metadata["download_url"]
    thumbnail_url = retrieved_document.metadata.extended_metadata["image_url"]
    # 发布时间 (这里取的更新时间) # noqa
    publish_time = retrieved_document.metadata.extended_metadata["last_modified"]
    # 高亮部分
    ppt_highlight = retrieved_document.text.replace(pptreq.search_txt, "<em>" + pptreq.search_txt + "</em>")
    highlight = Highlight(ATTACH_PARAGRAPH=[ppt_highlight[0:100]])
    return PptHelperResponse(
        ATTACH_DIR_TITLE=attach_dir_title,
        ATTACH_PAGE_NO=attach_page_no,
        DOC_ATTACH_ID=doc_attach_id,
        DOC_ATTACH_NAME=doc_attach_name,
        DOC_ATTACH_TEXT=doc_attach_text,
        DOWNLOAD_URL=download_url,
        PUBLISH_TIME=publish_time,
        THUMBNAIL_URL=thumbnail_url,
        highlight=highlight,
    )


def _validate_feedback(request_log: RequestResponseLog):
    if not request_log:
        raise RequestIdNotExistsException


def _valid_feedback_request_param(request_id: str = None, question_id: str = None):
    if not request_id and not question_id:
        raise FeedbackRequestParamException("问答反馈请求错误，Request ID或 Question ID同时为空")


def feedback_answer_to_db(feedback: FeedbackAnswer, session: Session, background_tasks: BackgroundTasks):
    _valid_feedback_request_param(feedback.request_id, feedback.question_id)
    if feedback.request_id:
        request_log = session.exec(
            select(RequestResponseLog).where(RequestResponseLog.request_id == feedback.request_id)
        ).one_or_none()
    else:
        request_log = session.exec(
            select(RequestResponseLog).where(RequestResponseLog.question_id == feedback.question_id)
        ).one_or_none()

    _validate_feedback(request_log)

    if feedback.acceptance is None or feedback.acceptance:
        request_log.answer_source = None
        request_log.answer_user_want = None
        request_log.acceptance = feedback.acceptance
        request_log.dislike_tag = None
        request_log.process_status = None
        request_log.ungoverned_corpus_links = None
    else:
        request_log.answer_source = feedback.answer_source
        request_log.answer_user_want = feedback.expected_answer
        request_log.acceptance = feedback.acceptance
        request_log.score = feedback.score
        request_log.dislike_tag = feedback.dislike_tag
        if feedback.dislike_tag and feedback.dislike_tag in KNOWLEDGE_BASE_OWNER_DISLIKE_TAGS:
            request_log.process_status = False
        request_log.ungoverned_corpus_links = feedback.ungoverned_corpus_links

    session.add(request_log)
    session.commit()
    background_tasks.add_task(send_log_to_aigc_record, feedback.request_id)


def update_response_log_process_status(req: UpdateFeedbackReq, session: Session):
    _valid_feedback_request_param(req.request_id, req.question_id)
    if req.request_id:
        request_log = session.exec(
            select(RequestResponseLog).where(RequestResponseLog.request_id == req.request_id)
        ).one_or_none()
    else:
        request_log = session.exec(
            select(RequestResponseLog).where(RequestResponseLog.question_id == req.question_id)
        ).one_or_none()
    _validate_feedback(request_log)
    if req.process_status is not None:
        request_log.process_status = req.process_status
    if req.dislike_tag is not None and req.dislike_tag != request_log.dislike_tag:
        request_log.dislike_tag = req.dislike_tag
        if req.dislike_tag in KNOWLEDGE_BASE_OWNER_DISLIKE_TAGS:
            request_log.process_status = False
        else:
            request_log.process_status = None
    if req.expected_answer is not None:
        request_log.answer_user_want = req.expected_answer
    session.add(request_log)
    session.commit()


def get_department_asset_info(
    department_request: DepartmentRequest, session: Session
) -> DepartmentKnowledgeBaseInformationAnswer:
    _valid_request_time(department_request.start_time, department_request.end_time)
    kb_count = get_user_knowledge_base_numbers(
        department_request.uid_list, session, department_request.start_time, department_request.end_time
    )
    kba_count = get_user_knowledge_base_asset_numbers(
        department_request.uid_list, session, department_request.start_time, department_request.end_time
    )
    index_list = get_user_vector_stores(
        department_request.uid_list, session, department_request.start_time, department_request.end_time
    )
    document_count = get_vector_store_manager().count(index_list)
    return DepartmentKnowledgeBaseInformationAnswer(
        kb_count=kb_count, kba_count=kba_count, document_count=document_count
    )


def _valid_request_time(start_time: datetime, end_time: datetime):
    if not start_time and end_time:
        raise TimeIntervalException("起始时间为空")
    if start_time and not end_time:
        raise TimeIntervalException("终止时间为空")
    if not start_time and not end_time:
        return
    if end_time < start_time:
        raise TimeIntervalException("起始时间大于终止时间")


def generate_related_questions_by_user_question(req: QueryRequest, session: Session):
    generate_question_llm = RagLLM()
    kb_sn_list = req.kb_sn_list if req.kb_sn_list else [req.kb_sn]
    knowledge_base_list = list(batch_validate_knowledge_base(session, kb_sn_list, req.uid))
    related_documents = retrieve_documents(req, knowledge_base_list, session, collect_info=False)
    validate_retrieved_documents(related_documents, req)

    context_str = ""
    for idx, document in enumerate(related_documents):
        context_str += f"知识片段{idx + 1}: {document.text}\n"
    query = (
        get_prompt(default_prompt=QUESTION_GENERATION_PROMPT, prompt_id=QUESTION_GENERATION_PROMPT_ID)
        .replace("{{ top_k }}", str(req.top_k))
        .replace("{{ context }}", context_str)
        .replace("{{ question }}", req.question)
    )
    questions = select_llm_and_get_answer([req.kb_sn], generate_question_llm, query, collect_info=False)
    try:
        return questions.strip().split("\n")
    except Exception as e:
        logger.exception(e)
        return []


def export_document_info_by_kb_sn(kb_sn: str, session: Session):
    knowledge_base_document_info_list = get_knowledge_base_document_info(session, kb_sn)
    document_data_list = []
    for document_info in knowledge_base_document_info_list:
        document_data = list(document_info)
        document_data[1] = MAPPING_ASSET_TYPE_TO_LABEL.get(document_data[1], document_data[1].name)
        document_data[3] = document_info[3] if document_info[3] else document_info[5]
        document_data[6] = document_data[6].get("author", "")
        document_data_list.append(document_data)
    df = pd.DataFrame(
        data=document_data_list,
        columns=["资产名", "资产类型", "资产链接", "文档名", "文档更新时间", "文档链接", "作者"],
    )
    df["文档更新时间"] = df["文档更新时间"].apply(lambda a: pd.to_datetime(a).strftime("%Y-%m-%d %H:%M:%S"))
    return create_excel_file_stream(df), KNOWLEDGE_BASE_DOCUMENTS_METADATA_FILE


def export_knowledge_base_info_by_kb_sn(req: ExportKnowledgeBaseInfoRequest, session: Session):
    validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.uid)
    if req.type == KnowledgeBaseInfoType.DOCUMENT:
        file_stream, file_name = export_document_info_by_kb_sn(req.kb_sn, session)
    elif req.type == KnowledgeBaseInfoType.ASSET:
        file_stream, file_name = export_asset_info_by_kb_sn(req.kb_sn, session)
    else:
        raise ExportKnowledgeBaseInfoTypeException(f"不支持的知识库信息导出文件类型<{req.type}>")
    return file_stream, file_name


def export_asset_info_by_kb_sn(kb_sn: str, session: Session):
    knowledge_base_asset_info_list = get_knowledge_base_asset_info(session, kb_sn)
    asset_data_list = []
    for asset_info in knowledge_base_asset_info_list:
        asset_data = list(asset_info)
        asset_data[1] = MAPPING_ASSET_TYPE_TO_LABEL.get(asset_data[1], asset_data[1].name)
        asset_data_list.append(asset_data)
    df = pd.DataFrame(data=asset_data_list, columns=["资产名", "资产类型", "资产链接", "资产创建时间"])
    df["资产创建时间"] = df["资产创建时间"].apply(lambda a: pd.to_datetime(a).strftime("%Y-%m-%d %H:%M:%S"))
    return create_excel_file_stream(df), KNOWLEDGE_BASE_ASSETS_METADATA_FILE


@safe_trace()
def validate_domain_manager_permission(member, domain):
    if not is_domain_manager(domain, member) and not is_super_root(member):
        raise OperationNotPermittedException(f"<{member}>非领域知识库管理员或超级管理员，无权限对领域知识库执行该操作")


def validate_dept_manager_permission(member, dept_code):
    if not is_department_manager(dept_code, member) and not is_super_root(member):
        raise OperationNotPermittedException(f"<{member}>非部门知识库管理员，无权限对部门知识库执行该操作")


def create_excel_file_stream(data: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    # 创建一个 BytesIO 对象
    output = BytesIO()
    # # 使用 Pandas 将 DataFrame 写入 BytesIO 对象
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        data.to_excel(writer, sheet_name=sheet_name, index=False)
        workbook = writer.book
        cell_format = workbook.add_format({"text_wrap": True})
        worksheets = writer.sheets
        for worksheet in worksheets.values():
            # 设置列宽
            worksheet.set_column(0, data.shape[1] - 1, EXCEL_WIDTH, cell_format)
            # 设置行高
            for row in range(len(data) + 1):  # 包括表头行
                worksheet.set_row(row, EXCEL_HEIGHT, cell_format)
    output.seek(0)
    return output.read()


def store_stopword_or_synonym_to_kb(req: UploadStopwordOrSynonymReq, session: Session):
    knowledge_base = validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.uid)
    repeat_stopwords, repeat_synonyms = None, None
    if req.stopwords:
        repeat_stopwords = store_stopwords_or_synonyms_to_kb(
            session, knowledge_base, req.stopwords, StopWordOrSynonymType.STOPWORD
        )
    if req.synonyms:
        repeat_synonyms = store_stopwords_or_synonyms_to_kb(
            session, knowledge_base, req.synonyms, StopWordOrSynonymType.SYNONYM
        )

    if repeat_stopwords and repeat_synonyms:
        return f"知识库{req.kb_sn} 添加部分同义词和停用词成功，其中停用词: {repeat_stopwords} 重复以及同义词: {repeat_synonyms}"
    if repeat_stopwords:
        return f"知识库{req.kb_sn} 添加部分同义词或停用词成功，其中停用词: {repeat_stopwords} 重复"
    if repeat_synonyms:
        return f"知识库{req.kb_sn} 添加部分同义词或停用词成功，其中同义词: {repeat_synonyms} 重复"
    return f"知识库{req.kb_sn} 成功添加停用词或同义词"


def store_stopwords_or_synonyms_to_kb(
    session: Session, knowledge_base: KnowledgeBase, words: List[str], word_type: StopWordOrSynonymType
):
    exist_words = set(get_knowledge_base_synonyms_or_stopwords(knowledge_base, word_type))
    upload_words = set(words)

    if word_type == StopWordOrSynonymType.SYNONYM:
        valid_synonyms(upload_words)
    knowledge_base.knowledge_base_stopword_and_synonyms.extend([
        KnowledgeBaseStopwordAndSynonym(word_type=word_type, word=word) for word in upload_words - exist_words
    ])
    session.add(knowledge_base)
    session.commit()
    repeat_words = list(upload_words & exist_words)
    return repeat_words[:100]


def get_knowledge_base_synonyms_or_stopwords(knowledge_base: KnowledgeBase, word_type: StopWordOrSynonymType):
    return [
        word_line.word
        for word_line in knowledge_base.knowledge_base_stopword_and_synonyms
        if word_line.word_type == word_type
    ]


def valid_synonyms(synonyms: Set):
    for synonym in synonyms:
        if "," not in synonym:
            raise SynonymFormatException(f"同义词: {synonym} 格式错误，单词之间需以','进行分隔")


def delete_stopwords_or_synonyms_from_kb(session: Session, req: DeleteStopwordOrSynonymReq):
    validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.uid)
    if req.word_type == StopWordOrSynonymType.STOPWORD:
        stopword = get_stopword_or_synonym_by_id(session, req.word_id, req.word_type)
        if not stopword:
            raise StopwordNotExistException("删除停用词失败，原因是该停用词不存在")
        session.delete(stopword)
        session.commit()
        return f"删除停用词: <{stopword.word}> 成功"
    synonym = get_stopword_or_synonym_by_id(session, req.word_id, req.word_type)
    if not synonym:
        raise SynonymNotExistException("删除同义词失败，原因是该同义词不存在")
    session.delete(synonym)
    session.commit()
    return f"删除同义词: <{synonym.word}> 成功"


def get_stopwords_or_synonyms_by_kb_sn(session: Session, req: QueryStopwordOrSynonymReq):
    valid_batch_knowledge_base_member_read_permission(session, [req.kb_sn], req.uid)
    query = get_stopword_or_synonym(req)
    return paginate(
        session,
        query,
        Params(page=req.page, size=req.size),
        transformer=lambda words: [StopwordOrSynonymInfo(word_id=word.id, word=word.word) for word in words],
    )


def update_stop_or_synonym_by_id(session: Session, req: UpdateStopwordOrSynonymReq):
    validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.uid)
    if req.word_type == StopWordOrSynonymType.STOPWORD:
        stopword = get_stopword_or_synonym_by_id(session, req.word_id, req.word_type)
        if not stopword:
            raise StopwordNotExistException("更新停用词失败，原因是该停用词不存在")
        tmp_word = stopword.word
        stopword.word = req.word
        session.add(stopword)
        session.commit()
        return f"成功将停用词: <{tmp_word}> 更新为 <{req.word}>"
    valid_synonyms({req.word})
    synonym = get_stopword_or_synonym_by_id(session, req.word_id, req.word_type)
    if not synonym:
        raise SynonymNotExistException("更新同义词失败，原因是该同义词不存在")
    tmp_word = synonym.word
    synonym.word = req.word
    session.add(synonym)
    session.commit()
    return f"成功将同义词: <{tmp_word}> 更新为 <{req.word}>"


def change_knowledge_base_analyzer_to_ik_analyzer(
    session: Session, kb_sn: str, member: str
) -> List[VectorizationJobStatus]:
    knowledge_base = validate_knowledge_base_member_operate_permission(session, kb_sn, member)

    if knowledge_base.analyzer != Analyzer.STANDARD_ANALYZER:
        raise ReindexKnowledgeBaseError(f"知识库 <{knowledge_base.sn}> 分词器不是standard_analyzer，不支持知识迁移")
    job_ids_and_kba_names = reindex_knowledge_base(session, knowledge_base)
    return [
        VectorizationJobStatus(
            vectorization_job_id=job_id, message=f"正在迁移知识库 <{knowledge_base.sn}> 内资产 <{kba_name}> 的知识."
        )
        for job_id, kba_name in job_ids_and_kba_names
    ]


def synchronize_knowledge_base_synonym_stopword(
    session: Session, kb_sn: str, member: str
) -> List[VectorizationJobStatus]:
    knowledge_base = validate_knowledge_base_member_operate_permission(session, kb_sn, member)
    if knowledge_base.analyzer != Analyzer.IK_ANALYZER:
        raise SynchronousKnowledgeBaseSynonymAndStopwordError(
            f"知识库 <{knowledge_base.sn}> 分词器不是ik_analyzer，不支持同义词和停用词同步"
        )
    job_ids_and_kba_names = reindex_knowledge_base(session, knowledge_base)
    return [
        VectorizationJobStatus(
            vectorization_job_id=job_id,
            message=f"正在同步知识库 <{knowledge_base.sn}> 内资产 <{kba_name}> 的同义词和停用词列表.",
        )
        for job_id, kba_name in job_ids_and_kba_names
    ]


def reindex_knowledge_base(session: Session, knowledge_base: KnowledgeBase):
    # 校验知识库下是否有正在运行的资产
    running_asset = get_knowledge_base_running_knowledge_base_asset(session, knowledge_base.sn)
    if running_asset:
        running_asset_names = " ".join([asset.name for asset in running_asset])
        raise KnowledgeBaseAssetJobIsRunning(
            f"知识库 <{knowledge_base.sn}> 中 <{running_asset_names}> 存在正在运行的任务"
        )

    # 检索初始化成功且无运行任务的资产
    valid_knowledge_base_assets = get_init_and_not_running_knowledge_base_asset_list(session, knowledge_base.sn)
    if not valid_knowledge_base_assets:
        raise KnowledgeBaseAssetNotInitializedException(f"知识库 <{knowledge_base.sn}>没有已初始化的资产")
    job_ids_and_kba_names: List[Tuple[str, str]] = []
    for knowledge_base_asset in valid_knowledge_base_assets:
        vectorization_job = AutoJobInstances(status=JobStatus.PENDING, job_type=VectorizationJobType.REINDEX)
        knowledge_base_asset.auto_job_instances.append(vectorization_job)
        job_ids_and_kba_names.append((str(vectorization_job.id), knowledge_base_asset.name))
        session.add(knowledge_base_asset)
    session.commit()
    return job_ids_and_kba_names


def update_kb_config(session: Session, kb_sn: str, member: str, config: KnowledgeBaseConfig) -> str:
    knowledge_base = validate_knowledge_base_member_operate_permission(session, kb_sn, member)
    if config.prompt:
        _valid_llm_prompt(config.prompt, knowledge_base.analyzer)
    config.max_asset_num = KnowledgeBaseConfig.model_validate(knowledge_base.config).max_asset_num
    knowledge_base.config = json.loads(config.model_dump_json())
    session.add(knowledge_base)
    session.commit()
    return f"Update knowledge base <{kb_sn}> config success."


def get_kb_config(session: Session, kb_sn: str, member: str) -> KnowledgeBaseConfig:
    knowledge_base = valid_batch_knowledge_base_member_read_permission(session, [kb_sn], member)[0].knowledge_base
    return KnowledgeBaseConfig.model_validate(knowledge_base.config, strict=False)


def _valid_llm_prompt(prompt: str, analyzer: Analyzer):
    question_format, context_format, synonym_format = "{{ question }}", "{{ context }}", "{{ synonym }}"
    if analyzer == Analyzer.STANDARD_ANALYZER:
        if prompt.count(question_format) != 1 or prompt.count(context_format) != 1:
            raise CustomPromptFormatError(
                f"知识库为standard_analyzer分词器的prompt格式错误，需同时包含 {question_format} "
                f"和 {context_format} 且只出现一次"
            )
    else:
        if prompt.count(question_format) != 1 or prompt.count(context_format) != 1 or prompt.count(synonym_format) != 1:
            raise CustomPromptFormatError(
                f"知识库为ik_analyzer分词器的prompt格式错误，需同时包含 {question_format}、 {context_format} "
                f"以及 {synonym_format} 且只出现一次"
            )


def get_default_kb_config(session: Session, kb_sn: str, member: str) -> KnowledgeBaseConfig:
    knowledge_base = valid_batch_knowledge_base_member_read_permission(session, [kb_sn], member)[0].knowledge_base
    prompt = SYNONYM_PROMPT if knowledge_base.analyzer == Analyzer.IK_ANALYZER else DEFAULT_PROMPT_TEMPLATE
    return KnowledgeBaseConfig(
        top_k=DEFAULT_TOP_K,
        document_score_threshold=DOCUMENT_THRESHOLD_SCORE,
        rerank_model=RerankModel.BASIC,
        query_strategy=QueryStrategy.HYBRID_QUERY,
        prompt=prompt,
        llm_model=LLM_MODEL,
    )


def build_tree(structure: Dict[str, List[str]], code: str, dept_code_name_dict: Dict[str, str]) -> dict:
    if not code:
        return {}
    node = {"dept_name": dept_code_name_dict[code], "dept_code": code, "children": []}
    children = structure.get(code, [])
    for child in children:
        node["children"].append(build_tree(structure, child, dept_code_name_dict))
    return node


def get_dept_info_list_ui(user_info: Sequence[UserInfo]):
    dept_info_list = []
    for user in user_info:
        params = {}
        for i in range(1, 10):
            level = f"l{i}"
            params[f"{level}_dept_code"] = getattr(user, f"{level}_dept_code", None)
            params[f"{level}_name"] = getattr(user, f"{level}_name", None)
        dept_info_list.append(DeptInfo(**params))
    return dept_info_list


def build_department_structure(dept_info_list: List[DeptInfo]) -> List[dict]:
    dept_code_name_dict = {}
    dept_code_list = []
    root_dept_code_list = []

    for dept_info in dept_info_list:
        code_list = []
        for i in range(1, 10):
            level_name = f"l{i}_name"
            level_code = f"l{i}_dept_code"
            name = getattr(dept_info, level_name, None)
            code = getattr(dept_info, level_code, None)
            if code and i == 1 and code not in root_dept_code_list:
                root_dept_code_list.append(code)
            if name and code:
                dept_code_name_dict[code] = name
                code_list.append(code)
            else:
                break
        if code_list:
            dept_code_list.append(code_list)

    structure = defaultdict(list)
    for arr in dept_code_list:
        for i in range(len(arr) - 1):
            parent = arr[i]
            child = arr[i + 1]
            if child not in structure[parent]:
                structure[parent].append(child)
    return [build_tree(structure, root_code, dept_code_name_dict) for root_code in root_dept_code_list]


def get_random_doc_from_knowledge_base(session: Session, req: RandomDocRequest):
    valid_knowledge_base_list = valid_batch_knowledge_base_member_read_permission(session, req.kb_sn_list, req.uid)
    vector_store_indexes = get_knowledge_bases_vector_names(
        session, [kb_and_member_info.knowledge_base.sn for kb_and_member_info in valid_knowledge_base_list]
    )
    query = es_query_k_documents(req.size)
    return get_vector_store_manager().search_documents_by_indices_query(vector_store_indexes, query)[: req.size]


def batch_validate_knowledge_base(session: Session, kb_sn_list: List[str], uid: str) -> Sequence[KnowledgeBase]:
    knowledge_base_and_member_types = valid_batch_knowledge_base_member_read_permission(session, kb_sn_list, uid)
    return [kb_and_member_info.knowledge_base for kb_and_member_info in knowledge_base_and_member_types]


@safe_trace()
# 验证用户是否有多个知识库的可读权限
def valid_batch_knowledge_base_member_read_permission(
    session: Session, kb_sn_list: List[str], operator: str
) -> List[KnowledgeBaseAndMemberTypeInfo]:
    kb_member_type_list = get_knowledge_base_and_member_type(session, kb_sn_list, operator)
    if not kb_member_type_list:
        raise KnowledgeBaseNotExistsException(f"知识库<{', '.join(kb_sn_list)}>不存在")
    read_permission_kb_list = []
    for knowledge_base, member_type in kb_member_type_list:
        try:
            validated_kb = validate_knowledge_base_member_read_permission(knowledge_base, member_type, operator)
            read_permission_kb_list.append(validated_kb)
        except KnowledgeBaseMemberNotExistsException:
            logger.error(f"用户 {operator} 无权读取知识库 {knowledge_base.sn}")
    return read_permission_kb_list


def validate_knowledge_base_member_read_permission(
    knowledge_base: KnowledgeBase, knowledge_base_member_type: MemberType, operator: str
) -> KnowledgeBaseAndMemberTypeInfo:
    # 分层知识库
    if knowledge_base.kb_type != KnowledgeBaseType.PRIVATE:
        # 部门知识库
        if knowledge_base.layer_kb_type == LayerKnowledgeBaseType.COMMON:
            common_dept_code = knowledge_base.common_dept_code
            common_kb_type = knowledge_base.common_kb_type
            return KnowledgeBaseAndMemberTypeInfo(
                knowledge_base=knowledge_base,
                member_type=get_layer_common_permission(
                    common_dept_code, knowledge_base_member_type, operator, common_kb_type
                ),
            )

        # 领域知识库
        if knowledge_base.layer_kb_type == LayerKnowledgeBaseType.DOMAIN:
            domain = knowledge_base.domain
            return KnowledgeBaseAndMemberTypeInfo(
                knowledge_base=knowledge_base,
                member_type=get_layer_domain_permission(domain, knowledge_base_member_type, operator),
            )
        # 产品知识库
        pbi_project_id = knowledge_base.pbi_project_id
        return KnowledgeBaseAndMemberTypeInfo(
            knowledge_base=knowledge_base,
            member_type=get_layer_product_permission(pbi_project_id, knowledge_base_member_type, operator),
        )
    # 个人知识库
    if not knowledge_base_member_type:
        raise KnowledgeBaseMemberNotExistsException(f"<{operator}>不是知识库<{knowledge_base.sn}>的成员")
    return KnowledgeBaseAndMemberTypeInfo(
        knowledge_base=knowledge_base,
        member_type=knowledge_base_member_type,
    )


# 获取通用知识库权限
def get_layer_common_permission(common_dept_code, knowledge_base_member_type, operator, common_kb_type):
    is_dept_manager = is_department_manager(common_dept_code, operator)
    if is_dept_manager:
        return MemberType.OWNER
    if knowledge_base_member_type:
        return knowledge_base_member_type
    # 是部门成员但是不是知识库成员
    is_dept_user = is_department_user(common_dept_code, operator)
    if is_dept_user:
        return MemberType.SIMPLE_MEMBER
    # 全部公开、接口公开返回普通成员
    if (
        common_kb_type == CommonKnowledgeBaseType.ALL_SHARED
        or common_kb_type == CommonKnowledgeBaseType.INTERFACE_SHARED
    ):
        return MemberType.SIMPLE_MEMBER
    raise KnowledgeBaseMemberNotExistsException(f"<{operator}>不是部门<{common_dept_code}>的成员")


# 获取领域知识库权限
def get_layer_domain_permission(domain: Domain, knowledge_base_member_type: MemberType, operator: str):
    is_manager = is_domain_manager(domain.value, operator)
    if is_manager:
        return MemberType.OWNER
    if knowledge_base_member_type:
        return knowledge_base_member_type
    return MemberType.SIMPLE_MEMBER


# 获取产品知识库权限
def get_layer_product_permission(pbi_project_id, knowledge_base_member_type: MemberType, operator):
    pbi_manager = is_pbi_manager(pbi_project_id, operator)
    if pbi_manager:
        return MemberType.OWNER
    if knowledge_base_member_type:
        return knowledge_base_member_type
    pbi_user = is_pbi_user(pbi_project_id, operator)
    if pbi_user:
        return MemberType.SIMPLE_MEMBER
    raise KnowledgeBaseMemberNotExistsException(f"<{operator}>不是PBI<{pbi_project_id}>的成员")


# 验证是否是owner
def validate_knowledge_base_member_owner_permission(session: Session, kb_sn, operator: str):
    permission_knowledge_base_and_member_type = valid_batch_knowledge_base_member_read_permission(
        session, [kb_sn], operator
    )[0]
    knowledge_base, knowledge_base_member_type = (
        permission_knowledge_base_and_member_type.knowledge_base,
        permission_knowledge_base_and_member_type.member_type,
    )
    if knowledge_base_member_type != MemberType.OWNER:
        raise OperationNotPermittedException(f"<{operator}>非知识库Owner，无权限对知识库<{kb_sn}>执行该操作")
    return knowledge_base


# 验证是否是可操作权限
@safe_trace()
def validate_knowledge_base_member_operate_permission(session: Session, kb_sn: str, operator: str) -> KnowledgeBase:
    permission_knowledge_base_and_member_type = valid_batch_knowledge_base_member_read_permission(
        session, [kb_sn], operator
    )[0]
    knowledge_base, knowledge_base_member_type = (
        permission_knowledge_base_and_member_type.knowledge_base,
        permission_knowledge_base_and_member_type.member_type,
    )
    if knowledge_base_member_type == MemberType.SIMPLE_MEMBER:
        raise OperationNotPermittedException(f"<{operator}>非知识库管理员，无权限对知识库<{kb_sn}>执行该操作")
    return knowledge_base


def get_read_or_operate_permission_dept_knowledge_base(
    uid: str,
    kbs_with_members: List[Tuple[KnowledgeBase, List[str], List[MemberType]]],
    permission: KnowledgeBasePermission,
):
    if permission == KnowledgeBasePermission.READ:
        return [kb[0] for kb in kbs_with_members]
    operate_permission_kbs = []
    for kb, members, member_types in kbs_with_members:
        knowledge_base_member_type = get_member_type_by_uid(uid, members, member_types)
        member_type = get_layer_common_permission(
            kb.common_dept_code, knowledge_base_member_type, uid, kb.common_kb_type
        )
        if member_type in MemberType.get_operate_member_type():
            operate_permission_kbs.append(kb)
    return operate_permission_kbs


def get_member_type_by_uid(uid: str, members: List[str], member_types: List[MemberType]):
    for member, member_type in zip(members, member_types):
        if uid.lower() == member.lower():
            return member_type
    return None


def get_read_or_operate_permission_domain_knowledge_base(
    uid: str,
    kbs_with_members: List[Tuple[KnowledgeBase, List[str], List[MemberType]]],
    permission: KnowledgeBasePermission,
):
    if permission == KnowledgeBasePermission.READ:
        return [kb[0] for kb in kbs_with_members]
    operate_permission_kbs = []
    for kb, members, member_types in kbs_with_members:
        knowledge_base_member_type = get_member_type_by_uid(uid, members, member_types)
        member_type = get_layer_domain_permission(kb.domain, knowledge_base_member_type, uid)
        if member_type in MemberType.get_operate_member_type():
            operate_permission_kbs.append(kb)
    return operate_permission_kbs


# 如果不是管理员获取指定分层知识库权限
@safe_trace()
def get_layer_knowledge_base_permission(member_knowledge_base, member):
    knowledge_base_members = member_knowledge_base.knowledge_base_members
    # 默认为SIMPLE_MEMBER角色
    public_knowledge_base_member = MemberType.SIMPLE_MEMBER
    # 判断指定部门知识库的权限
    for knowledge_base_member in knowledge_base_members:
        if member == knowledge_base_member.employee_number:
            public_knowledge_base_member = knowledge_base_member.member_type
    return public_knowledge_base_member


# 获取领域或者pbi知识库
def get_domain_or_pbi_knowledge_bases(knowledge_bases, member, domain, pbi_project_id, is_manager):
    layer_knowledge_bases = []
    for knowledge_base in knowledge_bases:
        member_type = MemberType.OWNER if is_manager else get_layer_knowledge_base_permission(knowledge_base, member)
        public_knowledge_base = PublicKnowledgeBaseShowInfo(
            name=knowledge_base.name,
            sn=knowledge_base.sn,
            owner=knowledge_base.owner,
            created_at=knowledge_base.created_at,
            updated_at=knowledge_base.updated_at,
            domain=domain,
            member_type=member_type,
            pbi_project_id=pbi_project_id,
            analyzer=knowledge_base.analyzer,
        )
        layer_knowledge_bases.append(public_knowledge_base)
    return layer_knowledge_bases


def get_read_or_operate_permission_knowledge_base(
    session: Session, uid: str, permission: KnowledgeBasePermission, keyword: str = None
):
    # 个人知识库类别
    personal_kbs = get_personal_kbs(session, uid, permission, keyword)
    # 部门公开知识库
    dept_all_shared_kbs = get_read_or_operate_permission_dept_knowledge_base(
        uid, get_all_dept_shared_kbs(session, keyword), permission
    )
    # 非计算用户直接返回个人知识库以及部门公开的知识库
    if not is_compute_user(uid):
        return personal_kbs + dept_all_shared_kbs
    dept_all_shared_kb_ids = [kb.id for kb in dept_all_shared_kbs]
    # 查询从下到上部门的知识库
    dept_shared_kbs = get_read_or_operate_permission_dept_knowledge_base(
        uid, get_layer_dept_shared_kbs(session, uid, keyword), permission
    )
    dept_shared_kb_ids = [kb.id for kb in dept_shared_kbs]
    # 查询领域知识库
    domain_kbs = get_read_or_operate_permission_domain_knowledge_base(uid, get_domain_kbs(session, keyword), permission)
    domain_kb_ids = [kb.id for kb in domain_kbs]
    # 查询用户不在devuc成员列表里面但在分层知识库中加了权限的知识库
    kb_ids = dept_all_shared_kb_ids + dept_shared_kb_ids + domain_kb_ids
    layer_permission_kbs = get_user_not_in_devuc_layer_permission_kbs(session, kb_ids, uid, permission, keyword)
    # 返回个人知识库-部门知识库部门公开-部门知识库全部公开-分层知识库中有权限的知识库-领域知识库，缺少个人相关的产品知识库
    return personal_kbs + dept_shared_kbs + dept_all_shared_kbs + layer_permission_kbs + domain_kbs


def add_department_manager(operator: str, dept_code: str, w3_account: str) -> bool:
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限添加部门管理员")
    result = portal_util.add_department_manager(dept_code, w3_account)
    if result:
        portal_util.update_department_manager_cache(dept_code, w3_account)
    return result


def get_department_manager(operator: str, session: Session) -> List[DeptManagerResponse]:
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限查看部门管理员")
    global_department_manager_list = portal_util.get_department_manager_cache()
    dept_codes = list(global_department_manager_list.keys())
    departments = get_department_code_and_names(session, dept_codes)
    dept_name_dict = {department[0]: department[1] for department in departments}
    department_details = []
    for dept_code, user_list in global_department_manager_list.items():
        department_details.append(
            DeptManagerResponse(dept_code=dept_code, dept_name=dept_name_dict[dept_code], w3_account_list=user_list)
        )
    return department_details


def get_domain_manager(operator: str) -> List[DomainManagerResponse]:
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限查看领域管理员")
    global_domain_manager_list = portal_util.get_domain_manager_cache()
    domain_details = []
    for domain, user_list in global_domain_manager_list.items():
        domain_details.append(DomainManagerResponse(domain=domain, w3_account_list=user_list))
    return domain_details


def add_domain_manager(operator: str, domain: str, w3_account: str) -> bool:
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限添加领域管理员")
    result = portal_util.add_domain_manager(domain, w3_account)
    if result:
        portal_util.update_domain_manager_cache(domain, w3_account)
    return result


def delete_department_manager(operator: str, dept_code: str, w3_account: str) -> bool:
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限删除部门管理员")
    if is_department_manager(dept_code, w3_account):
        portal_util.delete_department_manager_cache(dept_code, w3_account)
    return True


def delete_domain_manager(operator: str, domain: str, w3_account: str) -> bool:
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限删除领域管理员")
    if is_domain_manager(domain, w3_account):
        portal_util.delete_domain_manager_cache(domain, w3_account)
    return True


def is_super_root(operator: str) -> bool:
    return operator == SUPER_ROOT_NAME


# 新增知识库member
def create_knowledge_base_blacklist_member(req: CreateUpdateBlacklistMemberReq, session: Session):
    # 验证操作者权限
    knowledge_base = validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.operator)
    if validate_knowledge_base_blacklist_member_exist(req.employee_number, knowledge_base):
        raise DuplicateKnowledgeBaseMemberException(f"知识库黑名单中已存在成员 <{req.employee_number}> .")
    control_member_list = get_knowledge_base_control_members(knowledge_base)
    if req.employee_number.lower() in control_member_list:
        raise OperationNotPermittedException(f"<{req.employee_number}>是知识库{req.kb_sn}的管理员，无法加入黑名单！")
    add_knowledge_base_blacklist_member(session, knowledge_base, req.employee_number)
    session.commit()


@safe_trace()
def validate_knowledge_base_blacklist_member_exist(employee_number: str, knowledge_base: KnowledgeBase):
    for knowledge_base_blacklist_member in knowledge_base.knowledge_base_blacklist_members:
        if employee_number.lower() == knowledge_base_blacklist_member.employee_number.lower():
            return True
    return False


def get_knowledge_base_control_members(knowledge_base: KnowledgeBase):
    return [
        knowledge_base_member.employee_number.lower()
        for knowledge_base_member in knowledge_base.knowledge_base_members
        if knowledge_base_member.member_type != MemberType.SIMPLE_MEMBER
    ]


def add_knowledge_base_blacklist_member(session: Session, knowledge_base: KnowledgeBase, employee_number: str):
    employee_name = get_user_name(employee_number)
    result = member_information_search("zh", employee_number, str(0), str(20), str(1))
    department = result["members"][0]["dept"]
    new_blacklist_member = KnowledgeBaseBlacklistMember(
        employee_number=employee_number,
        employee_name=employee_name,
        department=department,
        kb_id=knowledge_base.id,
    )
    knowledge_base.knowledge_base_blacklist_members.append(new_blacklist_member)
    session.add(knowledge_base)


# 批量新增知识库member
def batch_create_knowledge_base_blacklist_member(req: CreateUpdateBlacklistMultipleMemberReq, session: Session) -> str:
    failed_member_set = set()
    succeed_member_set = set()
    exist_member_set = set()
    control_member_set = set()
    employee_number_set = set(req.employee_number_list)

    # 验证操作者权限
    knowledge_base = validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.operator)
    # 验证管理员和普通成员参数列表
    if not req.employee_number_list:
        raise InvalidParamError("成员列表为空")
    control_member_list = get_knowledge_base_control_members(knowledge_base)
    for employee_number in employee_number_set:
        if validate_knowledge_base_blacklist_member_exist(employee_number, knowledge_base):
            logger.warning("用户已在黑名单列表中", employee_number)
            exist_member_set.add(employee_number)
            continue
        if employee_number.lower() in control_member_list:
            control_member_set.add(employee_number)
            logger.warning("<%s>是知识库<%s>的管理员，无法加入黑名单！", employee_number, req.kb_sn)
            continue
        try:
            add_knowledge_base_blacklist_member(session, knowledge_base, employee_number)
            succeed_member_set.add(employee_number)
        except Exception:  # noqa
            logger.warning("添加普通用户%s失败，原因是用户不存在", employee_number)
            failed_member_set.add(employee_number)

    session.commit()
    message = ""
    if succeed_member_set:
        message += f"{len(succeed_member_set)}个用户添加成功, 用户名单如下：\n"
        message += str(succeed_member_set) + "\n"
    if failed_member_set:
        message += f"{len(failed_member_set)}个用户搜素不到，添加失败, 用户名单如下：\n"
        message += str(failed_member_set) + "\n"
    if exist_member_set:
        message += f"{len(exist_member_set)}个用户已在黑名单中, 用户名单如下：\n"
        message += str(exist_member_set) + "\n"
    if control_member_set:
        message += f"{len(control_member_set)}个用户是知识库管理员，无法添加, 用户名单如下：\n"
        message += str(control_member_set) + "\n"
    return message


def batch_delete_knowledge_base_blacklist_member(req: DeleteMemberReq, session: Session):
    # 验证操作者权限
    validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.operator)
    employee_members = [employee.lower() for employee in req.employee_numbers]
    # 子查询
    subq = session.query(KnowledgeBase).filter(KnowledgeBase.sn == req.kb_sn).subquery()
    session.execute(
        delete(KnowledgeBaseBlacklistMember).where(
            func.lower(KnowledgeBaseBlacklistMember.employee_number).in_(employee_members),
            KnowledgeBaseBlacklistMember.kb_id == subq.c.id,
        )
    )
    session.commit()


def get_knowledge_base_blacklist_member_list(kb_sn: str, uid: str, session: Session):
    valid_batch_knowledge_base_member_read_permission(session, [kb_sn], uid)
    query = (
        select(KnowledgeBaseBlacklistMember)
        .join(KnowledgeBase, KnowledgeBaseBlacklistMember.kb_id == KnowledgeBase.id, isouter=True)
        .where(KnowledgeBase.sn == kb_sn)
        .order_by(KnowledgeBaseBlacklistMember.employee_number, KnowledgeBaseBlacklistMember.employee_name)
    )
    return paginate(
        session,
        query,
        transformer=lambda members: [
            KnowledgeBaseBlacklistMemberShowInfo(
                employee_number=member.employee_number,
                employee_name=member.employee_name if member.employee_name else "",
                department=member.department,
            )
            for member in members
        ],
    )


def is_migrated_dept_user(session: Session, uid: str):
    _, migrated_dept_list_array = _get_migrated_dept_list_config_array(session)
    user_upper_dept_codes = get_upper_dept_codes(uid)
    # 判断用户部门是否在迁移部门列表中，如果在，说明该用户所在部门的知识库已完成迁移，该用户不能再访问libing-rag，需跳转至ipd-rag
    return set(user_upper_dept_codes) & set(migrated_dept_list_array)


def redirect(uid: str, session: Session):
    if uid == SUPER_ROOT_NAME:
        return False
    # 判断用户所在部门是否为已经完成知识库迁移的部门，如果是，则需要跳转
    if is_migrated_dept_user(session, uid):
        return True
    # 判断是否为计算产品线
    is_computing_product_line_user = is_compute_user(uid)
    if is_computing_product_line_user:
        return False

    # 查询 Redis 缓存白名单列表
    if validate_whitelist_user(session, uid):
        return False

    # 查询数据库知识库成员表判断是否为老用户
    return not validate_old_user(session, uid)


def validate_old_user(session: Session, uid: str):
    value = rc.get(old_user_keyspace.resolve(uid))
    if value:
        return value == IS_OLD_USER
    user_record = get_user_record_from_all_knowledge_base_members(session, uid)
    value = IS_OLD_USER if user_record else NOT_OLD_USER
    rc.set(
        old_user_keyspace.resolve(uid),
        value,
        px=24 * 60 * 60 * 1000,
    )
    return value == IS_OLD_USER


def validate_whitelist_user(session: Session, uid: str):
    value = rc.get(whitelist_user_keyspace.resolve(uid))
    if value:
        return value == IS_WHITELIST_USER
    user_record = get_user_record_from_whitelist_members(session, uid)
    value = IS_WHITELIST_USER if user_record else NOT_WHITELIST_USER
    rc.set(
        whitelist_user_keyspace.resolve(uid),
        value,
        px=24 * 60 * 60 * 1000,
    )
    return value == IS_WHITELIST_USER


def add_whitelist_member(session: Session, employee_number: str):
    employee_name = get_user_name(employee_number)
    result = member_information_search("zh", employee_number, str(0), str(20), str(1))
    department = result["members"][0]["dept"]
    whitelist_member = WhitelistMember(
        employee_number=employee_number,
        employee_name=employee_name,
        department=department,
    )
    session.add(whitelist_member)
    session.commit()
    rc.set(
        whitelist_user_keyspace.resolve(employee_number.lower()),
        IS_WHITELIST_USER,
        px=24 * 60 * 60 * 1000,
    )


def batch_create_whitelist_member(employee_number_list: List[str], session: Session) -> str:
    failed_member_set = set()
    succeed_member_set = set()
    exist_member_set = set()
    employee_number_set = set(employee_number_list)

    if not employee_number_list:
        raise InvalidParamError("成员列表为空")
    whitelist_members = session.exec(select(WhitelistMember)).all()
    whitelist_employee_number_list = [whitelist_member.employee_number for whitelist_member in whitelist_members]
    for employee_number in employee_number_set:
        if employee_number.lower() in whitelist_employee_number_list:
            logger.warning("用户已在白名单列表中", employee_number)
            exist_member_set.add(employee_number)
            continue
        try:
            add_whitelist_member(session, employee_number.lower())
            succeed_member_set.add(employee_number)
        except Exception:  # noqa
            logger.warning("添加用户%s失败，原因是用户不存在", employee_number)
            failed_member_set.add(employee_number)

    session.commit()
    message = ""
    if succeed_member_set:
        message += f"{len(succeed_member_set)}个用户添加成功, 用户名单如下：\n"
        message += str(succeed_member_set) + "\n"
    if failed_member_set:
        message += f"{len(failed_member_set)}个用户搜素不到，添加失败, 用户名单如下：\n"
        message += str(failed_member_set) + "\n"
    if exist_member_set:
        message += f"{len(exist_member_set)}个用户已在白名单中, 用户名单如下：\n"
        message += str(exist_member_set) + "\n"
    return message


def batch_delete_whitelist_member(employee_numbers: List[str], session: Session):
    employee_members = [employee.lower() for employee in employee_numbers]
    session.execute(delete(WhitelistMember).where(func.lower(WhitelistMember.employee_number).in_(employee_members)))
    session.commit()
    for employee_member in employee_members:
        rc.delete(whitelist_user_keyspace.resolve(employee_member.lower()))


def update_dept_kb_share_scope(session: Session, req: UpdateDepartmentKnowledgeBaseShareScope):
    is_dept_manager = is_department_manager(req.common_dept_code, req.operator)
    if not is_dept_manager:
        raise OperationNotPermittedException(f"<{req.operator}>非部门管理员，无权限对知识库<{req.kb_sn}>执行该操作")
    knowledge_base = session.exec(select(KnowledgeBase).where(KnowledgeBase.sn == req.kb_sn)).one_or_none()
    if (
        req.share_scope != CommonKnowledgeBaseType.DEPARTMENT_SHARED.value
        and req.share_scope != CommonKnowledgeBaseType.INTERFACE_SHARED.value
    ):
        raise InvalidParamError("输入的公开范围值错误，公开的范围只能是<部门公开>或<接口公开>。")
    if req.share_scope == CommonKnowledgeBaseType.INTERFACE_SHARED.value:
        knowledge_base.common_kb_type = CommonKnowledgeBaseType.INTERFACE_SHARED
    else:
        knowledge_base.common_kb_type = CommonKnowledgeBaseType.DEPARTMENT_SHARED
    session.add(knowledge_base)
    session.commit()


def migrate_knowledge_base_to_ipd_rag(session: Session, kb_sn: str):
    knowledge_base = get_knowledge_base_by_kb_sn(session, kb_sn)
    create_kb_and_asset_in_ipd_rag(session, knowledge_base.name, knowledge_base.sn, PUBLIC_ACCOUNT_NAME)
    for knowledge_base_asset in knowledge_base.knowledge_base_assets:
        vectorization_job = AutoJobInstances(status=JobStatus.PENDING, job_type=VectorizationJobType.MIGRATE_TO_IPD_RAG)
        knowledge_base_asset.auto_job_instances.append(vectorization_job)
        session.add(knowledge_base_asset)
    knowledge_base.is_migrated = True
    session.add(knowledge_base)
    session.commit()


def get_kb_list_prompt(kb_sn_list: Optional[List[str]], member, question, session) -> str:
    if not kb_sn_list:
        return get_prompt().replace("{{ question }}", question).strip()
    knowledge_base = valid_batch_knowledge_base_member_read_permission(session, [kb_sn_list[0]], member)[
        0
    ].knowledge_base
    kb_config = KnowledgeBaseConfig.model_validate(knowledge_base.config)
    if kb_config and kb_config.prompt:
        result = kb_config.prompt
    else:
        result = (
            get_prompt(default_prompt=SYNONYM_PROMPT, prompt_id=SYNONYM_PROMPT_ID)
            if knowledge_base.analyzer == Analyzer.IK_ANALYZER
            else get_prompt()
        )
    synonyms = get_question_synonyms(session, question, [kb_sn_list[0]])
    if synonyms:
        synonym_prompt = (
            "\n" + "\n".join("<同义词>" + synonym + "</同义词>" for synonym in synonyms[:MAX_SYNONYM_SIZE]) + "\n"
        )
        return result.replace("{{ question }}", question).replace("{{ synonym }}", synonym_prompt).strip()
    return result.replace("{{ question }}", question).replace("\n\n<同义词列表>{{ synonym }}</同义词列表>", "").strip()


def cite_reference_answer(referenceAnswerReq: ReferenceAnswerReq) -> ReferenceAnswerResp:
    result = ReferenceAnswerResp(answer=referenceAnswerReq.answer)
    if referenceAnswerReq.citation and referenceAnswerReq.fetch_format in StreamAnswerFormat.get_markdown_formatted():
        citations = [document.text for document in referenceAnswerReq.documents]
        result.answer = get_answer_with_citations(referenceAnswerReq.answer, citations)
    if not referenceAnswerReq.only_llm_answer:
        result.reference = get_quote_reference(referenceAnswerReq.documents)
    return result


def open_knowledge_base_permissions_to_department(req: OpenKbPermissionsReq, session: Session):
    if req.member_type == MemberType.OWNER:
        raise KnowledgeBaseMemberValidationException("新增知识库的成员类型不能为owner")
    # 验证操作者权限
    knowledge_base = validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.operator)
    if (
        knowledge_base.kb_type != KnowledgeBaseType.LAYER
        or knowledge_base.layer_kb_type != LayerKnowledgeBaseType.COMMON
        or knowledge_base.common_kb_type != CommonKnowledgeBaseType.DEPARTMENT_SHARED
    ):
        raise NotDepartmentSharedLayerKnowledgeBaseException("只有部门公开的部门知识库才能选择部门开放权限")
    # 查询部门下的所有员工
    employee_list = get_dept_employee_list(req.dept_code)
    if not employee_list:
        raise DeptNotExistException(f"部门<{req.dept_name}>不存在")
    employee_ids = [employee["w3account"].lower() for employee in employee_list]
    knowledge_base_add_dept_member(session, req.member_type, employee_ids, knowledge_base)
    session.commit()


def knowledge_base_add_dept_member(
    session: Session, member_type: MemberType, employee_ids: [str], knowledge_base: KnowledgeBase
):
    knowledge_base_members = get_knowledge_base_members(session, knowledge_base.sn)
    knowledge_base_members_ids = [member.employee_number.lower() for member in knowledge_base_members]
    old_employee_ids = []
    new_employee_ids = []
    for employee_id in employee_ids:
        if employee_id in knowledge_base_members_ids:
            old_employee_ids.append(employee_id)
        else:
            new_employee_ids.append(employee_id)
    for new_employee_id in new_employee_ids:
        new_knowledge_base_member = KnowledgeBaseMember(
            employee_number=new_employee_id, member_type=member_type, kb_id=knowledge_base.id
        )
        knowledge_base.knowledge_base_members.append(new_knowledge_base_member)
    if old_employee_ids:
        user_record_list = get_user_record_list_from_knowledge_base_member(session, knowledge_base.sn, old_employee_ids)
        for user_record in user_record_list:
            if user_record.member_type != MemberType.OWNER:
                user_record.member_type = member_type
    session.add(knowledge_base)
    if new_employee_ids:
        thread_exec_set_old_user_flag_to_redis(new_employee_ids)


def thread_exec_set_old_user_flag_to_redis(new_employee_ids: [str]):
    with contextlib.suppress(Exception):
        CTXThread(target=set_old_user_flag_to_redis, args=[new_employee_ids]).start()


def set_old_user_flag_to_redis(new_employee_ids: [str]):
    for new_employee_id in new_employee_ids:
        rc.set(
            old_user_keyspace.resolve(new_employee_id),
            IS_OLD_USER,
            px=24 * 60 * 60 * 1000,
        )


def knowledge_base_delete_dept_member(session, employee_ids, knowledge_base):
    subq = session.query(KnowledgeBase).filter(KnowledgeBase.sn == knowledge_base.sn).subquery()
    session.execute(
        delete(KnowledgeBaseMember).where(
            func.lower(KnowledgeBaseMember.employee_number).in_(employee_ids),
            KnowledgeBaseMember.kb_id == subq.c.id,
            KnowledgeBaseMember.member_type != MemberType.OWNER,
        )
    )
    session.commit()
    old_user_flag_redis_keys = [old_user_keyspace.resolve(employee_id) for employee_id in employee_ids]
    rc.delete(*old_user_flag_redis_keys)


def cancel_knowledge_base_permissions_to_department(req: CancelKbPermissionsReq, session: Session):
    # 验证操作者权限
    knowledge_base = validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.operator)
    if (
        knowledge_base.kb_type != KnowledgeBaseType.LAYER
        or knowledge_base.layer_kb_type != LayerKnowledgeBaseType.COMMON
        or knowledge_base.common_kb_type != CommonKnowledgeBaseType.DEPARTMENT_SHARED
    ):
        raise NotDepartmentSharedLayerKnowledgeBaseException("只有部门公开的部门知识库才能选择部门取消权限")
    # 查询部门下的所有员工
    employee_list = get_dept_employee_list(req.dept_code)
    if not employee_list:
        raise DeptNotExistException(f"部门<{req.dept_name}>不存在")
    employee_ids = [employee["w3account"].lower() for employee in employee_list]
    knowledge_base_delete_dept_member(session, employee_ids, knowledge_base)


def get_kb_list_by_sn_list(kb_sn_list: List[str], session: Session):
    if not kb_sn_list:
        return {}
    kb_list = session.exec(select(KnowledgeBase).where(KnowledgeBase.sn.in_(kb_sn_list))).all()
    return {kb.sn: kb.name for kb in kb_list}


def set_default_knowledge_base(req: OperateDefaultKnowledgeBaseReq, session: Session):
    # 验证操作者权限
    knowledge_base = validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.operator)
    if (
        knowledge_base.kb_type != KnowledgeBaseType.LAYER
        or knowledge_base.layer_kb_type != LayerKnowledgeBaseType.COMMON
    ):
        raise NotDepartmentSharedLayerKnowledgeBaseException("只有部门知识库才能被设置为默认知识库")
    is_dept_manager = is_department_manager(knowledge_base.common_dept_code, req.operator)
    if not is_dept_manager:
        raise OperationNotPermittedError("只有部门管理员才有权限操作默认知识库")
    if req.is_default:
        knowledge_base.is_default = True
        if knowledge_base.default_status is None:
            knowledge_base.default_status = DefaultKbStatus.ONLINE
        if get_need_add_schedule_knowledge_base_list(session, knowledge_base.sn):
            check_default_knowledge_base_job_schedule = AutomatedJobSchedule(
                updated_cycle=timedelta(seconds=24 * 3600),
                next_updated_at=get_next_cycle_time_from_current(get_current_passed_seconds(), 0),
                last_updated_at=now_with_time_zone(),
                extra_info={"kb_sn": knowledge_base.sn},
            )
            session.add(check_default_knowledge_base_job_schedule)
        session.add(knowledge_base)
        session.commit()
        return f"已设置知识库 <{req.kb_sn}>为默认知识库"
    knowledge_base.is_default = False
    if knowledge_base.default_status == DefaultKbStatus.ONLINE and not get_default_kb_running_check_job(
        session, knowledge_base.sn
    ):
        knowledge_base.default_status = None
        delete_online_default_kb_schedule(session, knowledge_base)
    session.add(knowledge_base)
    session.commit()
    return f"已取消设置知识库 <{req.kb_sn}>为默认知识库"


def get_default_dept_knowledge_base_sn_list(session: Session, uid: str) -> Optional[list[str]]:
    upper_dept_codes = get_upper_dept_codes(uid)
    default_kb_list = get_default_kb_list_by_dept_list(session, upper_dept_codes, True)
    if not default_kb_list:
        return None

    dept_code_sn_dict = defaultdict(list)
    for kb in default_kb_list:
        dept_code_sn_dict[kb.common_dept_code].append(kb.sn)

    result = []
    for dept_code in reversed(upper_dept_codes):
        if dept_code in dept_code_sn_dict:
            result.extend(dept_code_sn_dict[dept_code])
    return result


def extract_default_dept_knowledge_base_list(
    default_dept_knowledge_base_sn_list: list[str],
    dept_shared_kbs: [KnowledgeBase],
    dept_all_shared_kbs: [KnowledgeBase],
):
    default_dept_knowledge_base_list = []
    dept_shared_knowledge_base_sn_dict = {kb.sn: kb for kb in dept_shared_kbs}
    dept_all_shared_knowledge_base_sn_dict = {kb.sn: kb for kb in dept_all_shared_kbs}
    for sn in default_dept_knowledge_base_sn_list:
        if sn in dept_shared_knowledge_base_sn_dict:
            default_dept_knowledge_base_list.append(dept_shared_knowledge_base_sn_dict[sn])
            dept_shared_kbs.remove(dept_shared_knowledge_base_sn_dict[sn])
        if sn in dept_all_shared_knowledge_base_sn_dict:
            default_dept_knowledge_base_list.append(dept_all_shared_knowledge_base_sn_dict[sn])
            dept_all_shared_kbs.remove(dept_all_shared_knowledge_base_sn_dict[sn])
    return default_dept_knowledge_base_list


def get_all_read_or_operate_permission_knowledge_base(
    session: Session, uid: str, permission: KnowledgeBasePermission, keyword: str = None
):
    # 个人知识库类别
    personal_kbs = get_personal_kbs(session, uid, permission, keyword)
    compute_user_dept_code_list = get_upper_dept_codes(uid) if is_compute_user(uid) else []
    # 全部公开的部门知识库
    dept_all_shared_kbs = get_read_or_operate_permission_dept_knowledge_base(
        uid, get_all_dept_shared_kbs(session, keyword, compute_user_dept_code_list), permission
    )
    result = {
        "personal_kbs": get_knowledge_bases_info(personal_kbs),
        "public_kbs": get_knowledge_bases_info(dept_all_shared_kbs),
    }
    # 非计算用户直接返回个人知识库以及部门公开的知识库
    if not is_compute_user(uid):
        return result
    dept_all_shared_kb_ids = [kb.id for kb in dept_all_shared_kbs]
    # 查询从下到上部门的知识库
    dept_shared_kbs = get_read_or_operate_permission_dept_knowledge_base(
        uid, get_layer_dept_shared_kbs(session, uid, keyword, True, True), permission
    )
    dept_shared_kb_ids = [kb.id for kb in dept_shared_kbs]
    # 查询领域知识库
    domain_kbs = get_read_or_operate_permission_domain_knowledge_base(uid, get_domain_kbs(session, keyword), permission)
    domain_kb_ids = [kb.id for kb in domain_kbs]
    # 查询用户不在devuc成员列表里面但在分层知识库中加了权限的知识库
    kb_ids = dept_all_shared_kb_ids + dept_shared_kb_ids + domain_kb_ids
    layer_permission_kbs = get_user_not_in_devuc_layer_permission_kbs(session, kb_ids, uid, permission, keyword)
    default_dept_knowledge_base_sn_list = get_default_dept_knowledge_base_sn_list(session, uid)
    if default_dept_knowledge_base_sn_list:
        default_dept_knowledge_base_list = extract_default_dept_knowledge_base_list(
            default_dept_knowledge_base_sn_list, dept_shared_kbs, dept_all_shared_kbs
        )
        result["default_kbs"] = get_knowledge_bases_info(default_dept_knowledge_base_list)
    else:
        result["default_kbs"] = []
    result["dept_kbs"] = get_knowledge_bases_info(dept_shared_kbs + layer_permission_kbs)
    result["public_kbs"] = get_knowledge_bases_info(dept_all_shared_kbs)
    result["domain_kbs"] = get_domain_kbs_group_by_domain(domain_kbs)
    return result


def get_domain_kbs_group_by_domain(domain_kbs: list[KnowledgeBase]):
    development_domain_kbs = []
    test_domain_kbs = []
    build_domain_kbs = []
    design_domain_kbs = []
    maintenance_domain_kbs = []
    security_domain_kbs = []
    for domain_kb in domain_kbs:
        if domain_kb.domain == Domain.DEVELOPMENT:
            development_domain_kbs.append(domain_kb)
        elif domain_kb.domain == Domain.TEST:
            test_domain_kbs.append(domain_kb)
        elif domain_kb.domain == Domain.BUILD:
            build_domain_kbs.append(domain_kb)
        elif domain_kb.domain == Domain.DESIGN:
            design_domain_kbs.append(domain_kb)
        elif domain_kb.domain == Domain.MAINTENANCE:
            maintenance_domain_kbs.append(domain_kb)
        elif domain_kb.domain == Domain.SECURITY:
            security_domain_kbs.append(domain_kb)
    return {
        "development": get_knowledge_bases_info(development_domain_kbs),
        "test": get_knowledge_bases_info(test_domain_kbs),
        "build": get_knowledge_bases_info(build_domain_kbs),
        "design": get_knowledge_bases_info(design_domain_kbs),
        "maintenance": get_knowledge_bases_info(maintenance_domain_kbs),
        "security": get_knowledge_bases_info(security_domain_kbs),
    }


def quit_knowledge_base(req: QuitKbMemberReq, session: Session):
    validate_quit_knowledge_base_permission(session, req.kb_sn, req.operator)
    operator = req.operator.lower()
    subq = session.query(KnowledgeBase).filter(KnowledgeBase.sn == req.kb_sn).subquery()
    session.execute(
        delete(KnowledgeBaseMember).where(
            func.lower(KnowledgeBaseMember.employee_number) == operator,
            KnowledgeBaseMember.kb_id == subq.c.id,
            KnowledgeBaseMember.member_type != MemberType.OWNER,
        )
    )
    session.commit()
    rc.delete(old_user_keyspace.resolve(operator))


def validate_quit_knowledge_base_permission(session: Session, kb_sn, operator: str):
    kb_member_type_list = get_knowledge_base_and_member_type(session, [kb_sn], operator)
    if not kb_member_type_list:
        raise KnowledgeBaseNotExistsException(f"知识库<{kb_sn}>不存在")
    knowledge_base, knowledge_base_member_type = kb_member_type_list[0]
    if knowledge_base_member_type == MemberType.OWNER:
        raise OperationNotPermittedException(f"<{operator}>是知识库Owner，无法退出知识库<{kb_sn}>")
    if not is_user_in_knowledge_base_member_list(operator, knowledge_base):
        raise OperationNotPermittedException(f"<{operator}>非知识库成员，无法退出知识库<{kb_sn}>")
    return knowledge_base


def is_user_in_knowledge_base_member_list(employee_number: str, knowledge_base: KnowledgeBase):
    for knowledge_base_member in knowledge_base.knowledge_base_members:
        if employee_number.lower() == knowledge_base_member.employee_number.lower():
            return True
    return False


def check_default_knowledge_base_rules(kb_sn: str, session: Session):
    if check_knowledge_base_unresolved_dislike_number_exceed(kb_sn, session):
        return False
    if check_knowledge_base_eval_dataset_not_exist(kb_sn, session):
        return False
    if check_knowledge_base_eval_score_below(kb_sn, session):
        return False
    if check_knowledge_base_dislike_rate_exceed(kb_sn, session):
        return False
    return True


def check_knowledge_base_unresolved_dislike_number_exceed(kb_sn: str, session: Session):
    kb_sn_alias = "kb_sn_in_list"
    query = (
        select(func.count(RequestResponseLog.request_id))
        .outerjoin(
            func.jsonb_array_elements_text(
                func.jsonb_extract_path(cast(RequestResponseLog.extra_info, JSONB), "kb_sn_list")
            ).alias(kb_sn_alias),
            true(),
        )
        .where(
            RequestResponseLog.process_status.is_(False),
            between(RequestResponseLog.dislike_tag, 100, 200),
            or_(
                RequestResponseLog.kb_sn == kb_sn,
                text(f"{kb_sn_alias} = '{kb_sn}'"),
            ),
        )
    )
    return session.execute(query).scalar() > 3


def check_knowledge_base_eval_dataset_not_exist(kb_sn: str, session: Session):
    query = (
        select(func.count(EvalDataset.id))
        .join(KnowledgeBase, KnowledgeBase.id == EvalDataset.kb_id)
        .where(
            KnowledgeBase.sn == kb_sn,
        )
    )
    return session.execute(query).scalar() == 0


def check_knowledge_base_eval_score_below(kb_sn: str, session: Session):
    query = (
        select(EvalDatasetComprehensiveMetric.score)
        .join(EvalDatasetJob, EvalDatasetJob.id == EvalDatasetComprehensiveMetric.eval_dataset_job_id)
        .join(KnowledgeBase, KnowledgeBase.id == EvalDatasetJob.kb_id)
        .where(
            KnowledgeBase.sn == kb_sn,
            EvalDatasetComprehensiveMetric.score * 100 >= 60,
            datetime.now(tz=timezone(timedelta(hours=8))) - EvalDatasetComprehensiveMetric.update_at
            <= timedelta(days=30),
        )
        .order_by(desc(EvalDatasetComprehensiveMetric.update_at))
    )
    return session.execute(query).one_or_none() is None


def check_knowledge_base_dislike_rate_exceed(kb_sn: str, session: Session):
    kb_sn_alias = "kb_sn_in_list"
    condition = RequestResponseLog.acceptance.is_(False)
    query = (
        select(func.round(func.count().filter(condition) * 100.0 / func.count(), 2).label("percentage"))
        .select_from(RequestResponseLog)
        .outerjoin(
            func.jsonb_array_elements_text(
                func.jsonb_extract_path(cast(RequestResponseLog.extra_info, JSONB), "kb_sn_list")
            ).alias(kb_sn_alias),
            true(),
        )
        .where(
            datetime.now(tz=timezone(timedelta(hours=8))) - RequestResponseLog.request_end_time <= timedelta(days=30),
            or_(
                RequestResponseLog.kb_sn == kb_sn,
                text(f"{kb_sn_alias} = '{kb_sn}'"),
            ),
        )
    )
    result = session.execute(query).fetchone()
    return result.percentage >= 5


def delete_online_default_kb_schedule(session: Session, knowledge_base: KnowledgeBase):
    if knowledge_base.default_status is None:
        session.execute(
            delete(AutomatedJobSchedule).where(
                cast(AutomatedJobSchedule.extra_info.op("->>")("kb_sn"), String) == knowledge_base.sn,
            )
        )
        session.commit()


def get_need_add_schedule_knowledge_base_list(session: Session, kb_sn: str = None):
    subquery = (
        select(cast(AutomatedJobSchedule.extra_info.op("->>")("kb_sn"), String))
        .select_from(AutomatedJobSchedule)
        .where(AutomatedJobSchedule.extra_info.op("->>")("kb_sn").isnot(None))
    )
    query = select(KnowledgeBase).where(
        KnowledgeBase.sn.not_in(subquery),
        KnowledgeBase.default_status.isnot(None),
    )
    if kb_sn:
        query = query.where(KnowledgeBase.sn == kb_sn)
    return session.execute(query).all()


def get_default_kb_running_check_job(session: Session, kb_sn: str):
    return session.execute(
        select(AutoJobInstances).where(
            cast(AutoJobInstances.extra_info.op("->>")("kb_sn"), String) == kb_sn,
            AutoJobInstances.job_type == VectorizationJobType.CHECK_DEFAULT_KB_RULES,
            AutoJobInstances.status.in_(JobStatus.types_running()),
        )
    ).one_or_none()


def get_knowledge_base_list_info(
    session: Session, knowledge_base_owner_request: KnowledgeBaseOwnerRequest
) -> Dict[str, str]:
    kb_sn_set = set(knowledge_base_owner_request.kb_sn_list)
    knowledge_base_owner_tuples = get_knowledge_base_owner(session, kb_sn_set)
    knowledge_base_owner_dict_from_db = {kb_sn: owner for kb_sn, owner in knowledge_base_owner_tuples}
    return {kb_sn: knowledge_base_owner_dict_from_db.get(kb_sn) for kb_sn in kb_sn_set}


def get_knowledge_base_owner_dept_info(session: Session) -> List[dict]:
    query = (
        select(UserInfo)
        .select_from(KnowledgeBase)
        .outerjoin(UserInfo, UserInfo.user_id == func.lower(KnowledgeBase.owner))
    )
    user_info = session.exec(query).all()
    return build_department_structure(get_dept_info_list_ui(user_info))


def update_ipd_rag_whitelist(dept_code: str, state: bool, session: Session):
    """
    该方法主要是用于更新ipd_rag的白名单，当部门在白名单中时表示开启了知识库同步功能，当启用同步时，该部门成员在创建/删除知识库时会触发ipd_rag
    的同步操作；当禁用同步（即部门不在白名单中）时，相关操作将不会在ipd_rag中进行同步。
    :param dept_code: 新增或删除的部门号
    :param state: state为true表示允许同步操作（即在允许同步操作的部门号列表中新增部门号），state为false表示不允许同步操作（即在允许同步
    操作的部门号列表中删除部门号）
    :param session: 数据库连接会话
    """
    ipd_rag_whitelist_config = session.exec(
        select(ServiceConfig).where(ServiceConfig.name == "ipd_rag_whitelist")
    ).one_or_none()
    ipd_rag_whitelist_array: List[str] = []
    if ipd_rag_whitelist_config and ipd_rag_whitelist_config.value:
        ipd_rag_whitelist_array = ipd_rag_whitelist_config.value.split(",")
    if state and dept_code not in ipd_rag_whitelist_array:
        ipd_rag_whitelist_array.append(dept_code)
        ipd_rag_whitelist_array_str = ",".join(ipd_rag_whitelist_array)
        ipd_rag_whitelist_config.value = ipd_rag_whitelist_array_str
        session.commit()
        session.add(ipd_rag_whitelist_config)
        return "部门添加成功"
    if not state and dept_code in ipd_rag_whitelist_array:
        ipd_rag_whitelist_array.remove(dept_code)
        ipd_rag_whitelist_array_str = ",".join(ipd_rag_whitelist_array)
        ipd_rag_whitelist_config.value = ipd_rag_whitelist_array_str
        session.commit()
        session.add(ipd_rag_whitelist_config)
        return "部门删除成功"
    return "部门已存在或已删除"


def get_ipd_rag_whitelist(session: Session):
    ipd_rag_whitelist = session.exec(
        select(ServiceConfig.value).where(ServiceConfig.name == "ipd_rag_whitelist")
    ).one_or_none()
    ipd_rag_whitelist_array: List[str] = []
    if ipd_rag_whitelist:
        ipd_rag_whitelist_array = ipd_rag_whitelist.split(",")
    return ipd_rag_whitelist_array


def validate_dept_in_ipd_rag_whitelist(session: Session, dept_code: str):
    ipd_rag_whitelist = session.exec(
        select(ServiceConfig.value).where(ServiceConfig.name == "ipd_rag_whitelist")
    ).one_or_none()
    ipd_rag_whitelist_array: List[str] = []
    if ipd_rag_whitelist:
        ipd_rag_whitelist_array = ipd_rag_whitelist.split(",")
    return dept_code in ipd_rag_whitelist_array


def control_kb_synchronous_update_switch(session: Session, kb_sn: str, state: bool):
    knowledge_base = get_knowledge_base_by_kb_sn(session, kb_sn)
    knowledge_base.allow_synchronous_update = state
    session.add(knowledge_base)
    session.commit()


def get_all_knowledge_bases_migrate_info(session: Session, req: GetKbMigrateInfoReq) -> Page[KnowledgeBaseMigrateInfo]:
    dept_name_fields = [getattr(UserInfo, f"l{i}_name") for i in range(9, 0, -1)]
    query = (
        select(KnowledgeBase, func.coalesce(*dept_name_fields).label("dept_name"))
        .select_from(KnowledgeBase)
        .outerjoin(UserInfo, UserInfo.user_id == func.lower(KnowledgeBase.owner))
        .order_by(KnowledgeBase.created_at)
    )
    if req.kb_sn:
        query = query.where(KnowledgeBase.sn == req.kb_sn)
    if req.dept_code:
        columns = UserInfo.__table__.columns
        dept_code_columns = [col for col in columns if col.name.endswith("_dept_code")]
        conditions = [col == req.dept_code for col in dept_code_columns]
        query = query.where(or_(*conditions))
    if req.kb_type:
        query = query.where(KnowledgeBase.kb_type == req.kb_type)
    return paginate(
        session,
        query,
        params=Params(page=req.page, size=req.size),
        transformer=lambda items: [
            KnowledgeBaseMigrateInfo(
                kb_sn=knowledge_base.sn,
                owner=knowledge_base.owner,
                dept_name=dept_name if dept_name else None,
                is_migrated=knowledge_base.is_migrated,
                allow_synchronous_update=knowledge_base.allow_synchronous_update,
            )
            for knowledge_base, dept_name in items
        ],
    )


def get_top_score_rerank_search_data(req: RerankSearchDataReq):
    question_and_data_pairs = []
    search_datas = json.loads(req.search_datas)
    for search_data in search_datas:
        high_light_list = search_data["HIGHLIGHT"]
        high_light = "\n".join(high_light_list)
        question_and_data_pairs.append((req.question, high_light))
    try:
        start = time.time()
        score_resp = rerank_embedding(question_and_data_pairs, req.rerank_model)
        end = time.time()
        logger.info("%s sentence pairs rerank 耗时: %s秒...", len(question_and_data_pairs), end - start)
    except Exception as e:
        logger.exception(f"Rerank error is {e}")
        return search_datas
    filter_datas = []
    for score, search_data in sorted(zip(score_resp, search_datas), reverse=True, key=lambda x: x[0]):
        if score > req.score_threshold and len(filter_datas) < req.need_data_num:
            filter_datas.append(search_data)
    return filter_datas


def get_domain_kb_sn_list(domain: Domain, session: Session):
    kb_list = session.execute(select(KnowledgeBase).where(KnowledgeBase.domain == domain)).all()
    return {"kb_sn_list": [kb.KnowledgeBase.sn for kb in kb_list]}


def _get_migrated_dept_list_config_array(session: Session):
    migrated_dept_list_config = session.exec(
        select(ServiceConfig).where(ServiceConfig.name == "migrated_dept_list")
    ).one_or_none()
    if not migrated_dept_list_config:
        migrated_dept_list_config = ServiceConfig(name="migrated_dept_list", value="")
        session.add(migrated_dept_list_config)
        session.commit()
    migrated_dept_list_array = []
    if migrated_dept_list_config.value:
        migrated_dept_list_array = migrated_dept_list_config.value.split(",")
    return migrated_dept_list_config, migrated_dept_list_array


def _update_migrated_dept_list(
    session: Session, migrated_dept_list_config: ServiceConfig, migrated_dept_list_array: List[str]
):
    migrated_dept_list_config.value = ",".join(migrated_dept_list_array)
    session.add(migrated_dept_list_config)
    session.commit()


def add_migrated_dept(dept_code: str, session: Session):
    migrated_dept_list_config, migrated_dept_list_array = _get_migrated_dept_list_config_array(session)
    if dept_code in migrated_dept_list_array:
        return "部门已在迁移列表中"
    migrated_dept_list_array.append(dept_code)
    _update_migrated_dept_list(session, migrated_dept_list_config, migrated_dept_list_array)
    return "部门添加成功"


def delete_migrated_dept(dept_code: str, session: Session):
    migrated_dept_list_config, migrated_dept_list_array = _get_migrated_dept_list_config_array(session)
    if dept_code not in migrated_dept_list_array:
        return "部门不在迁移列表中"
    migrated_dept_list_array.remove(dept_code)
    _update_migrated_dept_list(session, migrated_dept_list_config, migrated_dept_list_array)
    return "部门删除成功"


def get_migrated_dept_list(session: Session):
    _, migrated_dept_list_array = _get_migrated_dept_list_config_array(session)
    return migrated_dept_list_array


def get_whitelist_list(req: WhitelistQueryReq, session: Session) -> Page[WhitelistResp]:
    if not is_super_root(req.operator):
        raise OperationNotPermittedException(f"<{req.operator}>非RAG后台管理员，无权限查询白名单")
    query = (
        select(ComputeWhitelist).order_by(ComputeWhitelist.id)
    )
    if req.query_account:
        query = query.where(ComputeWhitelist.w3Account.ilike(f"%{req.query_account}%"))
    return paginate(
        session,
        query,
        params=Params(page=req.page, size=req.size),
        transformer=lambda items: [
            WhitelistResp(
                w3_account=item.w3Account,
                user_name=get_user_name_without_exception(item.w3Account),
            )
            for item in items
        ],
    )


def _validate_account(w3_account_list):
    for account in w3_account_list:
        try:
            get_user_name(account)
        except Exception as e:
            logger.warning("%s ：账号不存在", account)
            raise InvalidParamException(f"<{account}>账号不存在") from e


def add_whitelist_list(req: WhitelistReq, session: Session):
    if not req.w3_account_list:
        raise InvalidParamException("白名单账号为空")
    if not is_super_root(req.operator):
        raise OperationNotPermittedException(f"<{req.operator}>非RAG后台管理员，无权限添加白名单")
    _validate_account(req.w3_account_list)
    exist_whitelist_account_list = get_whitelist_list_by_account_list(session, req.w3_account_list)
    new_whitelist_list = [ComputeWhitelist(w3Account=whitelist)
                          for whitelist in set(req.w3_account_list) - set(exist_whitelist_account_list)]
    if not new_whitelist_list:
        return
    session.add_all(new_whitelist_list)
    session.commit()


def delete_whitelist_list(req: WhitelistReq, session: Session):
    if not req.w3_account_list:
        raise InvalidParamException("白名单账号为空")
    if not is_super_root(req.operator):
        raise OperationNotPermittedException(f"<{req.operator}>非RAG后台管理员，无权限删除白名单")
    delete_whitelist_list_by_account_list(session, req.w3_account_list)
    session.commit()


def validate_user_operation_permission(session: Session, uid: str):
    if is_migrated_dept_user(session, uid):
        raise OperationNotPermittedException(f"已停止对用户<{uid}>所在的部门提供服务，无法执行此操作")


def _start_change_owner(req: ChangeOwnerReq, session, knowledge):
    if req.new_owner == knowledge.owner:
        raise InvalidParamException(f'<{req.new_owner}>已经是知识库的OWNER')
    # 更新角色表中的owner信息
    req.old_owner = knowledge.owner
    req.kb_id = knowledge.id
    req.new_owner_name = get_user_name(req.new_owner)
    if req.keep_in_kb and not req.member_type:
        raise InvalidParamException(f"用户权限类型必填")
    update_kn_base_owner(session, req)

    # 更新知识库表的owner
    update_owner(session, knowledge.sn, req.new_owner)
    session.commit()


def change_owner(req: ChangeOwnerReq, session: Session):
    if is_super_root(req.operator):
        knowledge = validate_knowledge_base(session, req.kb_sn)
    else:
        knowledge = get_knowledge_base_detail(session, req.kb_sn, req.operator)
        member_type = knowledge.member_type
        if member_type != MemberType.OWNER:
            raise OperationNotPermittedException(f"<{req.operator}>非知识库OWNER，无权限转移知识库")
    if not knowledge:
        raise KnowledgeBaseNotExistsError(f"没有知识库匹配<{req.kb_sn}>")
    _start_change_owner(req, session, knowledge)


def get_list_by_owner(session, owner, operator):
    """
    超级管理员通过owner获取列表
    :param session:
    :param owner: 选择的owner用户
    :param operator: 操作用户
    :return: 返回owner对应的知识库列表
    """
    if not is_super_root(operator):
        return []
    kb_list = query_by_owner(session, owner)
    return [{"sn": item.sn, "name": item.name} for item in kb_list]


def get_kb_count_can_create(dept_or_domain_code: str, config_type: ServiceConfigType, session: Session) -> int:
    """
    查询部门或者领域支持创建的知识库的最大数量
    @param dept_or_domain_code: 部门编码或者领域编码
    @param config_type: 部门或者领域
    @param session: 数据库会话
    @return: 可创建的知识库的最大数量
    """
    kb_count = query_by_config_type_and_name(session, dept_or_domain_code, config_type)
    if not kb_count:
        return MAX_LAYER_KNOWLEDGE_BASE_NUMBER
    return max(int(kb_count.value), MAX_LAYER_KNOWLEDGE_BASE_NUMBER)


def _validate_msg_file(file: UploadFile):
    if not file.filename.endswith('.msg') or file.size == 0:
        raise InvalidParamException("请上传有效的邮件文件")


def _get_threems_community_doc_source_list(community_uris) -> List[str]:
    result = []
    for uri in community_uris:
        uri_type, uri_info = get_uri_params(uri)
        doc_list = get_threems_community_doc_list([uri])
        if doc_list:
            result.extend([get_threems_source(uri_type, str(uri_info.group_id), str(doc.id)) for doc in doc_list])
    return result


def _get_threems_personal_doc_source_list(personal_uris) -> List[str]:
    blog_list = get_personal_blogs_by_asset_uri(
        json.dumps(personal_uris)
    )
    return [get_source_by_id(blog.id) for blog in blog_list]


def _get_threems_doc_source_list(threems_url_list: List[str]) -> List[str]:
    result = []
    community_uris, personal_uris = identify_uri_type(threems_url_list)

    if community_uris:
        result.extend(_get_threems_community_doc_source_list(community_uris))

    if personal_uris:
        result.extend(_get_threems_personal_doc_source_list(personal_uris))

    return result


def _get_doc_source_list_by_asset_uri(url_list: List[str]) -> List[str]:
    # 3ms/jiaxian
    doc_source_list = []
    jiaxian_url_list = []
    threems_url_list = []
    for url in url_list:
        if url.startswith(JIAXIAN_URL_PRE):
            jiaxian_url_list.append(url)
        if url.startswith(THREEMS_URL_PRE):
            threems_url_list.append(url)

    if jiaxian_url_list:
        jiaxian_article_infos = get_article_infos(json.dumps(jiaxian_url_list))
        doc_source_list.extend([jiaxian_article_info.source for jiaxian_article_info in jiaxian_article_infos])

    if threems_url_list:
        doc_source_list.extend(_get_threems_doc_source_list(threems_url_list))
    return doc_source_list


def _get_doc_id_list_by_asset_uri(single_url_asset_uri_list: List[str], kb_sn: str,
                                  security_level_list: List[SecurityLevel], session: Session) -> List[str]:
    special_asset_url_list = []
    general_asset_url_list = []
    result = []
    for url in single_url_asset_uri_list:
        if url.startswith(DBOX_URL_PRE):
            # dbox链接模糊匹配
            special_asset_url_list.append(url)
        else:
            general_asset_url_list.append(url)
    if special_asset_url_list:
        special_asset_url_pattern_list = [f"%{url}%" for url in special_asset_url_list]
        result.extend(get_doc_id_list_like_asset_uri(special_asset_url_pattern_list, kb_sn, security_level_list, session))
    if general_asset_url_list:
        result.extend(get_doc_id_list_by_asset_uri(general_asset_url_list, kb_sn, security_level_list, session))
    return result


def _get_matched_doc_id_list(kb_sn, url_list, security_level_list: List[SecurityLevel], session: Session) -> List[str]:
    """
        1、针对ebook、gitee、wiki、gitcode issue这种只支持单个链接的资产，可通过资产asset_uri进行查询
        2、针对3ms和jiaxian这两种支持多个链接的资产，根据每个链接去数据源查询对应的文档source，再通过source查询文档
        3、直接根据输入的url去数据库文档通过source字段进行匹配
        将以上匹配到的文档id进行合并
    """
    doc_id_set = set()
    # 仅支持单个链接的资产链接
    single_url_asset_uri_list = []
    # 支持多个链接的资产链接（3ms和稼先）
    multi_url_asset_uri_list = []
    for url in url_list:
        if url.startswith(JIAXIAN_URL_PRE) or url.startswith(THREEMS_URL_PRE):
            multi_url_asset_uri_list.append(url)
        else:
            single_url_asset_uri_list.append(url)

    # 通过资产uri匹配
    if single_url_asset_uri_list:
        doc_id_set.update(_get_doc_id_list_by_asset_uri(single_url_asset_uri_list, kb_sn, security_level_list, session))

    #  通过source匹配
    doc_source_set = set(url_list)
    if multi_url_asset_uri_list:
        doc_source_set.update(_get_doc_source_list_by_asset_uri(multi_url_asset_uri_list))
    doc_id_set.update(get_doc_id_list_by_doc_source(doc_source_set, kb_sn, security_level_list, session))

    if not doc_id_set:
        raise InvalidParamException("未关联到文档，请先创建资产")
    return [str(doc_id) for doc_id in doc_id_set]


def _get_security_level_list(security_level: str) -> List[SecurityLevel]:
    if not security_level:
        return []
    security_level_str_list = security_level.split(",")
    result = []
    for security_level_str in security_level_str_list:
        item = SecurityLevel.__members__.get(security_level_str)
        if item is not None:
            result.append(item)
    return result


def _validate_url_list(url_str: str, kb_sn: str, security_level: str, session: Session) -> (List[str], List[str]):
    url_list = url_str.split()
    for url in url_list:
        if not url.startswith("http") and not url.startswith("ssh://"):
            raise InvalidParamException("请输入有效的链接")
    security_level_list_list = _get_security_level_list(security_level)
    # 根据链接找到匹配到的文档id
    doc_id_list = _get_matched_doc_id_list(kb_sn, url_list, security_level_list_list, session)
    return url_list, doc_id_list


def spillover_mark(req: SpilloverMarkRequest, file: UploadFile, session: Session):
    # 验证知识库权限
    knowledge_base = validate_knowledge_base_member_operate_permission(session, req.kb_sn, req.uid)
    # 验证邮件
    _validate_msg_file(file)
    # 验证并解析链接
    url_list, doc_id_list = _validate_url_list(req.url, req.kb_sn, req.security_level, session)
    # 上传邮件文件到s3
    download_key = upload_file_as_bytes(str(uuid.uuid4()), file.file)
    spillover_manage = SpilloverManage(kb_sn=knowledge_base.sn, user_id=req.uid, url_list=url_list,
                                      doc_id_list=doc_id_list,
                                      download_key=download_key)
    session.add(spillover_manage)
    # 更新文档状态为可外溢
    update_spill_over_status(doc_id_list, True, session)
    session.commit()


def get_pbi_version_knowledge_info(req: PbiVersionRequest) -> Optional[IpdRAGKnowledgeBaseInfo]:
    """查询pbi版本对应的雅典娜知识集信息
    @param req: PbiVersionRequest 传参是pbi版本名称
    @return: 雅典娜知识集id，雅典娜知识集链接，ipd rag知识库id等
    """
    # 将版本名称中的空格替换成下划线。雅典娜知识集名称不能包含空格
    if not req.version_name:
        return None
    version_name = req.version_name.replace(" ", "_")
    ipd_rag_knowledge_base_list = query_asset_by_name_in_ipd_rag(PBI_ATHENA_KNOWLEDGE_NAME_PREFIX + version_name)
    if not ipd_rag_knowledge_base_list:
        return None
    return ipd_rag_knowledge_base_list[0]


@cached(cache=TTLCache(maxsize=1, ttl=3600))
def get_all_pbi_version_knowledge_id_list() -> List[str]:
    ipd_rag_knowledge_list = query_asset_by_name_in_ipd_rag(PBI_ATHENA_KNOWLEDGE_NAME_PREFIX)
    return [knowledge.id for knowledge in ipd_rag_knowledge_list]
