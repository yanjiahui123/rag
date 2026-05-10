from mimetypes import guess_type
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Request, Response, status, BackgroundTasks, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi_pagination import Page
from sqlmodel import Session
from starlette.concurrency import run_in_threadpool
from starlette.responses import PlainTextResponse
from starlette_context import context

from rag_service.config import MULTI_TURN_CONVERSATIONS_QUESTION_REWRITE_PROMPT_ID
from rag_service.constants import MULTI_TURN_CONVERSATIONS_QUESTION_REWRITE_PROMPT
from rag_service.database import yield_session
from rag_service.exceptions import OperationNotPermittedException
from rag_service.llms.llm import get_advance_answer_service, get_advance_stream_answer_service
from rag_service.logger import Module, get_logger
from rag_service.models.api.models import (
    AdvanceAnswerRequest,
    BatchCreateMemberResponse,
    CancelKbPermissionsReq,
    ConfigureKnowledgeBaseReq,
    CreateKnowledgeBaseReq,
    CreateUpdateBlacklistMemberReq,
    CreateUpdateBlacklistMultipleMemberReq,
    CreateUpdateMemberReq,
    CreateUpdateMultipleMemberReq,
    CreateWhitelistMemberReq,
    DeleteMemberReq,
    DeleteStopwordOrSynonymReq,
    DeleteWhitelistMemberReq,
    DeptManagerResponse,
    DomainManagerResponse,
    ExportKnowledgeBaseInfoRequest,
    FavoriteRequest,
    FeedbackAnswer,
    GetKbMigrateInfoReq,
    GetKbPromptReq,
    KnowledgeBaseBlacklistMemberShowInfo,
    KnowledgeBaseConfig,
    KnowledgeBaseInfo,
    KnowledgeBaseMemberShowInfo,
    KnowledgeBaseOwnerRequest,
    KnowledgeBaseShowInfo,
    LayerKnowledgeBaseReq,
    LlmAnswer,
    OpenKbPermissionsReq,
    OperateDefaultKnowledgeBaseReq,
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
    StopwordOrSynonymInfo,
    UpdateDepartmentKnowledgeBaseShareScope,
    UpdateDeptManagerReq,
    UpdateDomainManagerReq,
    UpdateFeedbackReq,
    UpdateKnowledgeBaseName,
    UpdateStopwordOrSynonymReq,
    UploadStopwordOrSynonymReq,
    VectorizationJobStatus, WhitelistResp, WhitelistReq, WhitelistQueryReq, ChangeOwnerReq, SpilloverMarkRequest,
    PbiVersionRequest,
)
from rag_service.models.enums import Domain, KnowledgeBasePermission, LlmModel, ServiceConfigType
from rag_service.models.generic.models import KnowledgeBaseMigrateInfo, IpdRAGKnowledgeBaseInfo
from rag_service.postprocess.qa import extract_answer_from_qa_document
from rag_service.postprocess.synonym import get_question_synonyms
from rag_service.rag_app.service import knowledge_base_service
from rag_service.rag_app.service.depends.authentication_depend import api_authorization, get_uid
from rag_service.rag_app.service.knowledge_base_service import (
    batch_cancel_favorite_knowledge_base,
    can_create_member,
    change_knowledge_base_analyzer_to_ik_analyzer,
    cite_reference_answer,
    create_batch_knowledge_base_member,
    delete_knowledge_base,
    delete_stopwords_or_synonyms_from_kb,
    export_knowledge_base_info_by_kb_sn,
    favorite_knowledge_base,
    feedback_answer_to_db,
    generate_related_questions_by_user_question,
    get_default_kb_config,
    get_department_manager,
    get_domain_manager,
    get_favorite_knowledge_base_list,
    get_fuzzy_match_knowledge_base,
    get_kb_config,
    get_kb_list_prompt,
    get_kb_prompt,
    get_knowledge_base_detail,
    get_knowledge_base_list,
    get_knowledge_base_list_info,
    get_knowledge_base_list_no_page,
    get_knowledge_base_member_list,
    get_knowledge_base_page,
    get_layer_knowledge_base_list,
    get_llm_answer,
    get_llm_stream_answer,
    get_qa_stream_answer,
    get_random_doc_from_knowledge_base,
    get_self_member_type,
    get_stopwords_or_synonyms_by_kb_sn,
    get_top_score_rerank_search_data,
    is_super_root,
    retrieve_documents,
    store_stopword_or_synonym_to_kb,
    synchronize_knowledge_base_synonym_stopword,
    update_kb_config,
    update_knowledge_base_name,
    update_response_log_process_status,
    update_stop_or_synonym_by_id,
    validate_retrieved_documents, permission_judge,
)
from rag_service.telemetry.collect_usage import (
    async_collect_advance_request_usage,
    async_collect_usage,
    collect_advance_request_usage,
    collect_usage,
)
from rag_service.utils.background_task import ensure_background_tasks
from rag_service.utils.redis_util import (
    operation_switch_status_keyspace,
    rc,
)
from rag_service.utils.response_util import set_content_disposition

router = APIRouter(prefix="/kb", tags=["Knowledge Base"], dependencies=[Depends(api_authorization)])
logger = get_logger(module=Module.APP)


class EvidencePackageRequest(QueryRequest):
    pass


@router.post("/create")
def create(request: Request, req: CreateKnowledgeBaseReq, session: Session = Depends(yield_session)) -> str:
    req.owner = request.state.uid if request.state.uid else req.owner
    return knowledge_base_service.create_knowledge_base(req, session)


@router.post("/get_related_docs")
@ensure_background_tasks
@collect_usage
def get_related_docs(
    request: Request, req: QueryRequest, background_tasks: BackgroundTasks, session: Session = Depends(yield_session)
) -> List[RetrievedDocument]:
    req.request_id = context.data.get("X-Request-ID")
    req.uid = request.state.uid if request.state.uid else req.uid
    related_docs = knowledge_base_service.get_related_docs(req, background_tasks, session)
    return related_docs


@router.post("/get_evidence_packages", response_model=None)
@ensure_background_tasks
@collect_usage
def get_evidence_packages(
    request: Request,
    req: EvidencePackageRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(yield_session),
) -> dict:
    req.request_id = context.data.get("X-Request-ID")
    req.uid = request.state.uid if request.state.uid else req.uid
    return knowledge_base_service.get_evidence_packages(req, background_tasks, session).to_dict()


@router.post("/get_answer_evidence", response_model=None)
@ensure_background_tasks
@collect_usage
def get_answer_evidence(
    request: Request,
    req: EvidencePackageRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(yield_session),
) -> dict:
    req.request_id = context.data.get("X-Request-ID")
    req.uid = request.state.uid if request.state.uid else req.uid
    return knowledge_base_service.get_answer_evidence(req, background_tasks, session)


@router.get("/evidence_artifacts/{artifact_id}", response_model=None)
def get_evidence_artifact(request: Request, artifact_id: str) -> Response:
    object_key, content = knowledge_base_service.get_evidence_artifact(artifact_id, request.state.uid)
    return Response(content=content, media_type=guess_type(object_key)[0] or "application/octet-stream")


@router.post("/get_answer")
@ensure_background_tasks
@collect_usage
def get_answer(request: Request, req: QueryRequest, background_tasks: BackgroundTasks, session: Session = Depends(yield_session)) -> LlmAnswer:
    req.uid = request.state.uid if request.state.uid else req.uid
    req.request_id = context.data.get("X-Request-ID")
    ans = get_llm_answer(req, background_tasks, session)
    ans.request_id = req.request_id
    return ans


@router.post("/get_stream_answer", response_class=HTMLResponse)
@ensure_background_tasks
@async_collect_usage
async def get_stream_answer(
    request: Request, req: QueryRequest, response: Response, background_tasks: BackgroundTasks, session: Session = Depends(yield_session)
):
    req.request_id = context.data.get("X-Request-ID")
    req.uid = request.state.uid if request.state.uid else req.uid
    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"

    kb_sn_list = req.kb_sn_list if req.kb_sn_list else [req.kb_sn]
    knowledge_base_list = permission_judge(session, kb_sn_list, req.uid)
    documents = await run_in_threadpool(retrieve_documents, req, knowledge_base_list, session, True, background_tasks)
    validate_retrieved_documents(documents, req)

    qa_answer, document = extract_answer_from_qa_document(req.question, documents)
    if qa_answer:
        return StreamingResponse(
            await run_in_threadpool(get_qa_stream_answer, req, document, qa_answer),
            status_code=status.HTTP_200_OK,
            headers=response.headers,
        )

    synonyms = get_question_synonyms(session, req.question, kb_sn_list)
    kb_prompt = (
        req.user_custom_retrieve_config.prompt
        if req.user_custom_retrieve_config and req.user_custom_retrieve_config.prompt
        else get_kb_prompt(kb_sn_list, session)
    )
    return StreamingResponse(
        await run_in_threadpool(get_llm_stream_answer, req, knowledge_base_list, documents, synonyms, background_tasks, kb_prompt),
        status_code=status.HTTP_200_OK,
        headers=response.headers,
    )


@router.post("/list", response_model=Page[KnowledgeBaseShowInfo])
def get_kb_list(
    request: Request, req_param: PrivaterKnowledgeBaseListReq, session: Session = Depends(yield_session)
) -> Page[KnowledgeBaseShowInfo]:
    req_param.member = request.state.uid if request.state.uid else req_param.member
    knowledge_base_list = get_knowledge_base_list(session, req_param)
    return knowledge_base_list


@router.get("/detail", response_model=KnowledgeBaseShowInfo)
def get_kb_detail(
    request: Request, kb_sn: str, member: str, session: Session = Depends(yield_session)
) -> KnowledgeBaseShowInfo:
    member = request.state.uid if request.state.uid else member
    return get_knowledge_base_detail(session, kb_sn, member)


@router.post("/layer_list", response_model=Page[PublicKnowledgeBaseShowInfo])
def get_public_kb_list(
    request: Request, req: LayerKnowledgeBaseReq, session: Session = Depends(yield_session)
) -> Page[PublicKnowledgeBaseShowInfo]:
    req.member = request.state.uid if request.state.uid else req.member
    return get_layer_knowledge_base_list(session, req)


# 用于判断是否是领域知识库或者部门知识库的全部公开，如果是则默认没有添加member这种权限(所有人默认为member角色)
@router.get("/can_create_member")
def kb_can_create_member(kb_sn: str, session: Session = Depends(yield_session)) -> bool:
    return can_create_member(session, kb_sn)


@router.get("/list_no_page")
def get_kb_list_no_page(
    request: Request,
    member: str,
    top_k: int = Query(50, ge=1, le=100),
    permission: KnowledgeBasePermission = KnowledgeBasePermission.READ,
    session: Session = Depends(yield_session),
) -> List[KnowledgeBaseInfo]:
    member = request.state.uid if request.state.uid else member
    return get_knowledge_base_list_no_page(member, permission, session)[:top_k]


@router.get("/all_list")
def get_all_kb_list_page(
    request: Request,
    member: str,
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(50, ge=1, le=100, description="Page size"),
    permission: KnowledgeBasePermission = KnowledgeBasePermission.READ,
    session: Session = Depends(yield_session),
    keyword: str = None,
    filter_athena_kb: bool = False,  # 用于过滤掉与雅典娜绑定的特殊知识库
) -> Page[KnowledgeBaseInfo]:
    member = request.state.uid if request.state.uid else member
    return get_knowledge_base_page(member, page, size, permission, session, keyword, filter_athena_kb)


@router.delete("/delete")
def delete_kb(request: Request, kb_sn: str, operator: str, session: Session = Depends(yield_session)):
    operator = request.state.uid if request.state.uid else operator
    delete_knowledge_base(kb_sn, operator, session)
    return f"知识库 <{kb_sn}> 已删除。"


@router.post("/favorite")
def favorite_kb(request: Request, req: FavoriteRequest, session: Session = Depends(yield_session)):
    operator = request.state.uid if request.state.uid else req.operator
    return favorite_knowledge_base(req.kb_sn, operator, req.is_favorite, session)


@router.post("/batch_cancel_favorite")
def batch_cancel_favorite_kb(request: Request, req: FavoriteRequest, session: Session = Depends(yield_session)):
    operator = request.state.uid if request.state.uid else req.operator
    return batch_cancel_favorite_knowledge_base(req.kb_sn_list, operator, session)


@router.post("/favorites_list", response_model=Page[KnowledgeBaseShowInfo])
def get_favorites_kb_list(
    request: Request, req_param: PrivaterKnowledgeBaseListReq, session: Session = Depends(yield_session)
) -> Page[KnowledgeBaseShowInfo]:
    req_param.member = request.state.uid if request.state.uid else req_param.member
    return get_favorite_knowledge_base_list(session, req_param)


@router.post("/update_kb_name")
def update_kb_name(request: Request, req: UpdateKnowledgeBaseName, session: Session = Depends(yield_session)):
    req.operator = request.state.uid if request.state.uid else req.operator
    update_knowledge_base_name(session, req.kb_sn, req.kb_name, req.operator)
    return f"知识库 <{req.kb_sn}> 名称已成功更改为{req.kb_name}。"


@router.post("/try_it_out")
def try_it_out(kb_sn: str, uid: str = Depends(get_uid)):
    knowledge_base_service.try_it_out(kb_sn, uid)
    return f"试用知识库 <{kb_sn}> 成功。"


@router.get("/member_infomation_search")
def member_information_search(lang: str, search_value: str, search_type: str, page_size: str, page: str):
    return knowledge_base_service.member_information_search(lang, search_value, search_type, page_size, page)


@router.post("/create_member")
def create_knowledge_base_member(
    request: Request, req: CreateUpdateMemberReq, session: Session = Depends(yield_session)
):
    req.operator = request.state.uid if request.state.uid else req.operator
    knowledge_base_service.create_knowledge_base_member(req, session)
    logger.info(
        "管理员<%s>在知识库<%s>添加成员<%s>，成员类型为<%s>",
        req.operator,
        req.kb_sn,
        req.employee_number,
        req.member_type,
    )
    return f"知识库添加成员<{req.employee_number}>成功。"


@router.post("/create_multiple_members")
def create_knowledge_base_multiple_member(
    request: Request, req: CreateUpdateMultipleMemberReq, session: Session = Depends(yield_session)
) -> BatchCreateMemberResponse:
    req.operator = request.state.uid if request.state.uid else req.operator
    failed_set, succeed_set = create_batch_knowledge_base_member(req, session)
    logger.info(
        "管理员<%s>在知识库<%s>成功添加普通成员%s，" "成功添加管理员%s",
        req.operator,
        req.kb_sn,
        req.simple_employee_number_list,
        req.control_employee_number_list,
    )
    return BatchCreateMemberResponse(succeed_member_set=succeed_set, failed_member_set=failed_set)


@router.post("/update_member")
def update_knowledge_base_member(
    request: Request, req: CreateUpdateMemberReq, session: Session = Depends(yield_session)
):
    req.operator = request.state.uid if request.state.uid else req.operator
    knowledge_base_service.update_knowledge_base_member(req, session)
    logger.info(
        "管理员<%s>在知识库<%s>更新成员<%s>，成员类型为<%s>",
        req.operator,
        req.kb_sn,
        req.employee_number,
        req.member_type,
    )
    return f"知识库<{req.kb_sn}>更新成员<{req.employee_number}>成功。"


@router.get("/member_list", response_model=Page[KnowledgeBaseMemberShowInfo])
def get_kb_member_list(
    kb_sn: str,
    search_txt: Optional[str] = None,
    session: Session = Depends(yield_session),
) -> Page[KnowledgeBaseMemberShowInfo]:
    return get_knowledge_base_member_list(kb_sn, search_txt, session)


@router.get("/self_member_type")
def get_self_kb_member_type(
    request: Request, kb_sn: str, employee_number: str, session: Session = Depends(yield_session)
):
    employee_number = request.state.uid if request.state.uid else employee_number
    return get_self_member_type(kb_sn, employee_number, session)


@router.get("/fuzzy_match_knowledge_base")
def fuzzy_match_knowledge_base(
    request: Request, kb_info: str, employee_number: str, session: Session = Depends(yield_session)
):
    """模糊匹配知识库(通过知识库名称模糊匹配或者知识库sn号去匹配)"""
    employee_number = request.state.uid if request.state.uid else employee_number
    return get_fuzzy_match_knowledge_base(kb_info, employee_number, session)


@router.post("/delete_member")
def batch_delete_knowledge_base_member(
    request: Request, req: DeleteMemberReq, session: Session = Depends(yield_session)
):
    req.operator = request.state.uid if request.state.uid else req.operator
    knowledge_base_service.batch_delete_knowledge_base_member(req, session)
    logger.info("管理员<%s>在知识库<%s>删除成员<%s>", req.operator, req.kb_sn, req.employee_numbers)
    return "知识库删除成员成功。"


@router.post("/feedback")
def feedback_answer(feedback: FeedbackAnswer, background_tasks: BackgroundTasks, session: Session = Depends(yield_session)):
    feedback_answer_to_db(feedback, session, background_tasks)
    if feedback.request_id:
        return f"Request id <{feedback.request_id}> 反馈成功"
    return f"Question id <{feedback.question_id}> 反馈成功"


@router.post("/update_feedback")
def update_process_status(request: Request, req: UpdateFeedbackReq, session: Session = Depends(yield_session)):
    req.operator = request.state.uid if request.state.uid else req.operator
    update_response_log_process_status(req, session)
    return f"User {req.operator} update feedback {req.request_id or req.question_id} success"


@router.post("/generate_related_questions")
def generate_related_questions(
    request: Request, req: QueryRequest, session: Session = Depends(yield_session)
) -> List[str]:
    req.uid = request.state.uid if request.state.uid else req.uid
    return generate_related_questions_by_user_question(req, session)


@router.post("/export_knowledge_base_info")
def export_knowledge_base_info(
    request: Request, req: ExportKnowledgeBaseInfoRequest, session: Session = Depends(yield_session)
):
    req.uid = request.state.uid if request.state.uid else req.uid
    file_path, file_name = export_knowledge_base_info_by_kb_sn(req, session)
    resp = Response(file_path, media_type=guess_type(file_name)[0] or "application/octet-stream")
    return set_content_disposition(resp, file_name)


@router.post("/upload_stopword_or_synonym")
def upload_stopword_or_synonym(
    request: Request, req: UploadStopwordOrSynonymReq, session: Session = Depends(yield_session)
) -> str:
    req.uid = request.state.uid if request.state.uid else req.uid
    return store_stopword_or_synonym_to_kb(req, session)


@router.post("/delete_stopword_or_synonym")
def delete_stopword_or_synonym(
    request: Request, req: DeleteStopwordOrSynonymReq, session: Session = Depends(yield_session)
) -> str:
    req.uid = request.state.uid if request.state.uid else req.uid
    return delete_stopwords_or_synonyms_from_kb(session, req)


@router.post("/stopword_or_synonym_list")
def get_stopword_or_synonym_list(
    request: Request, req: QueryStopwordOrSynonymReq, session: Session = Depends(yield_session)
) -> Page[StopwordOrSynonymInfo]:
    req.uid = request.state.uid if request.state.uid else req.uid
    return get_stopwords_or_synonyms_by_kb_sn(session, req)


@router.post("/update_stopword_or_synonym_list")
def update_stopword_or_synonym_list(
    request: Request, req: UpdateStopwordOrSynonymReq, session: Session = Depends(yield_session)
) -> str:
    req.uid = request.state.uid if request.state.uid else req.uid
    return update_stop_or_synonym_by_id(session, req)


@router.get("/knowledge_base_reindex")
def reindex_knowledge_base(
    request: Request, kb_sn: str, member: str, session: Session = Depends(yield_session)
) -> List[VectorizationJobStatus]:
    member = request.state.uid if request.state.uid else member
    return change_knowledge_base_analyzer_to_ik_analyzer(session, kb_sn, member)


@router.get("/synchronize_knowledge_base_synonym_stopword")
def synchronous_knowledge_base_synonym_and_stopword(
    request: Request, kb_sn: str, member: str, session: Session = Depends(yield_session)
) -> List[VectorizationJobStatus]:
    member = request.state.uid if request.state.uid else member
    return synchronize_knowledge_base_synonym_stopword(session, kb_sn, member)


@router.post("/update_knowledge_base_config")
def update_knowledge_base_config(
    request: Request, req: ConfigureKnowledgeBaseReq, session: Session = Depends(yield_session)
) -> str:
    member = request.state.uid if request.state.uid else req.member
    return update_kb_config(session, req.kb_sn, member, req.config)


@router.get("/knowledge_base_config")
def get_knowledge_base_config(
    request: Request, kb_sn: str, member: str, session: Session = Depends(yield_session)
) -> KnowledgeBaseConfig:
    member = request.state.uid if request.state.uid else member
    return get_kb_config(session, kb_sn, member)


@router.get("/default_knowledge_base_config")
def get_default_knowledge_base_config(
    request: Request, kb_sn: str, member: str, session: Session = Depends(yield_session)
) -> KnowledgeBaseConfig:
    member = request.state.uid if request.state.uid else member
    return get_default_kb_config(session, kb_sn, member)


@router.post("/get_random_doc")
def get_random_doc_from_kb(
    request: Request, req: RandomDocRequest, session: Session = Depends(yield_session)
) -> List[RetrievedDocument]:
    req.uid = request.state.uid if request.state.uid else req.uid
    return get_random_doc_from_knowledge_base(session, req)


@router.post("/get_advance_answer")
@collect_advance_request_usage
def get_advance_answer(request: Request, req: AdvanceAnswerRequest, background_tasks: BackgroundTasks) -> str:
    req.uid = request.state.uid if request.state.uid else req.uid
    req.request_id = context.data.get("X-Request-ID")
    return get_advance_answer_service(req, background_tasks)


@router.post("/get_advance_stream_answer", response_class=HTMLResponse)
@async_collect_advance_request_usage
async def get_advance_stream_answer(request: Request, req: AdvanceAnswerRequest, response: Response, background_tasks: BackgroundTasks):
    req.uid = request.state.uid if request.state.uid else req.uid
    req.request_id = context.data.get("X-Request-ID")
    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"

    return StreamingResponse(
        await run_in_threadpool(get_advance_stream_answer_service, req, background_tasks),
        status_code=status.HTTP_200_OK,
        headers=response.headers,
    )


@router.get("/department_manager_list")
def get_department_manager_list(
    request: Request, operator: str, session: Session = Depends(yield_session)
) -> List[DeptManagerResponse]:
    uid = request.state.uid if request.state.uid else operator
    return get_department_manager(uid, session)


@router.get("/domain_manager_list")
def get_domain_manager_list(request: Request, operator: str) -> List[DomainManagerResponse]:
    uid = request.state.uid if request.state.uid else operator
    return get_domain_manager(uid)


@router.post("/add_department_manager")
def add_department_manager(request: Request, req: UpdateDeptManagerReq) -> str:
    operator = request.state.uid if request.state.uid else req.operator
    result = knowledge_base_service.add_department_manager(operator, req.dept_code, req.w3_account)
    return "部门管理员添加成功" if result else "部门管理员添加失败"


@router.post("/add_domain_manager")
def add_domain_manager(request: Request, req: UpdateDomainManagerReq) -> str:
    operator = request.state.uid if request.state.uid else req.operator
    result = knowledge_base_service.add_domain_manager(operator, req.domain, req.w3_account)
    return "领域管理员添加成功" if result else "领域管理员添加失败"


@router.post("/delete_department_manager")
def delete_department_manager(request: Request, req: UpdateDeptManagerReq) -> str:
    operator = request.state.uid if request.state.uid else req.operator
    result = knowledge_base_service.delete_department_manager(operator, req.dept_code, req.w3_account)
    return "部门管理员删除成功" if result else "部门管理员删除失败"


@router.post("/delete_domain_manager")
def delete_domain_manager(request: Request, req: UpdateDomainManagerReq) -> str:
    operator = request.state.uid if request.state.uid else req.operator
    result = knowledge_base_service.delete_domain_manager(operator, req.domain, req.w3_account)
    return "领域管理员删除成功" if result else "领域管理员删除失败"


@router.post("/create_blacklist_member")
def create_knowledge_base_blacklist_member(
    request: Request, req: CreateUpdateBlacklistMemberReq, session: Session = Depends(yield_session)
) -> str:
    req.operator = request.state.uid if request.state.uid else req.operator
    knowledge_base_service.create_knowledge_base_blacklist_member(req, session)
    logger.info(
        "管理员<%s>在知识库<%s>添加黑名单成员<%s>",
        req.operator,
        req.kb_sn,
        req.employee_number,
    )
    return f"知识库黑名单添加成员<{req.employee_number}>成功。"


@router.post("/create_multiple_blacklist_members")
def create_multiple_knowledge_base_blacklist_member(
    request: Request, req: CreateUpdateBlacklistMultipleMemberReq, session: Session = Depends(yield_session)
) -> str:
    req.operator = request.state.uid if request.state.uid else req.operator
    message = knowledge_base_service.batch_create_knowledge_base_blacklist_member(req, session)
    logger.info("管理员<%s>在知识库<%s>的操作结果如下：%s", req.operator, req.kb_sn, message)
    return message


@router.post("/delete_blacklist_member")
def batch_delete_knowledge_base_blacklist_member(
    request: Request, req: DeleteMemberReq, session: Session = Depends(yield_session)
) -> str:
    req.operator = request.state.uid if request.state.uid else req.operator
    knowledge_base_service.batch_delete_knowledge_base_blacklist_member(req, session)
    logger.info("管理员<%s>在知识库<%s>删除成员<%s>", req.operator, req.kb_sn, req.employee_numbers)
    return "知识库删除黑名单成员成功。"


@router.get("/blacklist_member_list", response_model=Page[KnowledgeBaseBlacklistMemberShowInfo])
def get_kb_blacklist_member_list(
    request: Request,
    kb_sn: str,
    session: Session = Depends(yield_session),
) -> Page[KnowledgeBaseBlacklistMemberShowInfo]:
    return knowledge_base_service.get_knowledge_base_blacklist_member_list(kb_sn, request.state.uid, session)


@router.get("/redirect")
def redirect(request: Request, uid: str, session: Session = Depends(yield_session)) -> bool:
    if rc.get(operation_switch_status_keyspace.resolve("open_redirection")) != "1":
        return False
    uid = request.state.uid if request.state.uid else uid
    return knowledge_base_service.redirect(uid, session)


@router.post("/create_whitelist_member")
def batch_create_whitelist_member(req: CreateWhitelistMemberReq, session: Session = Depends(yield_session)) -> str:
    message = knowledge_base_service.batch_create_whitelist_member(req.employee_numbers, session)
    logger.info("创建结果如下：%s", message)
    return message


@router.post("/delete_whitelist_member")
def batch_delete_whitelist_member(req: DeleteWhitelistMemberReq, session: Session = Depends(yield_session)) -> str:
    knowledge_base_service.batch_delete_whitelist_member(req.employee_numbers, session)
    logger.info("成功删除成员<%s>", req.employee_numbers)
    return "删除白名单成员成功。"


@router.post("/control_redirection_switch")
def control_redirection_switch(request: Request, state: bool):
    operator = request.state.uid if request.state.uid else ""
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限控制页面重定向功能")
    rc.set(operation_switch_status_keyspace.resolve("open_redirection"), "1" if state else "0")
    if state:
        return "已开启页面重定向"
    return "已关闭页面重定向"


@router.get("/get_redirection_state")
def get_redirection_state(request: Request):
    operator = request.state.uid if request.state.uid else ""
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限删除领域管理员")
    return rc.get(operation_switch_status_keyspace.resolve("open_redirection")) == "1"


@router.get("/list_llm_models", response_model=List[str])
def get_support_query_strategy() -> List[str]:
    return LlmModel.get_llm_model_list()


@router.post("/update_dept_kb_share_scope")
def update_dept_kb_share_scope(
    request: Request, req: UpdateDepartmentKnowledgeBaseShareScope, session: Session = Depends(yield_session)
):
    req.operator = request.state.uid if request.state.uid else req.operator
    knowledge_base_service.update_dept_kb_share_scope(session, req)
    return f"知识库 <{req.kb_sn}> 公开范围已成功更改为{req.share_scope}。"


@router.post("/migrate_knowledge_base_to_ipd_rag")
def migrate_knowledge_base_to_ipd_rag(kb_sn: str, session: Session = Depends(yield_session)):
    knowledge_base_service.migrate_knowledge_base_to_ipd_rag(session, kb_sn)
    return f"开始迁移知识库{kb_sn}下的所有资产"


@router.post("/get_kb_prompt")
def get_knowledge_list_prompt(
    request: Request, req_param: GetKbPromptReq, session: Session = Depends(yield_session)
) -> str:
    member = request.state.uid if request.state.uid else req_param.member
    return get_kb_list_prompt(req_param.kb_sn_list, member, req_param.question, session)


@router.post("/cite_answer")
def cite_answer(referenceAnswerReq: ReferenceAnswerReq) -> ReferenceAnswerResp:
    return cite_reference_answer(referenceAnswerReq)


@router.post("/open_department")
def open_knowledge_base_permissions_to_department(
    request: Request, req: OpenKbPermissionsReq, session: Session = Depends(yield_session)
):
    req.operator = request.state.uid if request.state.uid else req.operator
    knowledge_base_service.open_knowledge_base_permissions_to_department(req, session)
    logger.info(
        "管理员<%s>在知识库<%s>对部门<%s>公开成功",
        req.operator,
        req.kb_sn,
        req.dept_name,
    )
    return f"知识库<{req.kb_sn}>对部门<{req.dept_name}>公开成功。"


@router.post("/cancel_department")
def cancel_knowledge_base_permissions_to_department(
    request: Request, req: CancelKbPermissionsReq, session: Session = Depends(yield_session)
):
    req.operator = request.state.uid if request.state.uid else req.operator
    knowledge_base_service.cancel_knowledge_base_permissions_to_department(req, session)
    logger.info(
        "管理员<%s>在知识库<%s>对部门<%s>取消权限成功",
        req.operator,
        req.kb_sn,
        req.dept_name,
    )
    return f"知识库<{req.kb_sn}>对部门<{req.dept_name}>取消权限成功。"


@router.post("/get_kb_list_by_sn_list")
def get_kb_list_by_sn_list(kb_sn_list: List[str], session: Session = Depends(yield_session)):
    return knowledge_base_service.get_kb_list_by_sn_list(kb_sn_list, session)


@router.post("/set_default")
def set_default_knowledge_base(
    request: Request, req: OperateDefaultKnowledgeBaseReq, session: Session = Depends(yield_session)
):
    req.operator = request.state.uid if request.state.uid else req.operator
    return knowledge_base_service.set_default_knowledge_base(req, session)


@router.get("/all_kb_list")
def get_all_kb_list(
    request: Request,
    member: str,
    permission: KnowledgeBasePermission = KnowledgeBasePermission.READ,
    session: Session = Depends(yield_session),
    keyword: str = None,
):
    member = request.state.uid if request.state.uid else member
    return knowledge_base_service.get_all_read_or_operate_permission_knowledge_base(
        session, member, permission, keyword
    )


@router.post("/get_rewrite_question", response_class=PlainTextResponse)
def get_rewrite_question(request: Request, req: QueryRequest, session: Session = Depends(yield_session)):
    req.uid = request.state.uid if request.state.uid else req.uid
    return knowledge_base_service.question_rewrite(
        req.question,
        MULTI_TURN_CONVERSATIONS_QUESTION_REWRITE_PROMPT,
        MULTI_TURN_CONVERSATIONS_QUESTION_REWRITE_PROMPT_ID,
        req.historical_questions,
        session,
    )


@router.post("/control_migrate_switch")
def control_migrate_switch(request: Request, state: bool):
    operator = request.state.uid if request.state.uid else ""
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限控制数据迁移功能")
    rc.set(operation_switch_status_keyspace.resolve("enable_migrate_data"), "1" if state else "0")
    if state:
        return "已开启数据迁移"
    return "已关闭数据迁移"


@router.post("/control_synchronous_operation_switch")
def control_synchronous_operation_switch(request: Request, state: bool):
    operator = request.state.uid if request.state.uid else ""
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限控制数据同步更新功能")
    rc.set(operation_switch_status_keyspace.resolve("allow_synchronous_operation"), "1" if state else "0")
    if state:
        return "已开启IPD_RAG同步操作"
    return "已关闭IPD_RAG同步操作"


@router.get("/get_synchronous_operation_switch_status")
def get_synchronous_operation_switch_status(request: Request):
    operator = request.state.uid if request.state.uid else ""
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限执行此操作")
    return rc.get(operation_switch_status_keyspace.resolve("allow_synchronous_operation")) == "1"


@router.post("/control_kb_synchronous_update_switch")
def control_kb_synchronous_update_switch(
    request: Request, kb_sn: str, state: bool, session: Session = Depends(yield_session)
):
    operator = request.state.uid if request.state.uid else ""
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限执行此操作")
    knowledge_base_service.control_kb_synchronous_update_switch(session, kb_sn, state)
    if state:
        return f"知识库<{kb_sn}>开启IPD_RAG同步更新"
    return f"知识库<{kb_sn}>关闭IPD_RAG同步更新"


@router.post("/get_all_knowledge_bases_migrate_info")
def get_all_knowledge_bases_migrate_info(
    request: Request, req: GetKbMigrateInfoReq, session: Session = Depends(yield_session)
) -> Page[KnowledgeBaseMigrateInfo]:
    operator = request.state.uid if request.state.uid else req.request_user_id
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限执行此操作")
    return knowledge_base_service.get_all_knowledge_bases_migrate_info(session, req)


@router.post("/quit_knowledge_base")
def quit_knowledge_base(request: Request, req: QuitKbMemberReq, session: Session = Depends(yield_session)):
    req.operator = request.state.uid if request.state.uid else req.operator
    knowledge_base_service.quit_knowledge_base(req, session)
    logger.info("用户<%s>退出知识库<%s>", req.operator, req.kb_sn)
    return "退出知识库成功。"


@router.post("/get_knowledge_base_owner")
def get_knowledge_base_owner(
    knowledge_base_owner_request: KnowledgeBaseOwnerRequest, session: Session = Depends(yield_session)
):
    return get_knowledge_base_list_info(session, knowledge_base_owner_request)


@router.get("/get_knowledge_base_owner_dept_info")
def get_knowledge_base_owner_dept_info(
    request: Request, operator: str, session: Session = Depends(yield_session)
) -> List[dict]:
    operator = request.state.uid if request.state.uid else operator
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限执行该操作")
    return knowledge_base_service.get_knowledge_base_owner_dept_info(session)


@router.post("/update_synchronous_operation_department")
def update_synchronous_operation_department(
    request: Request, operator: str, dept_code: str, state: bool, session: Session = Depends(yield_session)
):
    operator = request.state.uid if request.state.uid else operator
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限执行该操作")
    return knowledge_base_service.update_ipd_rag_whitelist(dept_code, state, session)


@router.get("/get_synchronous_operation_department")
def get_synchronous_operation_department(request: Request, operator: str, session: Session = Depends(yield_session)):
    operator = request.state.uid if request.state.uid else operator
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限执行该操作")
    return knowledge_base_service.get_ipd_rag_whitelist(session)


@router.get("/get_department_synchronous_operation_status")
def get_department_synchronous_operation_status(
    request: Request, operator: str, dept_code: str, session: Session = Depends(yield_session)
):
    operator = request.state.uid if request.state.uid else operator
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限执行该操作")
    return knowledge_base_service.validate_dept_in_ipd_rag_whitelist(session, dept_code)


@router.post("/rerank_search_data")
def rerank_search_data(request: Request, req: RerankSearchDataReq):
    req.uid = request.state.uid if request.state.uid else req.uid
    return get_top_score_rerank_search_data(req)


@router.get("/get_domain_kb_sn_list")
def get_domain_kb_sn_list(domain: Domain, session: Session = Depends(yield_session)):
    return knowledge_base_service.get_domain_kb_sn_list(domain, session)


@router.post("/add_migrated_dept")
def add_migrated_dept(
    request: Request, operator: str, dept_code: str, session: Session = Depends(yield_session)
):
    operator = request.state.uid if request.state.uid else operator
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限执行该操作")
    return knowledge_base_service.add_migrated_dept(dept_code, session)


@router.post("/delete_migrated_dept")
def delete_migrated_dept(
    request: Request, operator: str, dept_code: str, session: Session = Depends(yield_session)
):
    operator = request.state.uid if request.state.uid else operator
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限执行该操作")
    return knowledge_base_service.delete_migrated_dept(dept_code, session)


@router.get("/get_migrated_dept_list")
def get_migrated_dept_list(request: Request, operator: str, session: Session = Depends(yield_session)):
    operator = request.state.uid if request.state.uid else operator
    if not is_super_root(operator):
        raise OperationNotPermittedException(f"<{operator}>非RAG后台管理员，无权限执行该操作")
    return knowledge_base_service.get_migrated_dept_list(session)


@router.post("/get_whitelist_list")
def get_whitelist_list(request: Request, req: WhitelistQueryReq, session: Session = Depends(yield_session)
) -> Page[WhitelistResp]:
    req.operator = request.state.uid if request.state.uid else req.operator
    return knowledge_base_service.get_whitelist_list(req, session)


@router.post("/add_whitelist_list")
def add_whitelist_list(request: Request, req: WhitelistReq, session: Session = Depends(yield_session)) -> str:
    req.operator = request.state.uid if request.state.uid else req.operator
    knowledge_base_service.add_whitelist_list(req, session)
    return "白名单添加成功"


@router.post("/delete_whitelist_list")
def delete_whitelist_list(request: Request, req: WhitelistReq, session: Session = Depends(yield_session)) -> str:
    req.operator = request.state.uid if request.state.uid else req.operator
    knowledge_base_service.delete_whitelist_list(req, session)
    return "白名单删除成功"


@router.post("/change_owner")
def change_owner(request: Request, req: ChangeOwnerReq, session: Session = Depends(yield_session)) -> str:
    req.operator = request.state.uid if request.state.uid else req.operator
    knowledge_base_service.change_owner(req, session)
    return "转让owner成功"


@router.get("/get_kb_count_can_create")
def get_kb_count_can_create(request: Request, dept_or_domain_code: str, config_type: ServiceConfigType,
                            session: Session = Depends(yield_session)) -> int:
    return knowledge_base_service.get_kb_count_can_create(dept_or_domain_code, config_type, session)


@router.post("/spillover_mark")
def spillover_mark(
    request: Request,
    file: UploadFile,
    req: SpilloverMarkRequest = Depends(),
    session: Session = Depends(yield_session),
) -> str:
    """
    外溢知识标记：提供链接、评审邮件，选择文档密级
    """
    req.uid = request.state.uid if request.state.uid else req.uid
    knowledge_base_service.spillover_mark(req, file, session)
    return "操作成功"


@router.post("/get_pbi_version_knowledge_info")
def get_pbi_version_knowledge_info(
    req: PbiVersionRequest,
) -> Optional[IpdRAGKnowledgeBaseInfo]:
    """
       查询pbi版本节点对应的雅典娜知识集链接
    """
    return knowledge_base_service.get_pbi_version_knowledge_info(req)
