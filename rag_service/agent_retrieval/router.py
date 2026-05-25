from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from rag_service.agent_retrieval.models import (
    DocumentOutlineRequest,
    OriginalTextRequest,
    SearchSlicesRequest,
    SectionRequest,
    TableRequest,
)
from rag_service.agent_retrieval.service import AgentRetrievalService
from rag_service.agent_retrieval.skill_generator import generate_opencode_skill_package

try:
    from rag_service.database import yield_session
except Exception:
    def yield_session():
        return None


router = APIRouter(prefix="/agent/retrieval", tags=["Agent Retrieval"])


@router.post("/search_slices", response_model=None)
def search_slices(
    request: Request,
    req: SearchSlicesRequest,
    session: Any = Depends(yield_session),
) -> dict:
    return _dump(
        _service().search_slices(
            req,
            uid=_request_uid(request, req),
            session=session,
            request_id=_request_id(request),
        )
    )


@router.post("/get_document_outline", response_model=None)
def get_document_outline(request: Request, req: DocumentOutlineRequest) -> dict:
    return _dump_or_403(lambda: _service().get_document_outline(req, uid=_request_uid(request, req)))


@router.post("/get_section", response_model=None)
def get_section(request: Request, req: SectionRequest) -> dict:
    return _dump_or_403(lambda: _service().get_section(req, uid=_request_uid(request, req)))


@router.post("/get_table", response_model=None)
def get_table(request: Request, req: TableRequest) -> dict:
    return _dump_or_403(lambda: _service().get_table(req, uid=_request_uid(request, req)))


@router.post("/get_original_text", response_model=None)
def get_original_text(request: Request, req: OriginalTextRequest) -> dict:
    return _dump_or_403(lambda: _service().get_original_text(req, uid=_request_uid(request, req)))


@router.get("/opencode/skill-package", response_model=None)
def get_opencode_skill_package(base_url: Optional[str] = None, kb_sn_list: Optional[str] = None) -> dict:
    parsed_kb_sn_list = [item.strip() for item in (kb_sn_list or "").split(",") if item.strip()]
    return _dump(generate_opencode_skill_package(base_url=base_url, kb_sn_list=parsed_kb_sn_list))


def _request_uid(request: Request, req: Any) -> str:
    uid = getattr(getattr(request, "state", None), "uid", None)
    if not uid:
        uid = getattr(req, "uid", None)
    if not uid:
        raise HTTPException(status_code=401, detail="authenticated user is required")
    return str(uid)


def _request_id(request: Request) -> Optional[str]:
    headers = getattr(request, "headers", None)
    if headers is None:
        return None
    value = headers.get("X-Request-ID") or headers.get("x-request-id")
    return str(value) if value else None


def _service() -> AgentRetrievalService:
    return AgentRetrievalService()


def _dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _dump_or_403(factory: Callable[[], Any]) -> dict:
    try:
        return _dump(factory())
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
