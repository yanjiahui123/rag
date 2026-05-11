from rag_service.utils.his_util.tsl_fetch_util import build_tsl_node_tree

try:
    import armorrasp
    armorrasp.start()
except ImportError:
    pass
from rag_service.middleware.request_middleware import RequestLoggerMiddleware
import functools

import fastapi
import uvicorn
from fastapi_pagination import add_pagination
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette_context import plugins
from starlette_context.middleware import RawContextMiddleware

from rag_service.logger import UVICORN_LOG_CONFIG
from rag_service.rag_app.api_exceptions import (
    api_exception_handler,
    biz_exception_handler,
    default_exception_handler,
    get_all_api_exceptions,
)
from rag_service.rag_app.router import routers
from rag_service.utils.ebook_fetch_util import write_ssh_key_if_not_exists
from rag_service.utils.portal_util import refresh_department_manager, refresh_domain_manager
from rag_service.utils.redis_util import operation_switch_status_keyspace, rc

middleware = [
    Middleware(RequestLoggerMiddleware),
    Middleware(RawContextMiddleware, plugins=(plugins.RequestIdPlugin(), plugins.CorrelationIdPlugin()))]

app = fastapi.FastAPI(middleware=middleware)
_configured = False


def configure():
    global _configured
    if _configured:
        return
    _patch()
    _configure_router()
    _configure_pagination()
    _configure_exception_handler()
    _configured = True


def _configure_router():
    for router in routers:
        app.include_router(router)


def _configure_pagination():
    add_pagination(app)


def _configure_exception_handler():
    app.add_exception_handler(Exception, default_exception_handler)
    for api_exception in get_all_api_exceptions():
        app.add_exception_handler(api_exception, api_exception_handler)
        for biz_exception in api_exception.get_biz_exception():
            app.add_exception_handler(biz_exception, biz_exception_handler)


def _patch():
    Request.form = functools.partialmethod(Request.form, max_files=10000, max_fields=10000)


def _set_default_status_for_switch():
    rc.set(operation_switch_status_keyspace.resolve("enable_migrate_data"), "0")
    rc.set(operation_switch_status_keyspace.resolve("allow_synchronous_operation"), "0")
    rc.set(operation_switch_status_keyspace.resolve("open_redirection"), "1")


def _set_git_config():
    write_ssh_key_if_not_exists()


def main():
    configure()
    # refresh_department_manager()
    # refresh_domain_manager()
    # _set_default_status_for_switch()
    # _set_git_config()
    # 构建技术标准规范库的节点树
    # build_tsl_node_tree()
    uvicorn.run(app, host="0.0.0.0", port=8001, log_config=UVICORN_LOG_CONFIG)


configure()


if __name__ == "__main__":
    main()
