import importlib
from pathlib import Path
from typing import List

from fastapi import APIRouter

__all__ = ["routers"]

routers: List[APIRouter] = []

base_path = Path(__file__).parent
for f in [f for f in base_path.rglob("*.py") if not f.name.endswith("__init__.py")]:
    module_name = f"{__package__}.{f.relative_to(base_path).with_suffix('').as_posix().replace('/', '.')}"
    module = importlib.import_module(module_name)
    if isinstance(module.router, APIRouter):
        routers.append(module.router)