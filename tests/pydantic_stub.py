import copy
import sys
import types


def install_pydantic_stub():
    try:
        import pydantic  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("pydantic")
    module.BaseModel = BaseModel
    module.Field = Field
    sys.modules["pydantic"] = module


def Field(default=None, default_factory=None):
    if default_factory is not None:
        return default_factory()
    return default


class BaseModel:
    def __init__(self, **data):
        for name in _model_fields(type(self)):
            if name in data:
                value = data[name]
            else:
                value = copy.deepcopy(getattr(type(self), name, None))
            setattr(self, name, value)
        for name, value in data.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def copy(self, deep=False):
        data = copy.deepcopy(self.__dict__) if deep else dict(self.__dict__)
        return type(self)(**data)


def _model_fields(model_type):
    fields = {}
    for cls in reversed(model_type.mro()):
        fields.update(getattr(cls, "__annotations__", {}))
    return fields
