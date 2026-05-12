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


class _FieldInfo:
    def __init__(self, default=None, default_factory=None, exclude=False):
        self.default = default
        self.default_factory = default_factory
        self.exclude = exclude

    def value(self):
        if self.default_factory is not None:
            return self.default_factory()
        return copy.deepcopy(self.default)


def Field(default=None, default_factory=None, **kwargs):
    return _FieldInfo(default=default, default_factory=default_factory, exclude=kwargs.get("exclude", False))


class BaseModel:
    def __init__(self, **data):
        for name in _model_fields(type(self)):
            if name in data:
                value = data[name]
            else:
                value = _field_default(type(self), name)
            setattr(self, name, value)
        for name, value in data.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def copy(self, deep=False):
        data = copy.deepcopy(self.__dict__) if deep else dict(self.__dict__)
        return type(self)(**data)

    @classmethod
    def model_validate(cls, value):
        if isinstance(value, cls):
            return value
        return cls(**value)

    def model_dump(self):
        return _dump_value(self)


def _model_fields(model_type):
    fields = {}
    for cls in reversed(model_type.mro()):
        fields.update(getattr(cls, "__annotations__", {}))
    return fields


def _field_default(model_type, name):
    for cls in model_type.mro():
        if name in cls.__dict__:
            value = cls.__dict__[name]
            if isinstance(value, _FieldInfo):
                return value.value()
            return copy.deepcopy(value)
    return None


def _field_info(model_type, name):
    for cls in model_type.mro():
        value = cls.__dict__.get(name)
        if isinstance(value, _FieldInfo):
            return value
    return None


def _dump_value(value):
    if isinstance(value, BaseModel):
        dumped = {}
        for name in _model_fields(type(value)):
            field_info = _field_info(type(value), name)
            if field_info and field_info.exclude:
                continue
            dumped[name] = _dump_value(getattr(value, name))
        return dumped
    if isinstance(value, list):
        return [_dump_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_dump_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _dump_value(item) for key, item in value.items()}
    return copy.deepcopy(value)
