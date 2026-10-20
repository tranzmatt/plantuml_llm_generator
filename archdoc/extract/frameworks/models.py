"""Data types: pydantic models, dataclasses, enums and ORM models."""

from __future__ import annotations

import copy

from ...interfaces import DataType
from ...ir import Field
from ..python_ast import ExtractContext

DATA_KINDS = ("pydantic", "dataclass", "enum", "orm")


def detect_data_types(ctx: ExtractContext) -> None:
    for cls in ctx.ir.iter_classes():
        if cls.kind not in DATA_KINDS:
            continue
        if cls.kind == "enum":
            fields = [Field(name=m) for m in cls.enum_members]
        else:
            fields = [copy.deepcopy(f) for f in cls.fields if f.required is not None or cls.kind == "orm"]
        ctx.ir.data_types.append(DataType(
            id=cls.id, name=cls.name, source=cls.kind, fields=fields, docstring=cls.docstring,
        ))
