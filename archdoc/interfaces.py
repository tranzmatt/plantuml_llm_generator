"""
Interface inventory: the provided and required interfaces of a repo.

Shared by the component diagram (lollipop/socket notation) and the ICDs, so
the two always describe the same interfaces with the same ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .ir import Field


@dataclass
class TypeRef:
    display: str                       # as written, e.g. "list[Invoice]"
    type_id: Optional[str] = None      # DataType id when resolvable


@dataclass
class Operation:
    id: str
    name: str                          # "POST /invoices/{id}", "export", "INVOICE_DB_URL"
    params: List["Field"] = field(default_factory=list)   # path/query params, CLI options, message fields
    outputs: List["Field"] = field(default_factory=list)  # named outputs, e.g. the UI components a handler fills
    request: Optional[TypeRef] = None
    response: Optional[TypeRef] = None
    errors: List[str] = field(default_factory=list)       # status codes, exceptions, exit codes
    handler: Optional[str] = None      # Function id
    docstring: Optional[str] = None
    meta: Dict[str, str] = field(default_factory=dict)    # e.g. {"status_code": "201"}


@dataclass
class Interface:
    id: str                            # "if:http:pkg.api:router"
    name: str
    kind: str                          # http | cli | message | rpc | library | file | env | db
    direction: str                     # provided | required
    owner: str                         # module id that provides or consumes it
    protocol: Optional[str] = None
    address: Optional[str] = None      # base path, program name, topic
    operations: List[Operation] = field(default_factory=list)
    source_refs: List[str] = field(default_factory=list)


@dataclass
class DataType:
    id: str                            # same as the class id it came from
    name: str
    source: str                        # pydantic | dataclass | enum | proto | openapi | orm
    fields: List["Field"] = field(default_factory=list)
    docstring: Optional[str] = None
