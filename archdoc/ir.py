"""
Intermediate representation (IR) of a repository's code structure.

The IR is the full, *real* local view of the repo: real names, real literals.
It never contains function bodies or comments. Sanitizing and pseudonymizing
for egress produce derived views of it (see docs/DESIGN.md sections 7-8).

Ids are stable strings:
  module    "pkg.mod"
  class     "pkg.mod:Class"
  function  "pkg.mod:func"  /  method "pkg.mod:Class.method"
  external  "ext:requests.get"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .interfaces import DataType, Interface

EXTERNAL_PREFIX = "ext:"


@dataclass
class Param:
    name: str
    annotation: Optional[str] = None
    # Literal source text for simple defaults ("0", "'x'", "None"), "<expr>" for
    # anything else, None when there is no default. Egress policy decides
    # whether the text ever leaves the machine.
    default: Optional[str] = None
    kind: str = "positional"     # positional | positional_only | keyword_only | var_positional | var_keyword


@dataclass
class Decorator:
    name: str                                         # dotted name, e.g. "app.get"
    args: List[str] = field(default_factory=list)     # simple literal args, "<expr>" otherwise
    kwargs: Dict[str, str] = field(default_factory=dict)


@dataclass
class Field:
    name: str
    annotation: Optional[str] = None
    required: Optional[bool] = None    # None when not meaningful (plain instance attributes)
    default: Optional[str] = None
    constraints: Dict[str, str] = field(default_factory=dict)   # pydantic Field(ge=0, max_length=..)
    docstring: Optional[str] = None
    type_id: Optional[str] = None      # resolved class id of the annotation, if internal


@dataclass
class Function:
    id: str
    name: str
    params: List[Param] = field(default_factory=list)
    returns: Optional[str] = None
    is_async: bool = False
    decorators: List[Decorator] = field(default_factory=list)
    visibility: str = "public"         # public | protected | private
    docstring: Optional[str] = None
    calls: List[str] = field(default_factory=list)     # resolved function ids or ext:..., source order
    creates: List[str] = field(default_factory=list)   # class ids instantiated, source order
    raises: List[str] = field(default_factory=list)    # class ids, ext:..., or builtin names


@dataclass
class Class:
    id: str
    name: str
    bases: List[str] = field(default_factory=list)     # resolved ids, ext:..., or raw names
    kind: str = "class"                # class | dataclass | enum | protocol | abc | pydantic | orm
    decorators: List[Decorator] = field(default_factory=list)
    fields: List[Field] = field(default_factory=list)
    methods: List[Function] = field(default_factory=list)
    enum_members: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class Module:
    id: str
    path: str                          # repo-relative, "/" separated
    docstring: Optional[str] = None
    imports: List[str] = field(default_factory=list)   # internal module ids or ext:...
    classes: List[Class] = field(default_factory=list)
    functions: List[Function] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)  # module-level assigned names (never values)
    has_main: bool = False             # contains `if __name__ == "__main__":`
    parse_error: Optional[str] = None


@dataclass
class Service:
    """One service of a deployment file. Environment values are never kept:
    `env` maps each variable name to where its value comes from."""
    name: str
    image: Optional[str] = None
    build: Optional[str] = None        # build context, repo-relative
    dockerfile: Optional[str] = None   # repo-relative, if found
    modules: List[str] = field(default_factory=list)    # module ids packaged into the image
    entry_module: Optional[str] = None                  # what the container runs
    ports: List[str] = field(default_factory=list)      # "host:container" as written
    depends_on: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)   # name -> "host" | "literal" | "unset"
    placement: List[str] = field(default_factory=list)  # e.g. "node.hostname == bio-host"


@dataclass
class Deployment:
    file: str                          # repo-relative
    kind: str = "docker-compose"
    services: List[Service] = field(default_factory=list)


@dataclass
class RepoIR:
    repo: str
    language: str = "python"
    modules: List[Module] = field(default_factory=list)
    interfaces: List[Interface] = field(default_factory=list)
    data_types: List[DataType] = field(default_factory=list)
    deployments: List[Deployment] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    # -- lookup helpers -----------------------------------------------------

    def module(self, module_id: str) -> Optional[Module]:
        return next((m for m in self.modules if m.id == module_id), None)

    def iter_classes(self):
        for m in self.modules:
            yield from m.classes

    def iter_functions(self):
        """All functions and methods."""
        for m in self.modules:
            yield from m.functions
            for c in m.classes:
                yield from c.methods

    def find_class(self, class_id: str) -> Optional[Class]:
        return next((c for c in self.iter_classes() if c.id == class_id), None)

    def find_function(self, function_id: str) -> Optional[Function]:
        return next((f for f in self.iter_functions() if f.id == function_id), None)
