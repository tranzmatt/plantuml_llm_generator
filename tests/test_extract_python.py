"""Python extractor: structure, resolution and the no-bodies guarantee."""

import json
import os

import pytest

from archdoc.extract import extract_python
from archdoc.extract.python_ast import collect_python_files, literal_text, module_id_for

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_app")

# Strings that appear only inside function bodies or comments of the fixture.
BODY_ONLY = [
    "CANARY_BODY_7f3a",
    "CANARY_COMMENT_19bd",
    "CANARY_SQL_a41c",
    "CANARY_DETAIL_55e0",
    "internal.example",
    "select 1",
]


@pytest.fixture(scope="module")
def ir():
    return extract_python(FIXTURE)


def _field(cls, name):
    return next(f for f in cls.fields if f.name == name)


# -- the core guarantee -------------------------------------------------------

def test_no_body_or_comment_text_in_ir(ir):
    blob = json.dumps(ir.to_dict())
    leaked = [c for c in BODY_ONLY if c in blob]
    assert leaked == []


def test_extraction_is_deterministic(ir):
    again = extract_python(FIXTURE)
    assert json.dumps(again.to_dict()) == json.dumps(ir.to_dict())


# -- modules ------------------------------------------------------------------

def test_src_layout_is_stripped_from_module_ids(ir):
    assert [m.id for m in ir.modules] == [
        "shop", "shop.api", "shop.cli", "shop.click_cli", "shop.models", "shop.repo", "shop.typer_cli",
    ]


def test_module_id_for():
    assert module_id_for("src/pkg/__init__.py", strip_src=True) == "pkg"
    assert module_id_for("src/pkg/mod.py", strip_src=False) == "src.pkg.mod"
    assert module_id_for("util/tool.py", strip_src=True) == "util.tool"
    assert module_id_for("__init__.py", strip_src=False, root_package="pip") == "pip"
    assert module_id_for("_internal/main.py", strip_src=False, root_package="pip") == "pip._internal.main"


def test_skip_dirs_and_excludes(tmp_path):
    for rel in ("a.py", "venv/x.py", "pkg.egg-info/y.py", "tests/test_a.py"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x = 1\n")
    assert collect_python_files(str(tmp_path)) == ["a.py", "tests/test_a.py"]
    assert collect_python_files(str(tmp_path), exclude=["tests/*"]) == ["a.py"]


def test_imports_follow_reexports(ir):
    # `from shop import Item` is recorded against the module that defines Item
    assert ir.module("shop.repo").imports == ["ext:requests", "shop.models"]


def test_main_guard_detected(ir):
    assert ir.module("shop.cli").has_main
    assert not ir.module("shop.api").has_main


def test_syntax_error_is_reported_not_raised(tmp_path):
    (tmp_path / "ok.py").write_text("def f(): pass\n")
    (tmp_path / "bad.py").write_text("def f(:\n")
    result = extract_python(str(tmp_path))
    assert result.module("bad").parse_error == "SyntaxError"
    assert result.find_function("ok:f") is not None


def test_caught_exceptions_do_not_reach_the_caller(tmp_path):
    (tmp_path / "m.py").write_text(
        "class AppError(Exception): pass\n"
        "class NotFound(AppError): pass\n"
        "def find(): raise NotFound()\n"
        "def look(): raise KeyError()\n"
        "def boom(): raise ValueError()\n"
        "def wrapped():\n"
        "    try:\n        find(); look()\n"
        "    except (AppError, LookupError):\n        return None\n"
        "def reraised():\n"
        "    try:\n        look()\n"
        "    except KeyError:\n        raise\n"
        "def partly():\n"
        "    try:\n        boom()\n    except Exception:\n        pass\n"
        "    boom()\n")
    ir = extract_python(str(tmp_path))
    assert ir.find_function("m:wrapped").caught == {"m:find": ["LookupError", "m:AppError"],
                                                    "m:look": ["LookupError", "m:AppError"]}
    assert ir.find_function("m:reraised").caught == {}
    assert ir.find_function("m:partly").caught == {}      # one call to boom is outside the try

    from archdoc.egress import build_function_payload
    from archdoc.policy import load_config
    from archdoc.pseudonym import AliasMap

    def offered(fid):
        return build_function_payload(ir, load_config(str(tmp_path)), "local", AliasMap(), fid).allowed["raises"]
    assert offered("m:wrapped") == []                    # NotFound via its base, KeyError via LookupError
    assert offered("m:reraised") == ["KeyError"] and offered("m:partly") == ["ValueError"]


# -- classes ------------------------------------------------------------------

def test_class_kinds(ir):
    kinds = {c.id: c.kind for c in ir.iter_classes()}
    assert kinds["shop.models:Item"] == "pydantic"
    assert kinds["shop.models:Order"] == "dataclass"
    assert kinds["shop.models:Status"] == "enum"
    assert kinds["shop.repo:ItemRepo"] == "class"


def test_pydantic_fields(ir):
    item = ir.find_class("shop.models:Item")
    assert [f.name for f in item.fields] == ["id", "name", "price", "status", "tags", "note"]
    assert _field(item, "id").required is True
    name = _field(item, "name")
    assert (name.required, name.constraints, name.docstring) == (True, {"max_length": "64"}, "Display name")
    price = _field(item, "price")
    assert (price.required, price.default, price.constraints) == (False, "0.0", {"ge": "0"})
    assert _field(item, "status").default == "Status.ACTIVE"
    assert _field(item, "status").type_id == "shop.models:Status"
    assert _field(item, "tags").default == "<factory>"
    assert _field(item, "note").docstring == "Free-text note."


def test_dataclass_and_enum(ir):
    order = ir.find_class("shop.models:Order")
    assert _field(order, "order_id").required is True
    assert _field(order, "items").type_id == "shop.models:Item"
    assert ir.find_class("shop.models:Status").enum_members == ["ACTIVE", "RETIRED"]


def test_instance_attributes_get_types(ir):
    repo = ir.find_class("shop.repo:ItemRepo")
    assert _field(repo, "db").type_id == "shop.repo:Database"
    assert _field(repo, "db").required is None


def test_bases_resolve_internally(ir):
    assert ir.find_class("shop.repo:CachedRepo").bases == ["shop.repo:ItemRepo"]


# -- calls --------------------------------------------------------------------

def test_call_through_typed_attribute(ir):
    f = ir.find_function("shop.repo:ItemRepo.get")
    assert f.calls == ["shop.repo:Database.query"]
    assert f.creates == ["shop.models:Item"]
    assert f.raises == ["KeyError"]


def test_super_call(ir):
    assert ir.find_function("shop.repo:CachedRepo.get").calls == ["shop.repo:ItemRepo.get"]


def test_module_level_instances_and_externals(ir):
    # source order: default_db.query(), session.post(), requests.get()
    assert ir.find_function("shop.repo:ping").calls == [
        "shop.repo:Database.query", "ext:requests.Session.post", "ext:requests.get",
    ]


def test_local_variable_types(ir):
    f = ir.find_function("shop.repo:build_repo")
    assert f.creates == ["shop.repo:CachedRepo"]
    assert f.calls == ["shop.repo:CachedRepo.get"]


def test_module_attribute_constructor(ir):
    assert ir.find_function("shop.repo:ItemRepo.make_order").creates == ["shop.models:Order"]


def test_return_annotation_drives_type(ir):
    # read_item's `repo` param is annotated ItemRepo
    assert "shop.repo:ItemRepo.get" in ir.find_function("shop.api:read_item").calls


# -- helpers ------------------------------------------------------------------

@pytest.mark.parametrize("src, expected", [
    ("1", "1"), ("-2.5", "-2.5"), ("'x'", "'x'"), ("None", "None"),
    ("[1, 'a']", "[1, 'a']"), ("{'k': 1}", "{'k': 1}"), ("Status.ACTIVE", "Status.ACTIVE"),
    ("compute()", "<expr>"), ("a + b", "<expr>"), ("[f()]", "<expr>"),
])
def test_literal_text(src, expected):
    import ast
    assert literal_text(ast.parse(src, mode="eval").body) == expected
