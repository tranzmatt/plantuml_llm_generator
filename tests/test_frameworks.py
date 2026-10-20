"""Framework detectors: the interface inventory."""

import os

import pytest

from archdoc.extract import extract_python

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_app")


@pytest.fixture(scope="module")
def ir():
    return extract_python(FIXTURE)


def _iface(ir, iface_id):
    return next(i for i in ir.interfaces if i.id == iface_id)


def _op(iface, name):
    return next(o for o in iface.operations if o.name == name)


def _params(op):
    return {p.name: p for p in op.params}


def test_inventory_ids(ir):
    assert [i.id for i in ir.interfaces] == [
        "if:cli:shop.cli:main:parser",
        "if:cli:shop.click_cli:admin",
        "if:cli:shop.click_cli:lone",
        "if:cli:shop.typer_cli:app",
        "if:env:sample_app",
        "if:http:shop.api:app",
        "if:http:shop.api:router",
    ]


def test_data_types(ir):
    assert [(d.id, d.source) for d in ir.data_types] == [
        ("shop.models:Item", "pydantic"),
        ("shop.models:Order", "dataclass"),
        ("shop.models:Status", "enum"),
    ]


# -- FastAPI ------------------------------------------------------------------

def test_router_prefix_combines_with_include_router(ir):
    router = _iface(ir, "if:http:shop.api:router")
    assert (router.kind, router.direction, router.address) == ("http", "provided", "/v1/items")
    assert [o.name for o in router.operations] == ["GET /{item_id}", "POST /"]


def test_get_route_details(ir):
    op = _op(_iface(ir, "if:http:shop.api:router"), "GET /{item_id}")
    assert op.handler == "shop.api:read_item"
    assert op.docstring == "Fetch one item."
    assert op.response.type_id == "shop.models:Item"
    assert op.errors == ["404"]
    params = _params(op)
    assert set(params) == {"item_id", "verbose", "limit"}       # Depends() is not a parameter
    assert (params["item_id"].required, params["item_id"].constraints) == (True, {"in": "path"})
    assert (params["verbose"].required, params["verbose"].default) == (False, "False")
    assert params["limit"].constraints == {"in": "query", "le": "100"}


def test_post_route_body_and_status(ir):
    op = _op(_iface(ir, "if:http:shop.api:router"), "POST /")
    assert op.request.type_id == "shop.models:Item"
    assert op.params == []
    assert op.meta == {"status_code": "201", "tags": "items"}


def test_app_routes_separate_from_router(ir):
    app = _iface(ir, "if:http:shop.api:app")
    assert app.address == "/"
    assert [o.name for o in app.operations] == ["GET /health"]


# -- argparse -----------------------------------------------------------------

def test_argparse_root_and_subcommands(ir):
    cli = _iface(ir, "if:cli:shop.cli:main:parser")
    assert (cli.name, cli.kind, cli.owner) == ("shopctl", "cli", "shop.cli")
    assert [o.name for o in cli.operations] == ["shopctl", "ping", "serve"]
    root = _op(cli, "shopctl")
    assert root.docstring == "Shop control"
    verbose = _params(root)["verbose"]
    assert (verbose.annotation, verbose.constraints["flags"], verbose.docstring) == (
        "bool", "-v, --verbose", "Chatty output")


def test_argparse_subcommand_args_and_handler(ir):
    serve = _op(_iface(ir, "if:cli:shop.cli:main:parser"), "serve")
    assert serve.handler == "shop.cli:cmd_serve"
    p = _params(serve)
    assert (p["port"].annotation, p["port"].default, p["port"].required) == ("int", "8000", False)
    assert (p["mode"].required, p["mode"].constraints["choices"]) == (True, "['dev', 'prod']")
    hosts = _params(_op(_iface(ir, "if:cli:shop.cli:main:parser"), "ping"))["hosts"]
    assert (hosts.annotation, hosts.required, hosts.constraints["kind"]) == ("list[str]", True, "positional")


# -- Click / Typer ------------------------------------------------------------

def test_click_group(ir):
    admin = _iface(ir, "if:cli:shop.click_cli:admin")
    assert admin.name == "admin"
    seed = _op(admin, "seed-data")
    p = _params(seed)
    assert (p["count"].annotation, p["count"].default, p["count"].docstring) == ("int", "1", "How many")
    assert p["count"].constraints["flags"] == "--count, -c"
    assert p["dry-run"].annotation == "bool"
    assert (p["name"].required, p["name"].constraints["kind"]) == (True, "positional")


def test_click_standalone_command(ir):
    assert [o.name for o in _iface(ir, "if:cli:shop.click_cli:lone").operations] == ["standalone"]


def test_typer_command(ir):
    op = _op(_iface(ir, "if:cli:shop.typer_cli:app"), "export-items")
    p = _params(op)
    assert (p["path"].required, p["path"].constraints) == (True, {"kind": "positional"})
    assert (p["fmt"].default, p["fmt"].docstring, p["fmt"].constraints["flags"]) == ("'csv'", "Output format", "--fmt")
    assert p["force"].default == "False"


# -- env ----------------------------------------------------------------------

def test_env_vars(ir):
    env = _iface(ir, "if:env:sample_app")
    assert env.direction == "required"
    ops = {o.name: o for o in env.operations}
    assert set(ops) == {"SHOP_DB_URL", "SHOP_SECRET"}
    assert ops["SHOP_DB_URL"].meta == {"used_in": "shop.api", "default": "'sqlite://'"}
