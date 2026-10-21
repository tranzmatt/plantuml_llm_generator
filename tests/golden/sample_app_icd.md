# sample_app Interface Control Document

| | |
|---|---|
| System | `sample_app` |
| Generated | archdoc 0.1.0 |
| Interfaces | 7 (6 provided, 1 required) |
| Operations | 11 |
| Data types | 2 |

> Tables in this document are generated from the source code and are
> authoritative for the version analysed. Items marked **TBD** cannot be
> determined from code and are listed in [Appendix A](#appendix-a-open-items).

## 1. Scope

**TBD**

## 2. Interface summary

| ID | Interface | Type | Direction | Owner | Protocol | Address | Ops |
|----|-----------|------|-----------|-------|----------|---------|----:|
| [1](#if-cli-shop-cli-main-parser) | `shopctl` | Command-line interface | provided | `shop.cli` | argv | `shopctl` | 3 |
| [2](#if-cli-shop-click-cli-admin) | `admin` | Command-line interface | provided | `shop.click_cli` | argv (click) | `admin` | 1 |
| [3](#if-cli-shop-click-cli-lone) | `standalone` | Command-line interface | provided | `shop.click_cli` | argv (click) | `standalone` | 1 |
| [4](#if-cli-shop-typer-cli-app) | `typer_cli.py` | Command-line interface | provided | `shop.typer_cli` | argv (typer) | `typer_cli.py` | 1 |
| [5](#if-env-sample-app) | `Environment` | Environment configuration | required | `sample_app` | process environment | — | 2 |
| [6](#if-http-shop-api-app) | `shop.api.app` | HTTP API | provided | `shop.api` | HTTP/JSON | `/` | 1 |
| [7](#if-http-shop-api-router) | `shop.api.router` | HTTP API | provided | `shop.api` | HTTP/JSON | `/v1/items` | 2 |

## 3. Interfaces

<a id="if-cli-shop-cli-main-parser"></a>
### 3.1 shopctl (Command-line interface)

| | |
|---|---|
| Direction | provided |
| Owner | `shop.cli` |
| Protocol | argv |
| Address | `shopctl` |
| Purpose | **TBD** |

<a id="if-cli-shop-cli-main-parser-shopctl"></a>
#### 3.1.1 `shopctl`

Shop control

- **Handler:** `shop.cli:main`

| Argument | Type | Required | Default | Constraints | Description |
|----------|------|----------|---------|-------------|-------------|
| `-v, --verbose` | `bool` | No | — | action=store_true | Chatty output |

<a id="if-cli-shop-cli-main-parser-ping"></a>
#### 3.1.2 `ping`

**TBD**

| Argument | Type | Required | Default | Constraints | Description |
|----------|------|----------|---------|-------------|-------------|
| `hosts` | `list[str]` | Yes | — | — | **TBD** |

<a id="if-cli-shop-cli-main-parser-serve"></a>
#### 3.1.3 `serve`

Run the server

- **Handler:** `shop.cli:cmd_serve`

| Argument | Type | Required | Default | Constraints | Description |
|----------|------|----------|---------|-------------|-------------|
| `--port` | `int` | No | `8000` | — | **TBD** |
| `--mode` | `str` | Yes | — | choices=['dev', 'prod'] | **TBD** |

<a id="if-cli-shop-click-cli-admin"></a>
### 3.2 admin (Command-line interface)

| | |
|---|---|
| Direction | provided |
| Owner | `shop.click_cli` |
| Protocol | argv (click) |
| Address | `admin` |
| Purpose | **TBD** |

<a id="if-cli-shop-click-cli-admin-seed-data"></a>
#### 3.2.1 `seed-data`

Seed the database.

- **Handler:** `shop.click_cli:seed_data`

| Argument | Type | Required | Default | Constraints | Description |
|----------|------|----------|---------|-------------|-------------|
| `--count, -c` | `int` | No | `1` | — | How many |
| `--dry-run` | `bool` | No | — | — | **TBD** |
| `name` | `str` | Yes | — | — | **TBD** |

<a id="if-cli-shop-click-cli-lone"></a>
### 3.3 standalone (Command-line interface)

| | |
|---|---|
| Direction | provided |
| Owner | `shop.click_cli` |
| Protocol | argv (click) |
| Address | `standalone` |
| Purpose | **TBD** |

<a id="if-cli-shop-click-cli-lone-standalone"></a>
#### 3.3.1 `standalone`

**TBD**

- **Handler:** `shop.click_cli:lone`

*No parameters.*

<a id="if-cli-shop-typer-cli-app"></a>
### 3.4 typer_cli.py (Command-line interface)

| | |
|---|---|
| Direction | provided |
| Owner | `shop.typer_cli` |
| Protocol | argv (typer) |
| Address | `typer_cli.py` |
| Purpose | **TBD** |

<a id="if-cli-shop-typer-cli-app-export-items"></a>
#### 3.4.1 `export-items`

Export items.

- **Handler:** `shop.typer_cli:export_items`

| Argument | Type | Required | Default | Constraints | Description |
|----------|------|----------|---------|-------------|-------------|
| `path` | `str` | Yes | — | — | **TBD** |
| `--fmt` | `str` | No | `'csv'` | — | Output format |
| `--force` | `bool` | No | `False` | — | **TBD** |

<a id="if-env-sample-app"></a>
### 3.5 Environment (Environment configuration)

| | |
|---|---|
| Direction | required |
| Owner | `sample_app` |
| Protocol | process environment |
| Purpose | **TBD** |

| Variable | Default | Required (no default) in | Used in | Description |
|----------|---------|--------------------------|---------|-------------|
| <a id="if-env-sample-app-shop-db-url"></a>`SHOP_DB_URL` | `'sqlite://'` | — | shop.api | **TBD** |
| <a id="if-env-sample-app-shop-secret"></a>`SHOP_SECRET` | — | shop.api | shop.api | **TBD** |

<a id="if-http-shop-api-app"></a>
### 3.6 shop.api.app (HTTP API)

| | |
|---|---|
| Direction | provided |
| Owner | `shop.api` |
| Protocol | HTTP/JSON |
| Address | `/` |
| Purpose | **TBD** |

<a id="if-http-shop-api-app-get-health"></a>
#### 3.6.1 `GET /health`

**TBD**

- **Handler:** `shop.api:health`
- **Response:** `dict`

*No parameters.*

<a id="if-http-shop-api-router"></a>
### 3.7 shop.api.router (HTTP API)

| | |
|---|---|
| Direction | provided |
| Owner | `shop.api` |
| Protocol | HTTP/JSON |
| Address | `/v1/items` |
| Purpose | **TBD** |

<a id="if-http-shop-api-router-get-item-id"></a>
#### 3.7.1 `GET /{item_id}`

Fetch one item.

- **Handler:** `shop.api:read_item`
- **Response:** [`Item`](#dt-shop-models-item)
- **Errors:** 404

| Parameter | In | Type | Required | Default | Constraints | Description |
|-----------|----|------|----------|---------|-------------|-------------|
| `item_id` | path | `int` | Yes | — | — | **TBD** |
| `verbose` | query | `bool` | No | `False` | — | **TBD** |
| `limit` | query | `int` | No | `10` | le=100 | **TBD** |

<a id="if-http-shop-api-router-post"></a>
#### 3.7.2 `POST /`

**TBD**

- **Handler:** `shop.api:create_item`
- **Request body:** [`Item`](#dt-shop-models-item)
- **Response:** [`Item`](#dt-shop-models-item) (status 201)
- **Tags:** items

*No parameters.*

## 4. Data dictionary

<a id="dt-shop-models-item"></a>
### 4.1 Item (pydantic)

A sellable item.

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `id` | `int` | Yes | — | — | **TBD** |
| `name` | `str` | Yes | — | max_length=64 | Display name |
| `price` | `float` | No | `0.0` | ge=0 | **TBD** |
| `status` | [`Status`](#dt-shop-models-status) | No | `Status.ACTIVE` | — | **TBD** |
| `tags` | `List[str]` | No | `<factory>` | — | **TBD** |
| `note` | `Optional[str]` | No | `None` | — | Free-text note. |

<a id="dt-shop-models-status"></a>
### 4.2 Status (enum)

**TBD**

Values: `ACTIVE`, `RETIRED`

<a id="appendix-a-open-items"></a>
## Appendix A. Open items

| # | Where | Open item |
|--:|-------|-----------|
| 1 | [shopctl](#if-cli-shop-cli-main-parser) | interface purpose |
| 2 | [shopctl](#if-cli-shop-cli-main-parser) | Exit codes |
| 3 | [shopctl](#if-cli-shop-cli-main-parser) | Required runtime environment |
| 4 | [shopctl / ping](#if-cli-shop-cli-main-parser-ping) | operation description |
| 5 | [shopctl / ping](#if-cli-shop-cli-main-parser-ping) | description of `hosts` |
| 6 | [shopctl / serve](#if-cli-shop-cli-main-parser-serve) | description of `port` |
| 7 | [shopctl / serve](#if-cli-shop-cli-main-parser-serve) | description of `mode` |
| 8 | [admin](#if-cli-shop-click-cli-admin) | interface purpose |
| 9 | [admin](#if-cli-shop-click-cli-admin) | Exit codes |
| 10 | [admin](#if-cli-shop-click-cli-admin) | Required runtime environment |
| 11 | [admin / seed-data](#if-cli-shop-click-cli-admin-seed-data) | description of `dry-run` |
| 12 | [admin / seed-data](#if-cli-shop-click-cli-admin-seed-data) | description of `name` |
| 13 | [standalone](#if-cli-shop-click-cli-lone) | interface purpose |
| 14 | [standalone](#if-cli-shop-click-cli-lone) | Exit codes |
| 15 | [standalone](#if-cli-shop-click-cli-lone) | Required runtime environment |
| 16 | [standalone / standalone](#if-cli-shop-click-cli-lone-standalone) | operation description |
| 17 | [typer_cli.py](#if-cli-shop-typer-cli-app) | interface purpose |
| 18 | [typer_cli.py](#if-cli-shop-typer-cli-app) | Exit codes |
| 19 | [typer_cli.py](#if-cli-shop-typer-cli-app) | Required runtime environment |
| 20 | [typer_cli.py / export-items](#if-cli-shop-typer-cli-app-export-items) | description of `path` |
| 21 | [typer_cli.py / export-items](#if-cli-shop-typer-cli-app-export-items) | description of `force` |
| 22 | [Environment](#if-env-sample-app) | interface purpose |
| 23 | [Environment](#if-env-sample-app) | Value formats and units |
| 24 | [Environment](#if-env-sample-app) | Which variables are secrets |
| 25 | [Environment / SHOP_DB_URL](#if-env-sample-app-shop-db-url) | operation description |
| 26 | [Environment / SHOP_SECRET](#if-env-sample-app-shop-secret) | operation description |
| 27 | [shop.api.app](#if-http-shop-api-app) | interface purpose |
| 28 | [shop.api.app](#if-http-shop-api-app) | Authentication / authorization |
| 29 | [shop.api.app](#if-http-shop-api-app) | Rate limits and timeouts |
| 30 | [shop.api.app](#if-http-shop-api-app) | API versioning policy |
| 31 | [shop.api.app / GET /health](#if-http-shop-api-app-get-health) | operation description |
| 32 | [shop.api.router](#if-http-shop-api-router) | interface purpose |
| 33 | [shop.api.router](#if-http-shop-api-router) | Authentication / authorization |
| 34 | [shop.api.router](#if-http-shop-api-router) | Rate limits and timeouts |
| 35 | [shop.api.router](#if-http-shop-api-router) | API versioning policy |
| 36 | [shop.api.router / GET /{item_id}](#if-http-shop-api-router-get-item-id) | description of `item_id` |
| 37 | [shop.api.router / GET /{item_id}](#if-http-shop-api-router-get-item-id) | description of `verbose` |
| 38 | [shop.api.router / GET /{item_id}](#if-http-shop-api-router-get-item-id) | description of `limit` |
| 39 | [shop.api.router / POST /](#if-http-shop-api-router-post) | operation description |
| 40 | [Item](#dt-shop-models-item) | description of field `id` |
| 41 | [Item](#dt-shop-models-item) | description of field `price` |
| 42 | [Item](#dt-shop-models-item) | description of field `status` |
| 43 | [Item](#dt-shop-models-item) | description of field `tags` |
| 44 | [Status](#dt-shop-models-status) | data type description |
