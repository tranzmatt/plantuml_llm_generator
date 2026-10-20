import requests

from shop import Item
from . import models


class Database:
    def query(self, sql: str) -> dict:
        return {}


default_db = Database()
session = requests.Session()


class ItemRepo:
    def __init__(self, db: Database):
        self.db = db
        self._cache = {}

    def get(self, item_id: int) -> Item:
        row = self.db.query("CANARY_SQL_a41c")
        if not row:
            raise KeyError(item_id)
        return Item(id=item_id, name=row["name"])

    def make_order(self) -> "models.Order":
        return models.Order(order_id="x")


class CachedRepo(ItemRepo):
    def get(self, item_id: int) -> Item:
        return super().get(item_id)


def ping() -> bool:
    default_db.query("select 1")
    session.post("https://internal.example/ping")
    return requests.get("https://internal.example").ok


def build_repo() -> ItemRepo:
    repo = CachedRepo(default_db)
    repo.get(1)
    return repo
