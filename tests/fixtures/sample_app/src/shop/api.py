import os

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status

from .models import Item
from .repo import ItemRepo, build_repo

DB_URL = os.getenv("SHOP_DB_URL", "sqlite://")

router = APIRouter(prefix="/items")
app = FastAPI()


def get_repo() -> ItemRepo:
    return build_repo()


@router.get("/{item_id}", response_model=Item)
def read_item(item_id: int, verbose: bool = False,
              limit: int = Query(10, le=100), repo: ItemRepo = Depends(get_repo)):
    """Fetch one item."""
    try:
        return repo.get(item_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="CANARY_DETAIL_55e0")


@router.post("/", status_code=status.HTTP_201_CREATED, tags=["items"])
async def create_item(item: Item) -> Item:
    return item


@app.get("/health")
def health() -> dict:
    token = os.environ["SHOP_SECRET"]
    return {"ok": bool(token)}


app.include_router(router, prefix="/v1")
