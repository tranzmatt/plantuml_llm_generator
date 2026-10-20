"""Data models."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Status(str, Enum):
    ACTIVE = "active"
    RETIRED = "retired"


class Item(BaseModel):
    """A sellable item."""
    id: int
    name: str = Field(..., max_length=64, description="Display name")
    price: float = Field(0.0, ge=0)
    status: Status = Status.ACTIVE
    tags: List[str] = Field(default_factory=list)
    note: Optional[str] = None
    """Free-text note."""


@dataclass
class Order:
    order_id: str
    items: List[Item] = field(default_factory=list)

    def total(self) -> float:
        # CANARY_COMMENT_19bd
        secret_formula = "CANARY_BODY_7f3a"
        return sum(i.price for i in self.items) * len(secret_formula)
