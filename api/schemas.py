from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from .settings import MAX_BATCH_SIZE

_V_FIELDS = {
    f"V{i}": (float, Field(..., ge=-200, le=200, allow_inf_nan=False,
                           description="Anonymised PCA component"))
    for i in range(1, 29)
}

_EXAMPLE = {**{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 49.9, "Time": 40000.0}

Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"example": _EXAMPLE}),
    **_V_FIELDS,
    Amount=(float, Field(..., ge=0, le=1_000_000, allow_inf_nan=False, description="Transaction amount")),
    Time=(float, Field(..., ge=0, le=1e9, allow_inf_nan=False,
                       description="Seconds since the dataset's first transaction (only time-of-day is used)")),
)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: List[Transaction] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class Reason(BaseModel):
    code: str
    feature: str
    direction: Optional[str] = None
    contribution: float
    description: str


class Prediction(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    threshold: float
    amount: float
    timestamp: str
    model_version: str
    reasons: Optional[List[Reason]] = None


class BatchResponse(BaseModel):
    total: int
    fraud_count: int
    fraud_rate: float
    model_version: str
    reasons_note: Optional[str] = None
    results: List[Prediction]
