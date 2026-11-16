"""FastAPI web service for the hotel booking cancellation model.

Endpoints
---------
    GET  /health    -> liveness probe + artifact metadata
    GET  /metadata  -> the full training metadata document
    POST /predict   -> cancellation probability for a single booking
    POST /predict/batch -> cancellation probabilities for a list of bookings

Run locally:
    uvicorn serve:app --host 0.0.0.0 --port 9696
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features  # noqa: E402

ROOT = Path(__file__).parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

if not MODEL_PATH.exists():
    raise RuntimeError(
        "artifacts/model.pkl not found. Run `python train.py` before starting the service."
    )

model = joblib.load(MODEL_PATH)

metadata: dict[str, Any] = {}
if METADATA_PATH.exists():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

THRESHOLD = float(metadata.get("decision_threshold", 0.5))
app = FastAPI(
    title="Hotel Booking Cancellation",
    version="1.0.0",
    description=(
        "Predicts the probability that a hotel reservation will be cancelled. "
        "Trained on the UCI 'Hotel booking demand datasets' (119,390 bookings, "
        "July 2015 - August 2017)."
    ),
)


class Lead(BaseModel):
    """A single booking. Every field is optional: omitted fields follow the same
    missing-value policy used during training."""

    hotel: str | None = Field(None, description="Resort Hotel or City Hotel")
    lead_time: float | None = Field(None, description="Days between booking and arrival")
    arrival_date_month: str | None = Field(None, description="e.g. 'August'")
    arrival_date_day_of_week: str | None = Field(None, description="e.g. 'Tuesday'")
    stays_in_weekend_nights: float | None = None
    stays_in_week_nights: float | None = None
    adults: float | None = None
    children: float | None = None
    babies: float | None = None
    country: str | None = None
    market_segment: str | None = Field(None, description="e.g. 'Online TA'")
    deposit_type: str | None = Field(
        None,
        description="No Deposit / Non Refund / Refundable (exact values from the dataset)",
    )
    customer_type: str | None = Field(None, description="Transient / Contract / Group")
    previous_cancellations: float | None = None
    previous_bookings_not_canceled: float | None = None
    booking_changes: float | None = None
    required_car_parking_spaces: float | None = None
    adr: float | None = Field(None, description="Average daily rate")


def _known_categories() -> dict[str, set[str]]:
    """Categories the fitted pipeline was trained on, per categorical feature.

    The *input* column names are read from the fitted ``ColumnTransformer``
    (``transformers_`` holds ``(name, fitted_transformer, columns)``), because
    the one-hot encoder's own ``get_feature_names_out()`` returns the expanded
    output names such as ``deposit_type_No Deposit``.

    ``country`` is deliberately **not** included: unknown countries are handled
    by design by :class:`CountryGrouper`, which maps every country outside the
    top-5 to the ``Other`` bucket. That is a graceful fallback, not a mistake,
    so it must not raise a warning.
    """
    pre = model.named_steps["pre"]
    known: dict[str, set[str]] = {}

    for entry_name, transformer, columns in pre.transformers_:
        if entry_name == "cat":
            encoder = transformer.named_steps["onehot"]
            for column, categories in zip(list(columns), encoder.categories_):
                known[column] = set(categories)

    return known


KNOWN_CATEGORIES = _known_categories()


def _unknown_categories(record: dict[str, Any], known: dict[str, set[str]]) -> list[str]:
    """Categorical values that the model has never seen.

    ``OneHotEncoder(handle_unknown="ignore")`` silently maps an unseen value to
    an all-zeros row, so the model answers with its "average" for that feature
    without telling anyone. A typo in ``deposit_type`` would therefore return
    a confident-looking number that is really a fallback. Surfacing the
    mismatch makes that visible to the caller.
    """
    unknown = []
    for feature, allowed in known.items():
        value = record.get(feature)
        if value is None:
            continue
        value = str(value)
        if value not in allowed:
            unknown.append(f"{feature}={value!r}")
    return unknown


def _score(record: dict[str, Any]) -> dict[str, Any]:
    """Apply the shared feature engineering and score one booking."""
    frame = build_features(pd.DataFrame([record]))
    probability = float(model.predict_proba(frame)[0, 1])
    response = {
        "conversion_probability": round(probability, 6),
        "cancelled": bool(probability >= THRESHOLD),
        "decision_threshold": THRESHOLD,
    }
    unknown = _unknown_categories(record, KNOWN_CATEGORIES)
    if unknown:
        response["warning"] = (
            "unseen categorical value(s) ignored by the model: " + ", ".join(unknown)
        )
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe."""
    return {
        "status": "ok",
        "model": metadata.get("model", {}).get("family"),
        "decision_threshold": THRESHOLD,
    }


@app.get("/metadata")
def get_metadata() -> dict[str, Any]:
    """Training metadata: features, metrics, provenance."""
    return metadata


@app.post("/predict")
def predict(lead: Lead) -> dict[str, Any]:
    """Cancellation probability for a single booking."""
    try:
        return _score(lead.model_dump())
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/batch")
def predict_batch(leads: list[Lead]) -> list[dict[str, Any]]:
    """Cancellation probabilities for a list of bookings."""
    if not leads:
        raise HTTPException(status_code=400, detail="empty batch")
    try:
        frame = build_features(pd.DataFrame([lead.model_dump() for lead in leads]))
        probabilities = model.predict_proba(frame)[:, 1]
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return [
        {
            "conversion_probability": round(float(p), 6),
            "cancelled": bool(p >= THRESHOLD),
            "decision_threshold": THRESHOLD,
        }
        for p in probabilities
    ]


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("serve:app", host="0.0.0.0", port=9696)
