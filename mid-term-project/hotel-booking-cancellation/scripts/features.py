"""Feature engineering for the hotel booking cancellation model.

This module is the single source of truth for data preparation. It is imported
by the notebook, ``train.py``, ``predict.py`` and ``serve.py`` so that the
exact same transformations are applied at training time and at inference time.

Missing-value policy (documented in the README):
    * categorical features  -> the string ``"NA"``
    * numerical features    -> left as ``NaN`` and imputed inside the sklearn
      pipeline (median, learned on the training split only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "is_canceled"

# --- raw columns we consume from the source CSV ------------------------------
RAW_CATEGORICAL_FEATURES: list[str] = [
    "hotel",
    "deposit_type",
    "customer_type",
    "market_segment",
    "arrival_date_month",
    "arrival_date_day_of_week",
    "country",
]

RAW_NUMERICAL_FEATURES: list[str] = [
    "lead_time",
    "adr",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "required_car_parking_spaces",
]

# --- engineered columns -------------------------------------------------------
ENGINEERED_CATEGORICAL_FEATURES: list[str] = ["lead_time_bin"]

ENGINEERED_NUMERICAL_FEATURES: list[str] = [
    "total_nights",
    "total_guests",
    "is_family",
    "arrival_month_sin",
    "arrival_month_cos",
]

CATEGORICAL_FEATURES: list[str] = RAW_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
NUMERICAL_FEATURES: list[str] = RAW_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

MONTH_TO_NUM: dict[str, int] = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

# Lead time is the strongest single predictor, but the relationship is strongly
# non-linear, so we also expose it as an ordinal-ish categorical bucket.
LEAD_TIME_BIN_EDGES: list[tuple[float, float]] = [
    (0, 0),
    (1, 7),
    (8, 30),
    (31, 90),
    (91, np.inf),
]
LEAD_TIME_BIN_LABELS: list[str] = ["0", "1-7", "8-30", "31-90", "91+"]


def _lead_time_bin(value: float) -> str:
    """Bucket the lead time into a categorical bucket."""
    if pd.isna(value):
        return "unknown"
    for (low, high), label in zip(LEAD_TIME_BIN_EDGES, LEAD_TIME_BIN_LABELS):
        if low <= value <= high:
            return label
    return LEAD_TIME_BIN_LABELS[-1]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing source column as NaN.

    Makes the function safe for partial inference payloads: a request that omits
    a field is treated exactly like a missing value in the raw data.
    """
    df = df.copy()
    for col in RAW_CATEGORICAL_FEATURES + RAW_NUMERICAL_FEATURES + [TARGET]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw bookings and add engineered features.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw bookings. May be a single row or the full dataset. May contain a
        subset of the source columns.

    Returns
    -------
    pd.DataFrame
        Frame containing every column in :func:`feature_columns`.
    """
    df = _ensure_columns(raw)

    # --- missing-value policy: categorical -> "NA" -------------------------
    for col in RAW_CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("NA").astype(str)

    # --- missing-value policy: numerical -> 0 for guest counts -------------
    # children/babies are counts of people; an unknown value is treated as 0.
    # All other numerical gaps stay NaN and are imputed by the pipeline.
    for col in ("children", "babies"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in RAW_NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- engineered features ----------------------------------------------
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["total_guests"] >= 2) & (df["children"] >= 1)).astype(int)

    # Cyclical encoding of the arrival month: December and January are adjacent
    # on a calendar but 11 and 1 apart as plain integers.
    month_num = df["arrival_date_month"].map(MONTH_TO_NUM).fillna(0).astype(float)
    df["arrival_month_sin"] = np.sin(2 * np.pi * (month_num - 1) / 12)
    df["arrival_month_cos"] = np.cos(2 * np.pi * (month_num - 1) / 12)

    df["lead_time_bin"] = df["lead_time"].map(_lead_time_bin).astype(str)

    return df


def feature_columns() -> list[str]:
    """Return the ordered list of model input columns."""
    return CATEGORICAL_FEATURES + NUMERICAL_FEATURES
