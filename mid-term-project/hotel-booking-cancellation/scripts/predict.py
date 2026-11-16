"""Command-line inference for the hotel booking cancellation model.

Examples
--------
    # score a single booking given as JSON
    python predict.py '{"hotel": "City Hotel", "lead_time": 120, "adr": 110.5, ...}'

    # score a booking stored in a JSON file
    python predict.py --input booking.json

    # score every row of a CSV that contains the raw source columns
    python predict.py --csv data/raw/hotel_bookings.csv --limit 5

The script prints the cancellation probability and the boolean decision, using
the exact decision threshold that was tuned on the validation split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import TARGET, build_features  # noqa: E402

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts" / "model.pkl"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"


def load_model() -> tuple[object, float]:
    if not MODEL_PATH.exists():
        raise SystemExit(
            "artifacts/model.pkl not found. Run `python train.py` first."
        )
    model = joblib.load(MODEL_PATH)
    threshold = 0.5
    if METADATA_PATH.exists():
        threshold = float(
            json.loads(METADATA_PATH.read_text(encoding="utf-8")).get(
                "decision_threshold", 0.5
            )
        )
    return model, threshold


def score(frame: pd.DataFrame, model, threshold: float) -> list[dict]:
    features = build_features(frame)
    probabilities = model.predict_proba(features)[:, 1]
    out = []
    for p in probabilities:
        out.append(
            {
                "conversion_probability": round(float(p), 6),
                "cancelled": bool(p >= threshold),
                "decision_threshold": threshold,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", nargs="?", help="JSON object as a string")
    parser.add_argument("--input", type=Path, help="path to a JSON file")
    parser.add_argument("--csv", type=Path, help="path to a CSV of raw bookings")
    parser.add_argument("--limit", type=int, default=5, help="rows to show for --csv")
    args = parser.parse_args()

    if not (args.payload or args.input or args.csv):
        parser.error("provide a JSON payload, --input or --csv")

    model, threshold = load_model()

    if args.payload:
        record = json.loads(args.payload)
        print(json.dumps(score(pd.DataFrame([record]), model, threshold)[0], indent=2))
        return

    if args.input:
        record = json.loads(args.input.read_text(encoding="utf-8"))
        print(json.dumps(score(pd.DataFrame([record]), model, threshold)[0], indent=2))
        return

    frame = pd.read_csv(args.csv).head(args.limit)
    if TARGET in frame.columns:
        frame = frame.drop(columns=[TARGET])
    print(json.dumps(score(frame, model, threshold), indent=2))


if __name__ == "__main__":
    main()
