import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from features import TARGET, build_features, feature_columns

df = build_features(pd.read_csv("data/raw/hotel_bookings.csv"))
X = df[feature_columns()]
y = df[TARGET].astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = joblib.load("artifacts/model.pkl")
proba = model.predict_proba(X_te)[:, 1]

te = X_te.copy()
te["y"] = y_te.values
te["p"] = proba

mask = (
    (te["hotel"] == "Resort Hotel")
    & (te["country"] == "GBR")
    & (te["market_segment"] == "Online TA")
    & (te["deposit_type"] == "No Deposit")
    & (te["lead_time"] > 300)
)
sl = te[mask]
print(f"subgroup n={len(sl)}")
print(f"  actual rate          {sl['y'].mean():.4f}")
print(f"  model mean prediction {sl['p'].mean():.4f}")
print(f"  model min / max       {sl['p'].min():.4f} / {sl['p'].max():.4f}")
print(f"  share of subgroup predicted < 0.1: {(sl['p'] < 0.1).mean():.3f}")

print()
print("=== model performance on a few subgroups (test set) ===")
groups = {
    "deposit_type = Non Refund": te["deposit_type"] == "Non Refund",
    "deposit_type = No Deposit": te["deposit_type"] == "No Deposit",
    "country = PRT": te["country"] == "PRT",
    "country = GBR": te["country"] == "GBR",
    "lead_time > 300": te["lead_time"] > 300,
    "lead_time <= 30": te["lead_time"] <= 30,
    "market_segment = Online TA": te["market_segment"] == "Online TA",
    "market_segment = Groups": te["market_segment"] == "Groups",
    "is_family = 1": te["is_family"] == 1,
}
from sklearn.metrics import roc_auc_score

rows = []
for name, m in groups.items():
    sub = te[m]
    if len(sub) < 50 or sub["y"].nunique() < 2:
        continue
    rows.append(
        {
            "subgroup": name,
            "n": len(sub),
            "actual": round(sub["y"].mean(), 4),
            "pred": round(sub["p"].mean(), 4),
            "gap": round(sub["y"].mean() - sub["p"].mean(), 4),
            "auc": round(roc_auc_score(sub["y"], sub["p"]), 4),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))
