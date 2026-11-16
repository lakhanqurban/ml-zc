"""Train the hotel booking cancellation classifier and export deployable artifacts.

Pipeline
--------
1. Load raw bookings and apply the shared feature engineering (``features.py``).
2. Stratified 60/20/20 train/validation/test split.
3. Cross-validate several model families with stratified 5-fold CV and a small
   hyper-parameter grid; select the best configuration by mean ROC AUC.
4. Fit the winner on train, pick the decision threshold on validation.
5. Refit on train+validation, report the untouched test score.
6. Export ``artifacts/model.pkl`` and ``artifacts/model_metadata.json``.

Usage
-----
    python train.py                 # full training run
    python train.py --quick         # fewer estimators, for a fast smoke test
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    build_features,
    feature_columns,
)
from transformers import CountryGrouper  # noqa: E402

# `country` gets its own transformer because it needs the top-N grouping.
OTHER_CATEGORICAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c != "country"]

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "raw" / "hotel_bookings.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

RANDOM_STATE = 42
N_SPLITS = 5


# --------------------------------------------------------------------------------------
# device selection
# --------------------------------------------------------------------------------------
def detect_device(requested: str = "auto") -> str:
    """Resolve which compute device to use for GPU-accelerated models.

    Only the gradient-boosting models (XGBoost) can use the GPU here. Scikit-learn's
    ``RandomForestClassifier``, ``DecisionTreeClassifier`` and
    ``LogisticRegression`` are CPU-only implementations, so they keep using every
    core via ``n_jobs=-1``.

    Parameters
    ----------
    requested : {"auto", "cpu", "cuda"}
        ``auto`` uses the GPU when XGBoost was built with CUDA support.
    """
    cuda_available = False
    try:
        build_info = xgb.build_info()
        cuda_available = bool(build_info.get("USE_CUDA")) and build_info.get(
            "CUDA_VERSION"
        ) is not None
    except Exception:  # pragma: no cover - depends on the local build
        cuda_available = False

    if requested == "cuda":
        if not cuda_available:
            raise SystemExit(
                "CUDA was requested but this XGBoost build has no CUDA support."
            )
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "cuda" if cuda_available else "cpu"


def describe_device(device: str) -> str:
    if device != "cuda":
        return "CPU (all cores) for scikit-learn models, CPU for XGBoost"
    try:
        info = xgb.build_info()
        cuda_version = ".".join(str(v) for v in info.get("CUDA_VERSION", []))
        return f"CUDA {cuda_version} for XGBoost, CPU (all cores) for scikit-learn models"
    except Exception:  # pragma: no cover
        return "CUDA for XGBoost, CPU for scikit-learn models"


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/download_data.py` first."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} bookings from {path}")
    return df


# --------------------------------------------------------------------------------------
# model definitions
# --------------------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Impute, scale and encode.

    * Numerical features: median imputation (learned on the training split
      only, inside every CV fold) followed by standardisation. Scaling is
      irrelevant for tree models but lets the logistic regression converge.
    * Categorical features: most-frequent imputation + one-hot encoding.
    * ``country``: the long tail of 178 country codes is collapsed into a
      single ``Other`` bucket by :class:`CountryGrouper`, which learns the
      top-5 countries from the training split only.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    country = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("group", CountryGrouper(n_top=5)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERICAL_FEATURES),
            ("cat", categorical, OTHER_CATEGORICAL_FEATURES),
            ("country", country, ["country"]),
        ]
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline([("pre", build_preprocessor()), ("clf", model)])


def make_xgb(device: str = "cpu", **overrides) -> XGBClassifier:
    """Build an XGBClassifier, putting it on the GPU when requested.

    ``tree_method="hist"`` is the GPU-capable algorithm; ``device="cuda"`` moves
    the histogram construction and the split search onto the GPU.
    """
    params: dict = {
        "n_estimators": 300,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "tree_method": "hist",
    }
    if device == "cuda":
        params["device"] = "cuda"
    params.update(overrides)
    return XGBClassifier(**params)


def candidate_models(quick: bool = False, device: str = "cpu") -> dict[str, list[tuple[str, object]]]:
    """Model families x hyper-parameter configurations to cross-validate."""
    if quick:
        return {
            "logistic_regression": [
                ("C=1.0", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE))
            ],
            "random_forest": [
                (
                    "n=50",
                    RandomForestClassifier(
                        n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1
                    ),
                )
            ],
            "xgboost": [
                (
                    "n=50,d=4",
                    make_xgb(device, n_estimators=50, max_depth=4, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.9),
                )
            ],
        }

    return {
        "logistic_regression": [
            ("C=1.0", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
            (
                "C=0.1",
                LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE),
            ),
            (
                "C=1.0,balanced",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        "decision_tree": [
            (
                "depth=None",
                DecisionTreeClassifier(random_state=RANDOM_STATE),
            ),
            (
                "depth=8,leaf=20",
                DecisionTreeClassifier(
                    max_depth=8, min_samples_leaf=20, random_state=RANDOM_STATE
                ),
            ),
        ],
        "random_forest": [
            (
                "n=300",
                RandomForestClassifier(
                    n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
            (
                "n=300,depth=15",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            (
                "n=500,depth=20",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=20,
                    min_samples_leaf=1,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ],
        "xgboost": [
            (
                "n=300,d=6,lr=0.1",
                make_xgb(device, n_estimators=300, max_depth=6, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=0.9),
            ),
            (
                "n=300,d=4,lr=0.05",
                make_xgb(device, n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9),
            ),
            (
                "n=500,d=8,lr=0.05",
                make_xgb(device, n_estimators=500, max_depth=8, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8),
            ),
        ],
    }


# --------------------------------------------------------------------------------------
# evaluation helpers
# --------------------------------------------------------------------------------------
def cross_validate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Stratified k-fold CV returning mean/std of the headline metrics."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    roc, f1, pr = [], [], []
    started = time.perf_counter()
    for train_idx, val_idx in skf.split(X, y):
        pipe = build_pipeline(clone(model))
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        with warnings.catch_warnings():
            # A CUDA-trained booster predicting on CPU input falls back to a
            # DMatrix. That is a correctness-preserving fallback, only slower.
            warnings.filterwarnings("ignore", message=".*mismatched devices.*")
            proba = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        roc.append(roc_auc_score(y.iloc[val_idx], proba))
        f1.append(f1_score(y.iloc[val_idx], (proba >= 0.5).astype(int), zero_division=0))
        pr.append(average_precision_score(y.iloc[val_idx], proba))
    elapsed = time.perf_counter() - started
    return {
        "roc_auc": float(np.mean(roc)),
        "roc_auc_std": float(np.std(roc)),
        "f1": float(np.mean(f1)),
        "pr_auc": float(np.mean(pr)),
        "seconds": float(elapsed),
    }


def select_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Pick the decision threshold that maximises F1 on the validation set."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        f1 = f1_score(y_true, (proba >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold, best_f1


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def artifact_size_mb(pipe: Pipeline) -> float:
    """Size of the compressed pickle, in megabytes."""
    buffer = io.BytesIO()
    joblib.dump(pipe, buffer, compress=3)
    return buffer.tell() / 1e6


def _quantised(score: float, digits: int = 3) -> float:
    """Round a CV score before it is used for a selection decision.

    GPU and CPU gradient boosting differ in their floating-point reduction
    order, so the same configuration scores up to ~0.0005 apart depending on the
    device. Comparing the raw scores would make the selected model depend on
    the hardware. Quantising to 3 decimals (the precision the coursework
    itself asks for) makes model selection reproducible on any machine.
    """
    return round(float(score), digits)


def select_final_model(
    candidates: list[dict], X: pd.DataFrame, y: pd.Series
) -> tuple[dict, list[dict]]:
    """Pick the model to deploy using the one-standard-error rule.

    The random forest with ``n=500, max_depth=20`` reaches the highest CV ROC
    AUC, but its artifact is ~394 MB (79 MB compressed) because it fits ~35M
    nodes. XGBoost with the same accuracy within noise produces a ~4 MB
    artifact.

    The one-standard-error rule (Hastie, Tibshirani & Friedman) says: among the
    models whose CV score is within one standard deviation of the best, prefer
    the *simplest* one. "Simplest" is operationalised here as "smallest
    artifact", because that is what actually costs money to deploy.

    Returns
    -------
    (chosen, qualified)
        ``chosen`` is the selected candidate; ``qualified`` is every candidate
        inside the one-standard-error band, annotated with its artifact size.
    """
    best_q = max(_quantised(c["scores"]["roc_auc"]) for c in candidates)
    best_std = next(
        c["scores"]["roc_auc_std"]
        for c in candidates
        if _quantised(c["scores"]["roc_auc"]) == best_q
    )
    threshold = best_q - best_std

    qualified = [
        c for c in candidates if _quantised(c["scores"]["roc_auc"]) >= threshold
    ]

    for candidate in qualified:
        pipe = build_pipeline(clone(candidate["model"]))
        pipe.fit(X, y)
        candidate["artifact_mb"] = artifact_size_mb(pipe)

    # smallest artifact among the statistically indistinguishable candidates
    chosen = min(
        qualified,
        key=lambda c: (c["artifact_mb"], -_quantised(c["scores"]["roc_auc"])),
    )
    return chosen, qualified


def make_portable(model):
    """Return a copy of the estimator that can run on CPU-only hardware.

    A booster trained with ``device="cuda"`` stores that device in its
    configuration, so the deployed artifact would fail to predict inside a
    CPU-only container. The final model is therefore always fitted on the CPU,
    which is also what the service actually runs on.
    """
    cloned = clone(model)
    if isinstance(cloned, XGBClassifier):
        cloned.set_params(device="cpu")
    return cloned


def permutation_importance_frame(
    pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 5
) -> pd.DataFrame:
    """Model-agnostic permutation importance on the test set.

    Impurity-based importance (random forest) and gain-based importance
    (XGBoost) are not directly comparable and are known to be biased towards
    high-cardinality features. Permutation importance measures the drop in
    ROC AUC when a feature is shuffled, so it means the same thing for every
    model family.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipe,
        X,
        y,
        n_repeats=n_repeats,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=1,  # a CUDA-trained booster cannot be sent to worker processes
    )
    frame = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return frame.reset_index(drop=True)


def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    """Mean impurity-based importance of the *original* features.

    One-hot encoded columns are summed back onto the feature they came from, so
    ``deposit_type=Non Refundable`` contributes to ``deposit_type`` rather than
    to a truncated name.
    """
    pre = pipe.named_steps["pre"]
    importances = np.asarray(pipe.named_steps["clf"].feature_importances_)

    names: list[str] = []
    for _, transformer, columns in pre.transformers_:
        columns = list(columns)
        if not hasattr(transformer, "named_steps"):
            continue
        if "onehot" in transformer.named_steps:
            encoder = transformer.named_steps["onehot"]
            for column, categories in zip(columns, encoder.categories_):
                names.extend([column] * len(categories))
        else:
            names.extend(columns)

    frame = pd.DataFrame({"feature": names, "importance": importances[: len(names)]})
    grouped = frame.groupby("feature", as_index=False)["importance"].sum()
    return grouped.sort_values("importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fast smoke-test run")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "compute device for the gradient-boosting models. 'auto' uses the GPU "
            "when XGBoost was built with CUDA support. Scikit-learn models are "
            "always CPU-bound and use all cores."
        ),
    )
    args = parser.parse_args()

    device = detect_device(args.device)
    print(f"compute device: {describe_device(device)}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_data()
    df = build_features(raw)
    df[TARGET] = df[TARGET].astype(int)

    features = feature_columns()
    X = df[features]
    y = df[TARGET]

    # --- stratified 60/20/20 split ------------------------------------------
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=RANDOM_STATE
    )
    print(f"split -> train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
    print(f"target balance: overall={y.mean():.3f}  train={y_train.mean():.3f}")

    # --- model selection ----------------------------------------------------
    print("\n=== Cross-validation (5-fold stratified) ===")
    results = []
    candidates = []

    for family, configs in candidate_models(quick=args.quick, device=device).items():
        for config, model in configs:
            scores = cross_validate(model, X_train, y_train)
            results.append(
                {
                    "model": family,
                    "config": config,
                    "roc_auc": round(scores["roc_auc"], 4),
                    "roc_auc_std": round(scores["roc_auc_std"], 4),
                    "f1": round(scores["f1"], 4),
                    "pr_auc": round(scores["pr_auc"], 4),
                    "seconds": round(scores["seconds"], 1),
                }
            )
            candidates.append(
                {"family": family, "config": config, "model": model, "scores": scores}
            )
            print(
                f"  {family:22s} {config:22s} "
                f"ROC AUC={scores['roc_auc']:.4f} (+/-{scores['roc_auc_std']:.4f}) "
                f"F1={scores['f1']:.4f}  PR AUC={scores['pr_auc']:.4f}  "
                f"[{scores['seconds']:.1f}s]"
            )

    results_frame = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print(f"\nBest CV ROC AUC: {results_frame.iloc[0]['roc_auc']:.4f} "
          f"({results_frame.iloc[0]['model']}, {results_frame.iloc[0]['config']})")

    # --- one-standard-error rule: smallest model that is statistically tied ---
    chosen, qualified = select_final_model(candidates, X_train, y_train)
    best_name = chosen["family"]
    best_config = chosen["config"]
    best_model = chosen["model"]
    best_scores = chosen["scores"]
    best_score = best_scores["roc_auc"]

    print("\n=== One-standard-error selection ===")
    print(f"  candidates within 1 std of the best ({len(qualified)}):")
    for candidate in sorted(qualified, key=lambda c: c["artifact_mb"]):
        marker = "  <-- selected" if candidate is chosen else ""
        print(
            f"    {candidate['family']:20s} {candidate['config']:20s} "
            f"ROC AUC={candidate['scores']['roc_auc']:.4f}  "
            f"artifact={candidate['artifact_mb']:7.2f} MB{marker}"
        )
    print(f"  selected: {best_name} ({best_config})")

    results_frame.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)

    # --- threshold tuning on validation -------------------------------------
    # The deployed model is fitted on the CPU so the artifact runs on
    # CPU-only hardware (e.g. the Docker container).
    portable_model = make_portable(best_model)
    pipe = build_pipeline(portable_model)
    pipe.fit(X_train, y_train)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    threshold, _ = select_threshold(y_val.values, val_proba)
    val_metrics = evaluate(y_val.values, val_proba, threshold)
    print(f"\nvalidation @ threshold={threshold:.2f}: {val_metrics}")

    # --- refit on train+val, score the untouched test set -------------------
    final_pipe = build_pipeline(make_portable(best_model))
    final_pipe.fit(X_train_val, y_train_val)
    test_proba = final_pipe.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test.values, test_proba, threshold)
    print(f"test      @ threshold={threshold:.2f}: {test_metrics}")

    # --- export artifacts ----------------------------------------------------
    joblib.dump(final_pipe, MODEL_PATH, compress=3)
    artifact_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"artifact size: {artifact_mb:.2f} MB")

    importances = feature_importance(final_pipe)
    importances.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)
    print("\ntop feature importances (gain/impurity based):")
    print(importances.head(10).to_string(index=False))

    print("\npermutation importance on the test set (model-agnostic)...")
    perm = permutation_importance_frame(final_pipe, X_test, y_test.values)
    perm.to_csv(ARTIFACTS_DIR / "permutation_importance.csv", index=False)
    print(perm.head(12).to_string(index=False))

    metadata = {
        "project": "hotel-booking-cancellation",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "file": "data/raw/hotel_bookings.csv",
            "rows": int(len(raw)),
            "sha256": "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06",
            "provenance": (
                "UCI/TidyTuesday clean version of 'Hotel booking demand datasets' "
                "(Antonio, Almeida & Nunes, 2019)"
            ),
        },
        "target": TARGET,
        "model": {
            "family": best_name,
            "config": best_config,
            "estimator": type(best_model).__name__,
            "params": {k: v for k, v in best_model.get_params().items()},
            "cv_roc_auc": round(float(best_score), 4),
            "cv_roc_auc_std": round(float(best_scores["roc_auc_std"]), 4),
            "cv_f1": round(float(best_scores["f1"]), 4),
            "cv_pr_auc": round(float(best_scores["pr_auc"]), 4),
            "artifact_mb": round(float(artifact_mb), 2),
            "selection_rule": (
                "one-standard-error rule: among candidates within one CV standard "
                "deviation of the best ROC AUC, the smallest artifact is deployed"
            ),
            "one_std_candidates": [
                {
                    "model": c["family"],
                    "config": c["config"],
                    "cv_roc_auc": round(float(c["scores"]["roc_auc"]), 4),
                    "artifact_mb": round(float(c["artifact_mb"]), 2),
                }
                for c in sorted(qualified, key=lambda c: c["artifact_mb"])
            ],
        },
        "decision_threshold": round(threshold, 4),
        "metrics": {
            "validation": {k: round(v, 4) for k, v in val_metrics.items()},
            "test": {k: round(v, 4) for k, v in test_metrics.items()},
        },
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numerical": NUMERICAL_FEATURES,
        },
        "missing_value_policy": {
            "categorical": "NA",
            "numerical": "median (learned on training split inside the pipeline)",
            "guest_counts": "0",
        },
        "random_state": RANDOM_STATE,
        "n_cv_folds": N_SPLITS,
        "compute_device": {
            "requested": args.device,
            "resolved": device,
            "description": describe_device(device),
            "note": (
                "scikit-learn estimators (random forest, decision tree, logistic "
                "regression) have no GPU implementation and always run on the CPU "
                "with all cores. Only XGBoost is CUDA-accelerated."
            ),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nsaved  -> {MODEL_PATH}")
    print(f"saved  -> {METADATA_PATH}")
    print(f"saved  -> {ARTIFACTS_DIR / 'model_comparison.csv'}")
    print(f"saved  -> {ARTIFACTS_DIR / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()
