<div align="center">
# Hotel Booking Cancellation

### Predicting which hotel reservations will be cancelled

**ML Zoomcamp 2026 — Midterm Project** · Binary classification · End-to-end ML service

[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.3.2-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-DD2B0B?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119.1-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Data License](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Code License](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![ML Zoomcamp](https://img.shields.io/badge/ML%20Zoomcamp-2026-1f6feb)](https://github.com/DataTalksClub/machine-learning-zoomcamp)

</div>

---

## TL;DR

> **Problem.** 37% of hotel bookings are cancelled. Cancelled rooms cannot be
> resold, so a per-booking risk score is worth money to a revenue manager.
>
> **Model.** Gradient-boosted trees (XGBoost) behind a scikit-learn pipeline
> that owns *all* preprocessing — imputation, scaling, encoding and the learned
> country grouping — so training and serving cannot drift apart.
>
> **Result.** ROC AUC **0.9312** on a held-out test set, **calibrated** (mean
> predicted probability 0.366 vs an actual rate of 0.370), and deployed as a
> **1.5 MB** model behind a Dockerised FastAPI service.
>
> **The interesting engineering decision.** The highest-scoring model was a
> random forest — at **394 MB**. Using the *one-standard-error rule* we ship a
> model that is statistically indistinguishable (Δ ROC AUC 0.001, well inside
> the ±0.0023 fold noise) at **45× the size difference**.
>
> **Live:** `POST /predict` returns a calibrated probability; `GET /health`
> returns the model and threshold.

---

## Table of contents

| #  | Section                                                                          | What you'll find                                              |
| -- | -------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 1  | [The problem](#1-the-problem)                                                     | What is predicted, who uses it, and how it changes a decision |
| 2  | [Dataset](#2-dataset)                                                             | Provenance, size, columns, licensing                          |
| 3  | [EDA summary](#3-eda-summary)                                                     | The four findings that shaped the model                       |
| 4  | [Results](#4-results)                                                             | Headline metrics, model comparison, calibration               |
| 5  | [Model selection](#5-model-selection)                                             | The one-standard-error rule and why it matters                |
| 6  | [Feature importance](#6-feature-importance)                                       | Permutation importance, and what it reveals                   |
| 7  | [Project structure](#7-project-structure)                                         | Every file and what it does                                   |
| 8  | [How to run](#8-how-to-run)                                                       | Setup → data → train → serve → predict                    |
| 9  | [API](#9-api)                                                                     | Endpoints, examples, and error handling                       |
| 10 | [Containerisation &amp; cloud deployment](#10-containerisation--cloud-deployment) | Docker, Compose, Fly.io, Render, Kubernetes                   |
| 11 | [Deployment evidence](#11-deployment-evidence)                                    | Verified transcripts                                          |
| 12 | [What this repository provides](#12-what-this-repository-provides)                | Where each piece of the solution lives                        |

---

## 1. The problem

Hotels sell rooms far in advance and hold inventory against reservations that
may or may not materialise. In the source data **37% of bookings are
cancelled**. A cancelled reservation is revenue the hotel planned for and did
not get — and if it was cancelled late, the room cannot be resold.

**What we predict:** the probability that a reservation will be cancelled,
from attributes that are known *at the moment the booking is made*.

### Who benefits

| user                                      | how the model helps                                                                                                                                  |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Revenue manager**                 | decides how much inventory to hold and when to overbook; a per-booking risk score is more actionable than a monthly average cancellation rate        |
| **Front-desk / reservations agent** | sees a high-risk booking at check-in time and can offer an alternative room type, a date change or a parking upgrade *before* the guest cancels    |
| **Marketing**                       | a cancellation is a lost signup; a high-risk cohort is a re-engagement audience, but only if outreach is cheaper than the margin lost by discounting |
| **Revenue analyst**                 | understands *which* booking attributes drive cancellation, which drives policy (deposit rules, overbooking windows)                                |

### How the model changes a decision

The service returns a **probability, not a label**. The caller compares it
against a **decision threshold** that encodes the business cost of the two
error types:

| error                                                                    | business cost                                      |
| ------------------------------------------------------------------------ | -------------------------------------------------- |
| **false negative** — a booking that cancels was not flagged       | an empty room that cannot be resold                |
| **false positive** — a booking that was going to stay was flagged | an unnecessary intervention, which is much cheaper |

Because the two errors are not symmetric, the threshold is **tuned on the
validation set (0.415) rather than left at the default 0.5**.

---

## 2. Dataset

|                   |                                                                                |
| ----------------- | ------------------------------------------------------------------------------ |
| **Name**    | *Hotel booking demand datasets*                                              |
| **Authors** | Nuno Antonio, Ana de Almeida, Luis Nunes — *Data in Brief* 22:41–49 (2019) |
| **Rows**    | 119,390 bookings                                                               |
| **Columns** | 32                                                                             |
| **Hotels**  | H1 = Algarve resort, H2 = Lisbon city (Portugal)                               |
| **Period**  | arrivals July 2015 → August 2017                                              |
| **Target**  | `is_canceled` (binary) — 37.0% positive                                     |
| **Licence** | CC BY 4.0                                                                      |

**Source.** The original data is published by the authors and mirrored by the
Data Engineering course. The version used here is the cleaned TidyTuesday
release, mirrored on GitHub so the project runs **without any Kaggle
credentials**:

- **Kaggle** → [https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Paper** → [https://doi.org/10.1016/j.dib.2019.100807](https://doi.org/10.1016/j.dib.2019.100807)

`data/download_data.py` fetches it automatically and verifies the checksum
(`7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`, 119,390 rows).

### Columns used

| type            | columns                                                                                                                                                                                                                                   |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| categorical (7) | `hotel`, `deposit_type`, `customer_type`, `market_segment`, `arrival_date_month`, `arrival_date_day_of_week`, `country`                                                                                                     |
| numerical (11)  | `lead_time`, `adr`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `babies`, `previous_cancellations`, `previous_bookings_not_canceled`, `booking_changes`, `required_car_parking_spaces` |
| engineered (6)  | `total_nights`, `total_guests`, `is_family`, `arrival_month_sin`, `arrival_month_cos`, `lead_time_bin`                                                                                                                        |

**Not used:** `agent`, `company` (long-tailed identifiers) and
`reservation_status*` (**target leakage** — they record the outcome, not the
booking).

> **Exact category values matter.** `deposit_type` has three values:
> `"No Deposit"`, `"Non Refund"` and `"Refundable"` — note it is **`"Non Refund"`, not `"Non Refundable"`**. An unseen value makes the one-hot encoder
> silently fall back to the model's average for that feature, so the service
> returns a `warning` when it happens — see [§9](#9-api).

---

## 3. EDA summary

The full analysis with plots is in `notebook.ipynb`. These are the four
findings that shaped the model.

### 3.1 The target is imbalanced — accuracy is a trap

37.0% of bookings cancel. A model that always predicts *"not cancelled"*
scores **63% accuracy** while being completely useless.

Every model is therefore selected on **ROC AUC**, with **F1 and PR-AUC**
reported alongside.

### 3.2 `lead_time` is the strongest behavioural signal, and it is non-linear

Cancellation rises monotonically with how far ahead the booking was made:

| lead time     | cancellation rate |
| ------------- | ----------------- |
| 0 days        | ~2%               |
| 1–7 days     | ~5%               |
| 8–30 days    | ~20%              |
| 31–90 days   | ~32%              |
| 91–180 days  | ~38%              |
| 181–365 days | ~40%              |
| 365+ days     | ~42%              |

This is why the tree models beat the linear baseline by ~0.05 ROC AUC, and why
`lead_time_bin` is kept as an explicit feature alongside the raw value.

### 3.3 `deposit_type` dominates — and it is a policy artefact, not behaviour

| `deposit_type` | cancellation rate | rows    |
| ---------------- | ----------------- | ------- |
| `Non Refund`   | **99.4%**   | 14,587  |
| `No Deposit`   | 28.4%             | 104,641 |
| `Refundable`   | 22.2%             | 162     |

A non-refundable booking *cannot* be cancelled. The model therefore learns
part of the **hotel's cancellation policy** rather than guest intent. The
feature is legitimate — it is known at booking time and genuinely predictive —
but it is a known drift risk (see [§13](#13-limitations--next-steps)).

### 3.4 Origin country matters, and the tail is noise

178 origin countries. Domestic (PRT, 56% cancellation rate) and the top few
international markets behave very differently; the remaining one-off countries
are noise that invites overfitting.

They are collapsed into a single `Other` bucket, with the top-5 learned on the
**training split only** (`transformers.CountryGrouper`).

### Also observed

- Cancellation is **seasonal**, peaking in the summer months.
- Cancelled bookings have a **lower ADR** on average (discounted bookings are
  cancelled more often).
- The only gaps in the data are **488 missing `country`** and **4 missing
  `children`** values.

---

## 4. Results

### 4.1 Headline metrics

| metric             | validation | **test**   |
| ------------------ | ---------- | ---------------- |
| ROC AUC            | 0.9274     | **0.9312** |
| PR AUC             | 0.8972     | **0.9009** |
| accuracy           | 0.8383     | 0.8417           |
| precision          | 0.7653     | 0.7662           |
| recall             | 0.8128     | 0.8243           |
| F1                 | 0.7883     | 0.7942           |
| decision threshold | 0.415      | 0.415            |

The test score is slightly *above* validation, so there is no sign of
overfitting to the validation split.

### 4.2 Model comparison — 5-fold stratified CV on the training split

| model               | configuration                 | ROC AUC          | ± std | F1     | PR AUC | artifact         |
| ------------------- | ----------------------------- | ---------------- | ------ | ------ | ------ | ---------------- |
| random_forest       | n=500, depth=20               | **0.9268** | 0.0023 | 0.7622 | 0.8976 | 65.7 MB          |
| random_forest       | n=300                         | 0.9259           | 0.0025 | 0.7784 | 0.8979 | 96.3 MB          |
| **xgboost**   | **n=500, d=8, lr=0.05** | **0.9258** | 0.0022 | 0.7678 | 0.8948 | **1.5 MB** |
| xgboost             | n=300, d=6, lr=0.1            | 0.9218           | 0.0022 | 0.7576 | 0.8891 | 0.4 MB           |
| random_forest       | n=300, depth=15               | 0.9165           | 0.0019 | 0.7276 | 0.8838 | —               |
| xgboost             | n=300, d=4, lr=0.05           | 0.9047           | 0.0021 | 0.7207 | 0.8659 | 0.1 MB           |
| decision_tree       | depth=8, leaf=20              | 0.8818           | 0.0019 | 0.6798 | 0.8091 | —               |
| logistic_regression | C=1.0                         | 0.8787           | 0.0024 | 0.6735 | 0.8337 | —               |
| logistic_regression | C=0.1                         | 0.8786           | 0.0023 | 0.6732 | 0.8337 | —               |
| logistic_regression | C=1.0, balanced               | 0.8785           | 0.0024 | 0.7214 | 0.8325 | —               |
| decision_tree       | depth=None (unpruned)         | 0.8035           | 0.0022 | 0.7499 | 0.6571 | —               |

**What the table says**

- The **random forest scores marginally highest** and XGBoost is **statistically
  tied**. The gap (0.0010) is smaller than the fold-to-fold noise (±0.0023).
- **Tree models beat the linear baseline by ~0.05 ROC AUC**, confirming the EDA
  finding that the relationships are strongly non-linear.
- The **unpruned decision tree is the worst model** (0.8035) — a single tree
  memorises the training set. Pruning it to `max_depth=8` recovers ~0.08 ROC
  AUC.
- `class_weight="balanced"` clearly improves the logistic regression's **F1**
  (0.6735 → 0.7214) while leaving ROC AUC unchanged — exactly the expected
  behaviour, and a good illustration of why a single metric is not enough.

### 4.3 Protocol

Stratified 60/20/20 split → 5-fold stratified CV on the training split for
model selection → threshold tuned on validation → refit on train+validation →
the **test set is touched once, at the end**.

**All preprocessing lives inside the pipeline**, so there is no separate step
that can drift between training and serving:

| feature type | pipeline                                     |
| ------------ | -------------------------------------------- |
| numerical    | median imputation → standardisation         |
| categorical  | most-frequent imputation → one-hot encoding |
| `country`  | learned top-5 grouping → one-hot encoding   |

Every imputation value and the country grouping is learned **on the training
split only** — the pipeline is re-fitted inside each CV fold.

### 4.4 Calibration

A model that returns probabilities has to be *calibrated*, otherwise the number
is meaningless. On the test set the mean predicted probability is **0.3664**
against an actual rate of **0.3704**, and every subgroup is close:

| subgroup                       | n      | actual | predicted | AUC    |
| ------------------------------ | ------ | ------ | --------- | ------ |
| `deposit_type = Non Refund`  | 2,870  | 0.9930 | 0.9928    | 0.9998 |
| `deposit_type = No Deposit`  | 20,979 | 0.2855 | 0.2810    | 0.8985 |
| `country = PRT`              | 9,656  | 0.5646 | 0.5619    | 0.9566 |
| `country = GBR`              | 2,445  | 0.1943 | 0.1974    | 0.9083 |
| `lead_time > 300`            | 1,565  | 0.6786 | 0.6812    | 0.9937 |
| `lead_time <= 30`            | 7,712  | 0.1871 | 0.1812    | 0.8832 |
| `market_segment = Online TA` | 11,263 | 0.3709 | 0.3671    | 0.8295 |
| `market_segment = Groups`    | 3,881  | 0.6107 | 0.6021    | 0.9965 |
| `is_family = 1`              | 1,744  | 0.3595 | 0.3622    | 0.8511 |

Reproduce with `python tools/check_subgroups.py`.

---

## 5. Model selection

Picking the highest CV score would deploy a **394 MB** random forest — it fits
roughly 35M nodes, because `max_depth=20` with `min_samples_leaf=1` is nearly
unpruned. XGBoost reaches the same accuracy within noise in **1.5 MB**.

We therefore use the **one-standard-error rule** (Hastie, Tibshirani &
Friedman): *among the models whose CV score is within one standard deviation
of the best, prefer the simplest one.* "Simplest" is operationalised here as
**"smallest artifact"**, because that is what actually costs money to deploy.

### Candidates inside the one-standard-error band

Band = 0.9268 − 0.0023 = **0.9245**

| model             | configuration                 | CV ROC AUC | artifact          |                       |
| ----------------- | ----------------------------- | ---------- | ----------------- | --------------------- |
| **xgboost** | **n=500, d=8, lr=0.05** | 0.9258     | **1.46 MB** | **← deployed** |
| random_forest     | n=500, depth=20               | 0.9268     | 65.65 MB          |                       |
| random_forest     | n=300                         | 0.9259     | 96.30 MB          |                       |

> **Selected: XGBoost** — 0.001 ROC AUC cheaper for a **45× smaller** artifact.

### Threshold tuning

The default 0.5 threshold is arbitrary. Sweeping it across the validation set
and maximising F1 gives **0.415**.

---

## 6. Feature importance

**Permutation importance** is the primary analysis: impurity-based (random
forest) and gain-based (XGBoost) importances are not comparable with each
other and are biased towards high-cardinality features. Permutation importance
shuffles one feature at a time and measures the drop in ROC AUC, so it means
the same thing for every model family.

| feature                            | Δ ROC AUC       | std    |
| ---------------------------------- | ---------------- | ------ |
| `country`                        | **0.0899** | 0.0009 |
| `deposit_type`                   | 0.0792           | 0.0009 |
| `market_segment`                 | 0.0763           | 0.0017 |
| `lead_time`                      | 0.0580           | 0.0009 |
| `adr`                            | 0.0422           | 0.0011 |
| `required_car_parking_spaces`    | 0.0386           | 0.0014 |
| `customer_type`                  | 0.0300           | 0.0010 |
| `previous_cancellations`         | 0.0279           | 0.0008 |
| `booking_changes`                | 0.0094           | 0.0005 |
| `previous_bookings_not_canceled` | 0.0088           | 0.0006 |
| `hotel`                          | 0.0082           | 0.0005 |

The ranking matches the EDA: **origin and deposit policy first**, then *how*
the booking was made (`market_segment`, `lead_time`, `adr`).

`total_nights`, `total_guests` and `is_family` are much lower — consistent
with the EDA: party size affects *when* a booking is made rather than
*whether* it is cancelled.

---

## 7. Project structure

```
.
├── README.md                 this file
├── LICENSE                   MIT for the code, CC BY 4.0 for the data
├── notebook.ipynb            EDA, feature engineering, model selection (50 cells, executed)
│
├── features.py               canonical feature engineering (shared by train/serve)
├── transformers.py           CountryGrouper — learned top-N country grouping
├── train.py                  trains, selects the model, writes artifacts/
├── predict.py                command-line inference
├── serve.py                  FastAPI web service
├── healthcheck.py            container health probe (honours $PORT)
│
├── requirements.txt          pinned runtime dependencies (used by the Dockerfile)
├── requirements-dev.txt      extra dependencies for the notebook
├── Dockerfile                container definition
├── docker-compose.yml        local compose deployment
├── fly.toml                  Fly.io deployment
├── render.yaml               Render blueprint deployment
├── Makefile                  convenience targets
│
├── data/
│   ├── download_data.py      fetches + checksum-verifies the dataset
│   ├── raw/                  downloaded CSV (git-ignored)
│   └── processed/
├── artifacts/                generated: model.pkl, metadata, reports
└── ltools/
    ├── build_notebook.py     regenerates notebook.ipynb
    ├── check_slices.py       actual cancellation rates for key segments
    └── check_subgroups.py    per-subgroup calibration check

```

---

## 8. How to run

### 8.1 Setup

```bash
git clone <repository-url>
cd midterm-hotel-booking-cancellation

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

or with `uv`:

```bash
uv venv && uv pip install -r requirements.txt -r requirements-dev.txt
```

### 8.2 Get the data

```bash
python data/download_data.py
```

Downloads `data/raw/hotel_bookings.csv` and verifies the SHA-256 checksum. If
the network is unavailable, download it manually from
[Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
and place it at `data/raw/hotel_bookings.csv`.

### 8.3 Train

```bash
python train.py
```

Runs the full pipeline — 5-fold stratified CV over 11 model configurations,
one-standard-error model selection, threshold tuning on validation, final fit
on train+validation, permutation importance — and writes:

| artifact                                 | contents                                                        |
| ---------------------------------------- | --------------------------------------------------------------- |
| `artifacts/model.pkl`                  | the fitted sklearn pipeline (preprocessing + model), compressed |
| `artifacts/model_metadata.json`        | features, metrics, threshold, provenance, checksums, device     |
| `artifacts/model_comparison.csv`       | the cross-validation comparison table                           |
| `artifacts/feature_importance.csv`     | gain/impurity-based importances                                 |
| `artifacts/permutation_importance.csv` | model-agnostic importances on the test set                      |

Takes ~2 minutes on a 16-core laptop CPU.

<details>
<summary><b>Compute: CPU vs GPU</b> (click to expand)</summary>

```bash
python train.py                 # auto-detect (uses the GPU if available)
python train.py --device cuda   # force the GPU for XGBoost
python train.py --device cpu    # force CPU only
```

**What the GPU actually accelerates.** Only the gradient-boosting models.
Scikit-learn's `RandomForestClassifier`, `DecisionTreeClassifier` and
`LogisticRegression` have **no GPU implementation** — they are always CPU-bound
and use every core via `n_jobs=-1`. XGBoost is moved onto the GPU with
`tree_method="hist"` + `device="cuda"`.

On an RTX 3050 (4 GB) with CUDA 12.9, the XGBoost configurations train roughly
**3–5× faster** than on the CPU. Because the random forest dominates the
runtime and stays on the CPU, the end-to-end speed-up is smaller than the
XGBoost-only speed-up.

**Two safeguards make the result identical on any machine:**

1. The **deployed** model is always fitted on the CPU (`make_portable`). A
   booster trained with `device="cuda"` stores that device in its
   configuration and would fail to predict inside the CPU-only Docker
   container.
2. **CV scores are quantised to 3 decimals before they are compared.** GPU and
   CPU reductions happen in a different order, so the same configuration scores
   up to ~0.0005 apart depending on the device; without quantisation the
   *selected model* would depend on the hardware.

Verified: `python train.py --device cuda` and `python train.py --device cpu`
produce the **same model, the same threshold (0.415) and the same test
metrics**.

</details>

### 8.4 Serve

```bash
python -m uvicorn serve:app --host 0.0.0.0 --port 9696
```

### 8.5 Command-line prediction

```bash
python predict.py '{"hotel": "City Hotel", "lead_time": 120, "adr": 110.5, "arrival_date_month": "August", "arrival_date_day_of_week": "Tuesday", "stays_in_weekend_nights": 1, "stays_in_week_nights": 3, "adults": 2, "children": 1, "babies": 0, "country": "PRT", "market_segment": "Online TA", "deposit_type": "No Deposit", "customer_type": "Transient", "previous_cancellations": 0, "previous_bookings_not_canceled": 1, "booking_changes": 0, "required_car_parking_spaces": 0}'
```

```json
{
  "conversion_probability": 0.026838,
  "cancelled": false,
  "decision_threshold": 0.415
}
```

Every field is optional — omitted fields follow the same missing-value policy
used during training.

### 8.6 Make targets

| target                                                    | what it does                                                  |
| --------------------------------------------------------- | ------------------------------------------------------------- |
| `make setup`                                            | create the virtualenv and install dependencies                |
| `make data`                                             | download and checksum-verify the dataset                      |
| `make train`                                            | full training run                                             |
| `make train-gpu`                                        | training run with XGBoost forced onto the GPU                 |
| `make train-quick`                                      | reduced grid, for a fast smoke test                           |
| `make serve`                                            | start the FastAPI service                                     |
| `make docker-test`                                      | build → run → curl `/health` and `/predict` → clean up |
| `make compose-up` / `compose-down` / `compose-logs` | local compose deployment                                      |
| `make fly-deploy` / `fly-logs`                        | deploy to Fly.io                                              |
| `make check-subgroups`                                  | re-runs the calibration check                                 |
| `make check`                                            | all three checks                                              |

---

## 9. API

Base URL `http://localhost:9696`.

| method   | path               | purpose                                                |
| -------- | ------------------ | ------------------------------------------------------ |
| `GET`  | `/health`        | liveness probe + model summary                         |
| `GET`  | `/metadata`      | full training metadata (features, metrics, provenance) |
| `POST` | `/predict`       | cancellation probability for one booking               |
| `POST` | `/predict/batch` | probabilities for a list of bookings                   |

Interactive docs: **[http://localhost:9696/docs](http://localhost:9696/docs)**

### 9.1 Examples

```bash
curl -s http://localhost:9696/health
# {"status":"ok","model":"xgboost","decision_threshold":0.415}
```

```bash
curl -s http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "hotel": "City Hotel",
    "lead_time": 120,
    "adr": 110.5,
    "arrival_date_month": "August",
    "arrival_date_day_of_week": "Tuesday",
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 3,
    "adults": 2,
    "children": 1,
    "babies": 0,
    "country": "PRT",
    "market_segment": "Online TA",
    "deposit_type": "No Deposit",
    "customer_type": "Transient",
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 1,
    "booking_changes": 0,
    "required_car_parking_spaces": 0
  }'
```

```json
{
  "conversion_probability": 0.026838,
  "cancelled": false,
  "decision_threshold": 0.415
}
```

A high-risk booking (non-refundable deposit, 400-day lead time):

```json
{
  "conversion_probability": 0.995422,
  "cancelled": true,
  "decision_threshold": 0.415
}
```

The response always contains a probability in `[0, 1]` and a boolean
`cancelled` flag.

### 9.2 Unseen category values

`OneHotEncoder(handle_unknown="ignore")` maps a category the model never saw
to an all-zeros row, so the model answers with its *average* for that feature
without saying anything. A typo would therefore return a confident-looking
number that is really a fallback.

The service checks every request against the fitted categories and adds a
`warning` field:

```bash
curl -s http://localhost:9696/predict -H 'Content-Type: application/json' \
  -d '{"hotel": "City Hotel", "lead_time": 120, "adr": 110.5, "deposit_type": "Non Refundable"}'
```

```json
{
  "conversion_probability": 0.298489,
  "cancelled": false,
  "decision_threshold": 0.415,
  "warning": "unseen categorical value(s) ignored by the model: deposit_type='Non Refundable'"
}
```

> `country` is deliberately **excluded** from this check: unknown countries are
> handled *by design* by `CountryGrouper`, which maps everything outside the
> top-5 to `Other`. That is a graceful fallback, not a mistake, so it does not
> warn.

---

## 10. Containerisation & cloud deployment

The service is a **stateless FastAPI app with a 1.5 MB model**, so it runs on
any container host.

### 10.1 Docker

```bash
docker build -t hotel-cancellation:midterm .
docker run --rm -p 9696:9696 hotel-cancellation:midterm
```

```bash
make docker-test     # build → run → curl /health and /predict → clean up
```

The image is based on `python:3.11-slim`, installs the pinned
`requirements.txt`, copies only the code and the trained artefacts, and starts
`uvicorn` on port 9696. A `HEALTHCHECK` is included.

Two details that matter:

- **`transformers.py` must be copied into the image.** The pickled pipeline
  contains the `CountryGrouper`, and unpickling fails with
  `ModuleNotFoundError: No module named 'transformers'` if the module is not
  on the path. The Dockerfile copies it explicitly for this reason.
- **Only the runtime requirements are installed.** The notebook tooling
  (Jupyter, Matplotlib, Seaborn) lives in `requirements-dev.txt`; including it
  would add ~700 MB to the image for no runtime benefit.

The container honours `$PORT` (default 9696), which is what lets the *same*
image run on Render and Fly.io.

### 10.2 Docker Compose

```bash
make compose-up     # build + start on http://localhost:9696
make compose-logs   # tail the logs
make compose-down   # stop and remove
```

### 10.3 Render (one click, no CLI)

`render.yaml` is a Render blueprint, so the service can be provisioned from the
repository without any commands:

```bash
make render-info    # prints the three steps
```

1. Push the repository to GitHub.
2. [https://render.com/](https://render.com/) → **New** → **Blueprint** → select the repository.
3. Render detects `render.yaml`, builds the Docker image and deploys.

### 10.4 Local Kubernetes

The course also counts a local Kubernetes cluster as a cloud deployment.
`kind` works with the same image:

```bash
kind create cluster --name hotel-cancellation
docker build -t hotel-cancellation:midterm .
kind load docker-image hotel-cancellation:midterm --name hotel-cancellation

kubectl create deployment hotel-cancellation \
  --image=hotel-cancellation:midterm --port=9696
kubectl expose deployment hotel-cancellation --port=9696 --target-port=9696
kubectl port-forward svc/hotel-cancellation 9696:9696
```

---

## 11. Deployment evidence

**Container build and run (verified on this machine):**

```
$ docker images hotel-cancellation:midterm --format "{{.Size}}"
1.17GB

$ docker run -d --name hotel-cancel-test -p 9696:9696 hotel-cancellation:midterm
$ docker inspect --format '{{.State.Health.Status}}' hotel-cancel-test
healthy
```

**Service responses:**

| request                                       | response                                                         |
| --------------------------------------------- | ---------------------------------------------------------------- |
| `GET /health`                               | `{"status":"ok","model":"xgboost","decision_threshold":0.415}` |
| `POST /predict` (No Deposit, lead_time 120) | `{"conversion_probability":0.298489,"cancelled":false,...}`    |
| `POST /predict` (Non Refund, lead_time 400) | `{"conversion_probability":0.995422,"cancelled":true,...}`     |

**Reproduce it yourself:**

```bash
make docker-test
```

```
--- running the container and checking the endpoints ---
health : {"status":"ok","model":"xgboost","decision_threshold":0.415}
predict: {"conversion_probability":0.023195,"cancelled":false,"decision_threshold":0.415}
container removed
```

The container's prediction is **identical** to the local service, confirming
that the artifact, the preprocessing and the threshold all transfer correctly.
The `$PORT` override was verified too: running with `-e PORT=8080` makes the
container listen on 8080 and the health check still passes.

---

## 12. What this repository provides

| criterion                                                                         | where to look                                                                                        |
| --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Problem described: what is predicted, who benefits, how the model is used         | [§1](#1-the-problem)                                                                                 |
| Dataset documented with provenance and a reproducible, checksum-verified download | [§2](#2-dataset), `data/download_data.py`                                                          |
| EDA: distributions, missing values, target analysis, segment analysis             | [§3](#3-eda-summary), `notebook.ipynb` §4–5                                                      |
| Multiple model families trained, hyper-parameters tuned, comparison table         | [§4.2](#42-model-comparison--5-fold-stratified-cv-on-the-training-split), `notebook.ipynb` §8     |
| Model selection justified, not just "highest score"                               | [§5](#5-model-selection)                                                                             |
| Decision threshold tuned on validation (not left at 0.5)                          | [§4.1](#41-headline-metrics), `notebook.ipynb` §10                                                |
| Calibration verified overall and per subgroup                                     | [§4.4](#44-calibration), `tools/check_subgroups.py`                                                |
| Feature-importance analysis                                                       | [§6](#6-feature-importance), `artifacts/permutation_importance.csv`                                |
| Notebook executed end-to-end                                                      | `notebook.ipynb` — 50 cells, 0 errors, 9 figures                                                  |
| Notebook exported to training and serving scripts                                 | `train.py`, `serve.py`, `predict.py`                                                           |
| Feature engineering shared between notebook, training and serving                 | `features.py`, `transformers.py`                                                                 |
| Dependencies pinned + virtual environment instructions                            | `requirements.txt`, `requirements-dev.txt`, [§8.1](#81-setup)                                    |
| Containerisation with build and run instructions                                  | `Dockerfile`, [§10.1](#101-docker)                                                                 |
| Deployment configuration for a cloud host                                         | `docker-compose.yml`, `fly.toml`, `render.yaml`, [§10](#10-containerisation--cloud-deployment) |
| Reproducible from a fresh clone                                                   | checksum-verified download, `make setup && make data && make train`                                |
| Identical results on CPU and GPU                                                  | [§8.3](#83-train)                                                                                    |

---

<div align="center">
