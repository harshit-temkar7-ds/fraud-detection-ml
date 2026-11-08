---
title: SENTINEL Fraud Detection
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
pinned: false
---
# SENTINEL — fraud detection (XGBoost + FastAPI + Kafka)

An end-to-end, **honestly evaluated** card-fraud scoring project: time-ordered training with CV model selection,
probability calibration, a cost-based decision threshold, a hardened scoring API with reason codes, an optional Kafka
replay pipeline, drift monitoring, tests and CI.

> **Read this first.** The repo ships with a model trained on **synthetic** data so everything runs offline.
> Numbers below are from that synthetic run and say nothing about real-world performance. Run
> `make data-kaggle && make train` to train on the real [ULB credit-card dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud);
> this section is rewritten automatically by `src/train.py`.

## Latest results (auto-generated from `model/metrics.json`)
<!-- METRICS:START -->

Model `20261107-184917` - strategy **scale_pos_weight**, calibration **sigmoid**,
decision threshold **0.0717** (cost-optimal: FN cost 111.58, FP cost 5.00).

Data: 284,807 rows / 492 frauds - time-ordered split train/val/test = 170,884 / 56,961 / 56,962 rows
(360 / 57 / 75 frauds). Source: user-provided dataset.

**Held-out test split, mean [95% bootstrap CI]**

| Metric | Value |
|---|---|
| PR-AUC (primary) | 0.790 [0.692, 0.870] |
| ROC-AUC | 0.981 [0.968, 0.992] |
| Recall at precision >= 80% | 0.760 [0.656, 0.848] |
| Precision @ threshold | 0.760 [0.662, 0.851] |
| Recall @ threshold | 0.760 [0.656, 0.848] |
| F1 @ threshold | 0.760 [0.681, 0.828] |
| Expected cost on test | 2,098 [1,236, 3,098] |

Confusion at threshold: TN 56,869, FP 18, FN 18, TP 57.
Cost baselines on test: flag nothing = 8,368; fixed 0.5 threshold = 2,140;
chosen threshold = 2,098.

**5-fold CV on the training split (model selection)**

| Strategy | PR-AUC (mean +- sd) | Recall @ precision |
|---|---|---|
| none | 0.876 +- 0.032 | 0.867 |
| scale_pos_weight (selected) | 0.878 +- 0.034 | 0.875 |
| smote | 0.866 +- 0.026 | 0.861 |

<!-- METRICS:END -->

Full details, limitations and failure modes: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md). Figures in `docs/figures/`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# (a) offline demo model is already in model/ -> just run the API + dashboard
export API_KEYS=dev-key-change-me                         # Windows PowerShell: $env:API_KEYS="dev-key-change-me"
uvicorn api.main:app --port 8000
# open http://localhost:8000  (the dashboard asks for the API key once)  |  API docs: http://localhost:8000/docs

# (b) train on the REAL data
pip install kaggle                                        # needs ~/.kaggle/kaggle.json
python scripts/download_data.py                           # -> data/creditcard.csv (validated: 284,807 rows / 492 frauds)
python -m src.train                                       # writes model/, docs/MODEL_CARD.md and the block above

# (c) offline demo training (synthetic data)
python -m src.train --synthetic
```

Docker: `docker compose up --build` (API + dashboard on :8000).
With Kafka replay: `docker compose --profile streaming up --build`.
(`.env.example` → `.env` to change the API key. Docker files were written carefully but **could not be executed in the
authoring sandbox**; CI builds the image and checks `/ready`.)

## What changed vs. the original project

| Area | Before | Now |
|---|---|---|
| Data | 10k-row Gaussian toy, planted shift | Real Kaggle data via a validated download script; harder, labelled-synthetic fallback for offline use |
| Evaluation | single 80/20 split, 40 test frauds, ROC-AUC headline | time-ordered train/val/test, 5-fold stratified CV for selection, PR-AUC primary, **bootstrap 95% CIs** |
| Imbalance | SMOTE before split-agnostic pipeline | `none` vs `scale_pos_weight` vs SMOTE (inside CV folds), chosen by CV PR-AUC on train only |
| Probabilities / threshold | fixed 0.5 / 0.4 / 0.7 | calibrated (isotonic/sigmoid, validation split) + **cost-optimal threshold**, risk bands derived from it |
| Artifacts | pickles | xgboost native JSON + JSON params, SHA-256 checked, version metadata, library-version check |
| API | zero-default fields, random demo scores, raw errors | required/range-checked fields (`extra=forbid`), batch ≤ 500, 503 when no model, API-key auth, rate limiting, request IDs, generic errors, optional SHAP (`?explain=true`, flagged only) |
| Explanations | random numbers in the dashboard | real TreeSHAP reason codes (honestly described as model attributions on anonymised features) |
| Streaming | hand-made fingerprint, crashes without SHAP, localhost | replays real held-out rows with true labels; env config, retries/backoff, dead-letter topic, running precision/recall |
| Dashboard | fake merchants/locations, invented threshold curve, random SHAP | labelled "replay of held-out data", real threshold sweep and explanations, synthetic-data banner |
| Docker | 3 broken Dockerfiles, `localhost` Kafka listener | one non-root image with HEALTHCHECK; compose with internal/external Kafka listeners and health-gated startup |
| Quality | none | 54 tests, ruff, GitHub Actions (lint, tests incl. training reproducibility, docker build), drift (PSI) endpoint |

## API

All endpoints except `/`, `/live`, `/ready`, `/health` need header `X-API-Key`.

| Endpoint | Purpose |
|---|---|
| `POST /predict?explain=false` | score one transaction (`V1..V28`, `Amount`, `Time` — all required). `explain=true` adds reason codes for flagged rows |
| `POST /predict/batch` | up to `MAX_BATCH_SIZE` (500) transactions |
| `GET /model/info`, `/stats/summary`, `/shap/global` | model metadata, evaluation metrics (+CIs), global importance |
| `GET /replay/feed`, `/replay/threshold?value=`, `/replay/explain` | dashboard data from the held-out replay sample (labelled; not live traffic) |
| `GET /monitoring/metrics`, `/monitoring/drift` | counts, flag rate, latency p50/p95/p99; PSI of `Amount` + top features vs. the training split |
| `GET /live`, `/ready`, `/health` | probes (503 if the model failed to load) |

```bash
curl -s -X POST localhost:8000/predict?explain=true -H "X-API-Key: dev-key-change-me" -H "Content-Type: application/json" \
  -d @example_transaction.json
```
(`example_transaction.json` is included.)

Environment variables: `API_KEYS`, `RATE_LIMIT_PER_MIN`, `MAX_BATCH_SIZE`, `ALLOWED_ORIGINS`, `MODEL_DIR`, `STRICT_VERSIONS`,
`DRIFT_WINDOW`, `DRIFT_MIN_SAMPLES`, `LOG_LEVEL`. Stream: `KAFKA_BROKER`, `KAFKA_TOPIC`, `KAFKA_DLQ_TOPIC`, `API_URL`, `API_KEY`,
`SEND_INTERVAL_SEC`, `SCORE_RETRIES`, `BACKOFF_BASE_SEC`.

## Kafka replay (without Docker)

```bash
# terminal 1: a broker on localhost:9092 (e.g. docker compose --profile streaming up kafka)
# terminal 2: API (see above)
# terminal 3:
API_KEY=dev-key-change-me python -m stream.consumer
# terminal 4:
python -m stream.producer
```
The consumer logs fraud alerts and a running TP/FP/FN/TN + precision/recall against the replayed true labels.

## Project layout

```
config/train.yaml      all training settings          src/        features, train, decision, metrics, drift, bundle
scripts/               download_data.py, make_synthetic.py   api/        FastAPI app (auth, limits, monitoring)
stream/                Kafka producer + consumer       frontend/   dashboard (served by the API at "/")
model/                 versioned artifacts + replay sample   tests/      pytest suite
docs/                  MODEL_CARD.md (generated) + figures   notebooks/  EDA only
```

## Limitations (also in the model card)
- Shipped metrics are synthetic. Real-data results will differ, and with only 492 frauds in the Kaggle set the CIs stay wide.
- Features are anonymised PCA components: reason codes explain the model, not real-world causes.
- Rate limiting and monitoring are in-memory (per process). Use a shared store for multi-replica deployments.
- No claim of GDPR / EU AI Act compliance is made; see the model card.
- Time-based generalisation is only tested over the ~2 days the Kaggle data spans.

## License
MIT (see `LICENSE`).
