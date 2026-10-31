# 🛡️ SENTINEL — Real-Time Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Kafka](https://img.shields.io/badge/Apache_Kafka-9092-231F20?style=for-the-badge&logo=apachekafka&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**An end-to-end MLOps pipeline for real-time credit card fraud detection with Explainable AI**

[Live Demo](#-quick-start) · [API Docs](#-api-reference) · [Architecture](#-architecture)

</div>

---

## 📌 Problem Statement

Financial fraud causes **$485 billion** in annual losses worldwide. Traditional rule-based systems suffer from high false-positive rates, and black-box ML models aren't trusted by banks due to regulatory requirements (EU AI Act, GDPR).

**SENTINEL solves all three:**
- ✅ XGBoost model with reproducible metrics on a held-out test set
- ✅ SHAP-powered per-transaction explanations for **regulatory compliance**
- ✅ FastAPI prediction endpoints with real model-scored replay data

---

## 🏗️ Architecture

```
Raw CSV Data
    │
    ▼
EDA + SMOTE Balancing        ← Jupyter Notebook
    │
    ▼
XGBoost Model Training       ← Scikit-learn + XGBoost
    │
    ├──► SHAP Explainability  ← Per-prediction feature attribution
    │
    ▼
FastAPI REST API             ← /predict, /batch, /stream/live, /stats
    ▲
    │
Kafka Consumer               ← Scores live transactions, raises fraud alerts
    ▲
    │
Kafka Producer               ← Simulates 2 transactions/sec (5% fraud rate)
    │
    ▼
HTML/JS Dashboard            ← Live feed, SHAP charts, risk gauges (port 3000)
    │
    ▼
Docker Compose               ← All services containerized
```

---

## 📁 Project Structure

```
fraud_detection/
├── api/
│   └── main.py                     # FastAPI app — predict, batch, stream, stats, SHAP endpoints
├── frontend/
│   └── dashboard.html              # Vanilla JS + Chart.js dashboard (served by nginx)
├── stream/
│   ├── producer.py                 # Kafka producer — synthetic transaction stream
│   └── consumer.py                 # Kafka consumer — scores via API, logs fraud alerts
├── model/
│   ├── xgb_fraud_model.pkl         # Trained XGBoost classifier
│   ├── amount_scaler.pkl           # StandardScaler for transaction amount
│   ├── feature_names.pkl           # Ordered feature list (V1–V28 + Amount)
│   ├── test_sample.csv             # Standardized held-out transactions for replay
│   ├── test_sample_labels.csv      # Ground-truth labels, same row order as test_sample.csv
│   └── metrics.json                # Metrics measured on the held-out test sample
├── notebooks/
│   ├── 01_eda_and_training.ipynb   # Full EDA, SMOTE, training, SHAP analysis
│   └── (input data is read from ../data/creditcard.csv)
├── data/                           # EDA + evaluation plots (PNG)
├── utils.js                        # Shared JS helpers — API client, formatters, charts
├── docker-compose.yml              # Zookeeper + Kafka + API + Frontend
├── Dockerfile.api                  # Python 3.10-slim, uvicorn
├── Dockerfile.frontend             # Node build → nginx:alpine, port 3000
└── requirements.txt
```

---

## 📊 Model Performance

These values were recomputed from the shipped model, `test_sample.csv`, and its
matching labels. They match the notebook's last run. The 2,000-row held-out set
contains 40 fraud cases; at the 0.5 threshold the model flagged 46 transactions.
The confusion counts are TN 1,944, FP 16, FN 10, and TP 30. Accuracy alone can
hide fraud misses, so read it alongside precision and recall.

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.9752 |
| PR-AUC | 0.8164 |
| F1-Score (fraud) | 0.6977 |
| Precision (fraud) | 0.6522 |
| Recall (fraud) | 0.7500 |
| Accuracy | 0.9870 |
| API latency | Measured at runtime by `/stats/summary` |

The notebook uses deterministic synthetic data when `data/creditcard.csv` is
absent. The checked-in training sample has 10,000 rows with 200 fraud rows (2%).
If you replace it with another dataset and retrain, these metrics will change.

---

## 🔬 Key Techniques

| Component | Technique | Why |
|-----------|-----------|-----|
| Class Imbalance | SMOTE | Oversamples fraud in the training split only |
| Model | XGBoost | State-of-the-art on tabular fraud data |
| Explainability | SHAP TreeExplainer | Per-prediction, per-feature attribution |
| API | FastAPI + Pydantic | Type-safe, async, auto-documented |
| Streaming | Apache Kafka | Industry-standard message queue |
| Deployment | Docker Compose | Reproducible multi-service setup |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for Kafka / full stack)

### Option A — Run Locally (fastest)

**1. Create a virtual environment and install dependencies**
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**2. Start the FastAPI backend**
```bash
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/       (dashboard)
# → http://localhost:8000/docs   (Swagger UI)
```

> The pre-trained model is already included in `model/` — no retraining needed.

**3. Serve the frontend**
```bash
cd frontend
python -m http.server 3000
# → http://localhost:3000/dashboard.html
```

### Option B — Full Stack with Docker

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| FastAPI + Dashboard | http://localhost:8000 |
| Frontend (nginx) | http://localhost:3000 |
| Kafka Broker | localhost:9092 |

### Option C — Retrain the Model

```bash
# Optional: place creditcard.csv in data/ from Kaggle
# If absent, the notebook generates a deterministic 10,000-row, 2%-fraud sample
cd notebooks
jupyter notebook 01_eda_and_training.ipynb
# Run all cells (~2–3 min) → saves artifacts to model/
```

### Option D — Kafka Streaming (optional)

```bash
# Start Kafka
docker-compose up zookeeper kafka -d

# Terminal 1 — simulate 2 txn/sec (5% fraud rate)
python stream/producer.py

# Terminal 2 — consume, score via API, alert on fraud
python stream/consumer.py
```

---

## 🔌 API Reference

The API auto-serves the dashboard at `GET /` and exposes the following endpoints:

### `POST /predict` — Score a single transaction
```json
// Request
{
  "V1": -1.36, "V2": -0.07, "V3": 2.54, "...", "V28": -0.02,
  "Amount": 149.62
}

// Response
{
  "transaction_id": "...",
  "is_fraud": true,
  "fraud_probability": 0.9241,
  "risk_level": "HIGH",
  "amount": 149.62,
  "timestamp": "2024-01-01T12:00:00",
  "top_risk_factors": [ ... ],
  "demo_mode": false
}
```

### `POST /predict/batch` — Score multiple transactions
```json
// Response
{
  "total": 50,
  "fraud_count": 3,
  "fraud_rate": 0.06,
  "results": [ ... ]
}
```

### `GET /stream/live` — Rolling 20-row model-scored test-sample replay
The feed advances by 5 rows per request, so adjacent responses overlap. IDs are
stable (`TXN1000` through `TXN2999`) and wrap after the 2,000-row test sample.
Probabilities come from the loaded model; amounts are inverse-scaled for display.
Merchant and location labels are simulated. The feed is a replay, not an external
card-processing stream.
### `GET /stats/summary` — Dashboard KPIs (sample counts, fraud rate, amount saved, model metrics)
### `GET /shap/global` — Top 15 feature importances
### `GET /health` — Service health + model load status
### `GET /model/info` — Model type, feature count, feature names

> Full interactive docs available at **http://localhost:8000/docs**

---

## 🖥️ Dashboard Features

The API serves `frontend/dashboard.html` directly at `/`. It is a single-file
vanilla HTML/CSS/JS dashboard with no frontend build step. Its current navigation
uses the `#dash`, `#ana`, and `#mod` hashes.

- **Dashboard** — summary KPIs, fraud split, and a refreshable table of scored sample rows
- **Analyzer** — typical and suspicious presets, ordered value paste for V1–V28,
  and optional individual feature inputs; omitted features default to zero
- **Model** — held-out test metrics and performance diagnostics
- **API status** — reflects `/health`; the page shows a demo-mode state when the
  model is not loaded
- **Responsive layout** — full-width content with a mobile top navigation bar

---

### 🖥️ Dashboard Interface
![Dashboard](data/dashboard.png)

### 📊 Model Performance
![Performance](data/evaluation_plots.png)

### 🔍 SHAP Feature Importance
![SHAP](data/shap_beeswarm.png)

### 📈 Retrieval / Analysis
![Analysis](data/eda_correlations.png)

---
---

## 🧠 Demo Mode

If model files are missing, the API reports **demo mode**. Single and batch
predictions are marked `demo_mode: true` and are not real model output; the live
feed is empty and sample counts are zero. The summary reads persisted evaluation
metrics from `model/metrics.json` when that file is present; those scores describe
the saved test run, not the unavailable runtime model. SHAP importance is empty.

---
## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | Python 3.10, XGBoost 2.0, Scikit-learn, imbalanced-learn (SMOTE) |
| Explainability | SHAP 0.44 (TreeExplainer) |
| API | FastAPI 0.108, Uvicorn, Pydantic v2 |
| Streaming | Apache Kafka 7.5, kafka-python |
| Frontend | Vanilla JS, Chart.js 4.4, HTML5/CSS3 |
| Infra | Docker, Docker Compose, nginx:alpine |
| Notebooks | Jupyter, matplotlib, seaborn, plotly |

---



## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
