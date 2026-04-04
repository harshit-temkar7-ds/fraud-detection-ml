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
- ✅ XGBoost model with **ROC-AUC of 0.982** on the Kaggle Credit Card Fraud dataset
- ✅ SHAP-powered per-transaction explanations for **regulatory compliance**
- ✅ Sub-15ms API response time with **real-time Kafka streaming**

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
│   └── test_sample.csv             # Sample transactions for live stream demo
├── notebooks/
│   ├── 01_eda_and_training.ipynb   # Full EDA, SMOTE, training, SHAP analysis
│   └── creditcard.csv              # Kaggle dataset (place here if retraining)
├── data/                           # EDA + evaluation plots (PNG)
├── utils.js                        # Shared JS helpers — API client, formatters, charts
├── docker-compose.yml              # Zookeeper + Kafka + API + Frontend
├── Dockerfile.api                  # Python 3.10-slim, uvicorn
├── Dockerfile.frontend             # Node build → nginx:alpine, port 3000
└── requirements.txt
```

---

## 📊 Model Performance

| Metric | Score | Notes |
|--------|-------|-------|
| ROC-AUC | **0.982** | Overall discrimination ability |
| PR-AUC | **0.824** | Critical metric for imbalanced data |
| F1-Score | **0.87** | Fraud class precision/recall balance |
| Precision | **0.89** | Low false-positive rate |
| API Latency | **< 15ms** | Per-prediction response time |

> Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 0.17% fraud rate

---

## 🔬 Key Techniques

| Component | Technique | Why |
|-----------|-----------|-----|
| Class Imbalance | SMOTE | 0.17% fraud rate → synthetic oversampling |
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

**1. Extract and install dependencies**
```bash
unzip fraud_detection.zip && cd fraud_detection
pip install -r requirements.txt
```

**2. Start the FastAPI backend**
```bash
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs  (Swagger UI)
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
# Optional: place creditcard.csv in notebooks/ from Kaggle
# The notebook auto-generates synthetic data if skipped
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
  "is_fraud": true,
  "fraud_probability": 0.9241,
  "risk_level": "HIGH",
  "amount": 149.62,
  "timestamp": "2024-01-01T12:00:00"
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

### `GET /stream/live` — Latest 20 live transactions with risk scores
### `GET /stats/summary` — Dashboard KPIs (total txns, fraud rate, amount saved, model metrics)
### `GET /shap/global` — Top 15 feature importances
### `GET /health` — Service health + model load status
### `GET /model/info` — Model type, feature count, feature names

> Full interactive docs available at **http://localhost:8000/docs**

---

## 🖥️ Dashboard Features

The `frontend/dashboard.html` is a self-contained single-page app (vanilla JS + Chart.js):

- **Live Transaction Feed** — real-time table with fraud/legit badges and risk levels
- **KPI Cards** — total transactions, fraud detected, amount saved, avg latency
- **Hourly Chart** — 24-hour transaction volume vs. fraud count
- **SHAP Feature Importance** — global bar chart of top risk factors
- **Risk Gauge** — visual fraud probability meter per transaction

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

If the model files are missing (e.g., first run before training), the API automatically enters **demo mode** — returning realistic randomised predictions so the dashboard and streaming pipeline remain fully functional without the trained model.

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
