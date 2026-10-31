import logging
import json
import os
import random
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import List

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="SENTINEL Fraud Detection API", version="1.0.0")

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins.split(",") if origin.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend", "dashboard.html")
MODEL_PATH = os.path.abspath(
    os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "model"))
)

model = scaler = feature_names = explainer = None
test_sample = None
test_probabilities = np.array([], dtype=float)
test_amounts = np.array([], dtype=float)
global_feature_importance = []
predict_latency_total_ms = 0.0
predict_latency_count = 0
latency_lock = threading.Lock()
live_offset = 0
live_offset_lock = threading.Lock()


def load_model():
    global model, scaler, feature_names, explainer
    global test_sample, test_probabilities, test_amounts, global_feature_importance
    try:
        model = joblib.load(os.path.join(MODEL_PATH, "xgb_fraud_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_PATH, "amount_scaler.pkl"))
        feature_names = joblib.load(os.path.join(MODEL_PATH, "feature_names.pkl"))
    except Exception:
        model = scaler = feature_names = explainer = None
        logger.exception("Model not loaded; API is running in demo mode")
        return

    try:
        # Construct this once and share it between prediction routes.
        explainer = shap.TreeExplainer(model)
        logger.info("Model and SHAP explainer loaded")
    except Exception:
        explainer = None
        logger.exception("SHAP explainer not available")

    sample_path = os.path.join(MODEL_PATH, "test_sample.csv")
    if not os.path.exists(sample_path):
        logger.warning("Test sample not found at %s", sample_path)
        return

    try:
        # test_sample.csv has already standardized Amount; do not scale it again.
        test_sample = pd.read_csv(sample_path)
        model_input = test_sample[feature_names] if feature_names else test_sample
        test_probabilities = model.predict_proba(model_input)[:, 1]
        test_amounts = scaler.inverse_transform(test_sample[["Amount"]]).ravel()

        if explainer is not None:
            shap_values = explainer.shap_values(model_input)
            shap_values = shap_matrix(shap_values)
            mean_absolute_values = np.abs(shap_values).mean(axis=0)
            ranked = sorted(
                zip(feature_names, mean_absolute_values),
                key=lambda item: item[1],
                reverse=True,
            )[:15]
            global_feature_importance = [
                {"feature": name, "importance": round(float(value), 6)}
                for name, value in ranked
            ]
        logger.info("Cached scores and SHAP importance for %d test rows", len(test_sample))
    except Exception:
        test_sample = None
        test_probabilities = np.array([], dtype=float)
        test_amounts = np.array([], dtype=float)
        global_feature_importance = []
        logger.exception("Could not score the shipped test sample")


def shap_matrix(values):
    if isinstance(values, list):
        values = values[-1]
    if hasattr(values, "values"):
        values = values.values
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def record_predict_latency(started_at: float):
    global predict_latency_total_ms, predict_latency_count
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    with latency_lock:
        predict_latency_total_ms += elapsed_ms
        predict_latency_count += 1


load_model()


class Transaction(BaseModel):
    V1: float = 0
    V2: float = 0
    V3: float = 0
    V4: float = 0
    V5: float = 0
    V6: float = 0
    V7: float = 0
    V8: float = 0
    V9: float = 0
    V10: float = 0
    V11: float = 0
    V12: float = 0
    V13: float = 0
    V14: float = 0
    V15: float = 0
    V16: float = 0
    V17: float = 0
    V18: float = 0
    V19: float = 0
    V20: float = 0
    V21: float = 0
    V22: float = 0
    V23: float = 0
    V24: float = 0
    V25: float = 0
    V26: float = 0
    V27: float = 0
    V28: float = 0
    Amount: float = 100.0


class BatchRequest(BaseModel):
    transactions: List[Transaction]


def risk_level(probability: float) -> str:
    return "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.4 else "LOW"


def top_factors(values, names):
    factors = [
        {
            "feature": name,
            "shap_value": round(float(value), 6),
            "impact": "increases_risk" if value > 0 else "decreases_risk",
        }
        for name, value in zip(names or [], values)
    ]
    return sorted(factors, key=lambda item: abs(item["shap_value"]), reverse=True)[:5]


def score_dataframe(df: pd.DataFrame):
    """Prepare API transactions and score them in one model call."""
    df = df.copy()
    df["Amount"] = scaler.transform(df[["Amount"]]).ravel()
    if feature_names:
        df = df[feature_names]
    probabilities = model.predict_proba(df)[:, 1]

    shap_rows = [[] for _ in range(len(df))]
    if explainer is not None:
        values = shap_matrix(explainer.shap_values(df))
        shap_rows = [top_factors(row, feature_names) for row in values]
    return probabilities, shap_rows


@app.get("/")
async def index():
    return FileResponse(FRONTEND)


@app.get("/health")
async def health():
    return {"status": "online", "model_loaded": model is not None, "timestamp": datetime.now().isoformat()}


@app.get("/model/info")
async def model_info():
    return {"model_type": "XGBoost", "features": len(feature_names) if feature_names else 29, "feature_names": feature_names or []}


@app.post("/predict")
def predict(t: Transaction):
    started_at = time.perf_counter()
    transaction_id = str(uuid.uuid4())
    if model is None:
        probability = random.uniform(0, 1)
        record_predict_latency(started_at)
        return {
            "transaction_id": transaction_id,
            "is_fraud": probability > 0.5,
            "fraud_probability": round(probability, 4),
            "risk_level": risk_level(probability),
            "amount": t.Amount,
            "timestamp": datetime.now().isoformat(),
            "demo_mode": True,
        }

    try:
        df = pd.DataFrame([t.model_dump()])
        probabilities, shap_rows = score_dataframe(df)
        probability = float(probabilities[0])
        record_predict_latency(started_at)
        return {
            "transaction_id": transaction_id,
            "is_fraud": probability > 0.5,
            "fraud_probability": round(probability, 4),
            "risk_level": risk_level(probability),
            "amount": t.Amount,
            "timestamp": datetime.now().isoformat(),
            "top_risk_factors": shap_rows[0],
            "demo_mode": False,
        }
    except Exception as exc:
        record_predict_latency(started_at)
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    transactions = req.transactions
    if model is None:
        results = []
        for transaction in transactions:
            probability = random.uniform(0, 1)
            results.append({
                "transaction_id": str(uuid.uuid4()),
                "is_fraud": probability > 0.5,
                "fraud_probability": round(probability, 4),
                "risk_level": risk_level(probability),
                "amount": transaction.Amount,
                "timestamp": datetime.now().isoformat(),
                "demo_mode": True,
            })
    else:
        try:
            df = pd.DataFrame([transaction.model_dump() for transaction in transactions])
            probabilities, shap_rows = score_dataframe(df) if transactions else ([], [])
            results = []
            for transaction, probability, factors in zip(transactions, probabilities, shap_rows):
                probability = float(probability)
                results.append({
                    "transaction_id": str(uuid.uuid4()),
                    "is_fraud": probability > 0.5,
                    "fraud_probability": round(probability, 4),
                    "risk_level": risk_level(probability),
                    "amount": transaction.Amount,
                    "timestamp": datetime.now().isoformat(),
                    "top_risk_factors": factors,
                    "demo_mode": False,
                })
        except Exception as exc:
            logger.exception("Batch prediction failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    fraud_count = sum(1 for result in results if result["is_fraud"])
    return {
        "total": len(results),
        "fraud_count": fraud_count,
        "fraud_rate": round(fraud_count / len(results), 4) if results else 0,
        "results": results,
    }


@app.get("/stream/live")
def live():
    global live_offset
    merchants = ["Amazon", "Walmart", "Shell", "Starbucks", "Netflix", "Uber", "Apple", "Zomato", "Swiggy", "Flipkart"]
    locations = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata", "Ahmedabad"]
    if model is None:
        return {"transactions": [], "count": 0, "demo_mode": True}
    if test_sample is None:
        return {"transactions": [], "count": 0, "demo_mode": False}

    now = datetime.now()
    txns = []
    sample_count = len(test_probabilities)
    if sample_count == 0:
        return {"transactions": [], "count": 0, "demo_mode": False}

    window_size = min(20, sample_count)
    with live_offset_lock:
        start = live_offset
        row_indices = [(start + i) % sample_count for i in range(window_size)]
        live_offset = (live_offset + min(5, sample_count)) % sample_count

    for window_position, sample_index in enumerate(row_indices):
        probability = float(test_probabilities[sample_index])
        amount = float(test_amounts[sample_index])
        txns.append({
            "id": f"TXN{1000 + sample_index}",
            "amount": round(amount, 2),
            "timestamp": (now - timedelta(seconds=window_position * 3)).isoformat(),
            "is_fraud": probability > 0.5,
            "fraud_probability": round(probability, 4),
            "risk_level": risk_level(probability),
            "merchant": random.choice(merchants),
            "location": random.choice(locations),
            "simulated": True,
        })
    return {"transactions": txns, "count": len(txns), "demo_mode": False}


@app.get("/stats/summary")
def stats():
    metrics_path = os.path.join(MODEL_PATH, "metrics.json")
    try:
        with open(metrics_path, "r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        source = "model/metrics.json"
    except (OSError, json.JSONDecodeError):
        metrics = {}
        source = "unavailable"
    if not isinstance(metrics, dict):
        metrics = {}
        source = "unavailable"

    if model is None:
        total = fraud = amount_saved = 0
    else:
        total = len(test_probabilities)
        fraud_mask = test_probabilities > 0.5
        fraud = int(fraud_mask.sum())
        amount_saved = round(float(test_amounts[fraud_mask].sum()), 2) if len(test_amounts) else 0.0

    def metric_value(name):
        try:
            return float(metrics.get(name, 0.0))
        except (TypeError, ValueError):
            return 0.0

    with latency_lock:
        average_latency = predict_latency_total_ms / predict_latency_count if predict_latency_count else 0.0

    return {
        "total_transactions": total,
        "fraud_detected": fraud,
        "fraud_rate": round(fraud / total, 4) if total else 0.0,
        "amount_saved": amount_saved,
        "avg_response_ms": round(average_latency, 3),
        "model_accuracy": metric_value("accuracy"),
        "precision": metric_value("precision"),
        "recall": metric_value("recall"),
        "f1_score": metric_value("f1"),
        "roc_auc": metric_value("roc_auc"),
        "source": source,
        "demo_mode": model is None,
    }


@app.get("/shap/global")
def shap_global():
    return {"feature_importance": global_feature_importance if model is not None else []}
