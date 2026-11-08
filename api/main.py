"""SENTINEL scoring API (create_app factory).

Run:  API_KEYS=dev-key-change-me uvicorn api.main:app --port 8000
"""

import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src import __version__
from src.decision import risk_level
from src.features import RAW_FEATURES, make_features
from src.model_bundle import ArtifactError, ModelBundle
from src.reasons import DISCLAIMER, build_reasons

from .monitoring import Monitor
from .schemas import BatchRequest, BatchResponse, Prediction, Transaction
from .security import AccessDep, SlidingWindowLimiter, configure_logging, new_request_id, request_id_var
from .settings import Settings

logger = logging.getLogger("sentinel.api")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class State:
    """Everything loaded at startup lives here (no module-level globals)."""

    def __init__(self):
        self.bundle = None
        self.load_error = None
        self.metrics = {}
        self.reference = {}
        self.replay = None            # DataFrame incl. Class
        self.replay_proba = np.array([])
        self.replay_importance = []
        self.replay_offset = 0
        self.explain_offset = 0
        self.replay_lock = threading.Lock()
        self.monitor = None


def load_state(settings: Settings) -> State:
    st = State()
    d = settings.model_dir
    try:
        st.bundle = ModelBundle.load(d, strict_versions=settings.strict_versions)
    except (ArtifactError, OSError, ValueError) as exc:
        st.load_error = str(exc)
        logger.error("Model not loaded: %s", exc)
        st.monitor = Monitor({}, settings.drift_window, settings.drift_min_samples, settings.log_every)
        return st

    for name, attr in (("metrics.json", "metrics"), ("reference_stats.json", "reference")):
        try:
            setattr(st, attr, json.loads((d / name).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            logger.warning("%s unavailable", name)
    st.monitor = Monitor(st.reference.get("psi_features", {}), settings.drift_window,
                         settings.drift_min_samples, settings.log_every)

    replay_path = d / "replay_sample.csv"
    if replay_path.exists():
        try:
            df = pd.read_csv(replay_path)
            st.replay = df.reset_index(drop=True)
            st.replay_proba = st.bundle.predict_proba(df[RAW_FEATURES])
            _, contrib = st.bundle.contributions(df[RAW_FEATURES])
            mean_abs = np.abs(contrib).mean(axis=0)
            ranked = sorted(zip(st.bundle.feature_names, mean_abs), key=lambda kv: kv[1], reverse=True)[:15]
            st.replay_importance = [{"feature": n, "importance": round(float(v), 6)} for n, v in ranked]
        except Exception:
            logger.exception("Could not score replay sample")
            st.replay = None
    logger.info("Loaded model %s (threshold %.4f)", st.bundle.version, st.bundle.threshold)
    return st


def create_app(settings: Settings = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.sentinel = load_state(settings)
        yield

    app = FastAPI(title="SENTINEL Fraud Detection API", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.limiter = SlidingWindowLimiter(settings.rate_limit_per_min)
    app.state.sentinel = State()
    app.state.sentinel.monitor = Monitor({}, 1, 1, 1)  # replaced in lifespan

    app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins),
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    # ------------------------------------------------------------ middleware
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = new_request_id(request.headers.get("x-request-id"))
        token = request_id_var.set(rid)
        request.state.request_id = rid
        started = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code,
                        (time.perf_counter() - started) * 1000)
            return response
        finally:
            request_id_var.reset(token)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Never echo the submitted values back (they may be NaN/huge/sensitive); report location + message only.
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "-")
        logger.error("Unhandled error on %s %s [%s]", request.method, request.url.path, rid, exc_info=exc)
        return JSONResponse({"detail": "Internal server error", "request_id": rid}, status_code=500,
                            headers={"X-Request-ID": rid})

    # ------------------------------------------------------------ helpers
    def state() -> State:
        return app.state.sentinel

    def require_model() -> State:
        st = state()
        if st.bundle is None:
            raise HTTPException(503, "Model is not loaded; service is not ready")
        return st

    def explain_rows(st: State, sub: pd.DataFrame):
        """TreeSHAP contributions + reason codes for raw rows. Returns (contrib[n, f], [reasons per row])."""
        bundle = st.bundle
        _, contrib = bundle.contributions(sub)
        unscaled = make_features(sub)
        stats = st.reference.get("feature_stats", {})
        names = bundle.feature_names
        hours = ((sub["Time"].values % 86400) // 3600).astype(int)
        out = [build_reasons(names, contrib[i], unscaled.iloc[i][names].values, stats, hour=int(hours[i]))
               for i in range(len(sub))]
        return contrib, out

    def score(st: State, txns, explain: bool):
        started = time.perf_counter()
        raw = pd.DataFrame([t.model_dump() for t in txns])[RAW_FEATURES]
        bundle = st.bundle
        proba = bundle.predict_proba(raw)
        flagged = proba >= bundle.threshold

        reasons = {}
        if explain and flagged.any():
            idx = np.where(flagged)[0]
            _, per_row = explain_rows(st, raw.iloc[idx])
            reasons = {int(orig): per_row[k] for k, orig in enumerate(idx)}
        results = []
        for i, t in enumerate(txns):
            p = float(proba[i])
            results.append({
                "transaction_id": str(uuid.uuid4()),
                "is_fraud": bool(flagged[i]),
                "fraud_probability": round(p, 6),
                "risk_level": risk_level(p, bundle.threshold),
                "threshold": round(bundle.threshold, 6),
                "amount": t.Amount,
                "timestamp": _now(),
                "model_version": bundle.version,
                "reasons": reasons.get(i),
            })
        drift_keys = list(st.reference.get("psi_features", {}).keys())
        drift_rows = [{k: float(raw.iloc[i][k]) for k in drift_keys if k in raw.columns} for i in range(len(raw))]
        st.monitor.record(len(raw), int(flagged.sum()), (time.perf_counter() - started) * 1000, drift_rows)
        return results

    # ------------------------------------------------------------ probes (open)
    @app.get("/live")
    def live():
        return {"status": "alive"}

    @app.get("/ready")
    def ready():
        st = state()
        if st.bundle is None:
            return JSONResponse({"status": "not_ready", "reason": "model not loaded"}, status_code=503)
        return {"status": "ready", "model_version": st.bundle.version}

    @app.get("/health")
    def health():
        st = state()
        ok = st.bundle is not None
        body = {"status": "ready" if ok else "not_ready", "model_loaded": ok,
                "model_version": st.bundle.version if ok else None, "timestamp": _now()}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/", include_in_schema=False)
    def index():
        if not settings.frontend_path.exists():
            raise HTTPException(404, "Dashboard not found")
        return FileResponse(settings.frontend_path)

    # ------------------------------------------------------------ prediction
    @app.post("/predict", response_model=Prediction, dependencies=[AccessDep])
    def predict(t: Transaction, explain: bool = Query(False, description="Add reason codes (flagged transactions only)")):
        st = require_model()
        return score(st, [t], explain)[0]

    @app.post("/predict/batch", response_model=BatchResponse, dependencies=[AccessDep])
    def predict_batch(req: BatchRequest, explain: bool = Query(False)):
        st = require_model()
        results = score(st, req.transactions, explain)
        fraud = sum(r["is_fraud"] for r in results)
        return {"total": len(results), "fraud_count": fraud, "fraud_rate": round(fraud / len(results), 6),
                "model_version": st.bundle.version, "reasons_note": DISCLAIMER if explain else None,
                "results": results}

    # ------------------------------------------------------------ info / stats
    @app.get("/model/info", dependencies=[AccessDep])
    def model_info():
        st = require_model()
        m = st.bundle.metadata
        return {"model_type": "XGBoost", "model_version": m["model_version"], "strategy": m["strategy"],
                "calibration": m["calibration_method"], "threshold": m["threshold"],
                "risk_bands": m["risk_bands"], "costs": m["costs"], "features": len(m["feature_names"]),
                "feature_names": m["feature_names"], "input_schema": m["raw_input_features"],
                "data": {k: m["data"][k] for k in ("source", "synthetic", "rows", "frauds", "split_kind")},
                "library_versions": m["library_versions"], "trained_at": m["trained_at"]}

    @app.get("/stats/summary", dependencies=[AccessDep])
    def stats():
        st = require_model()
        t = st.metrics.get("test", {})
        val = lambda k: t.get(k, {}).get("value", 0.0)  # noqa: E731
        total = flagged = caught = 0
        saved = 0.0
        if st.replay is not None and len(st.replay_proba):
            f = st.replay_proba >= st.bundle.threshold
            total, flagged = len(f), int(f.sum())
            label = st.replay["Class"].values == 1
            caught = int((f & label).sum())
            saved = round(float(st.replay.loc[f & label, "Amount"].sum()), 2)
        snap = st.monitor.snapshot()
        return {
            "source": "model/metrics.json (held-out test split)",
            "synthetic_data": st.metrics.get("synthetic_data", None),
            "model_version": st.bundle.version,
            "threshold": st.bundle.threshold,
            "replay_total": total, "replay_flagged": flagged, "replay_true_frauds_caught": caught,
            "replay_flag_rate": round(flagged / total, 4) if total else 0.0,
            "replay_fraud_amount_caught": saved,
            "avg_response_ms": round(snap["latency_ms"]["mean"], 3),
            "pr_auc": val("pr_auc"), "roc_auc": val("roc_auc"), "precision": val("precision"),
            "recall": val("recall"), "f1": val("f1"), "accuracy": val("accuracy"),
            "ci95": {k: t[k]["ci95"] for k in ("pr_auc", "precision", "recall", "f1") if k in t},
            "confusion": st.metrics.get("test_confusion"),
        }

    @app.get("/shap/global", dependencies=[AccessDep])
    def shap_global():
        st = require_model()
        return {"method": "mean |TreeSHAP| on replay sample (log-odds)", "feature_importance": st.replay_importance}

    @app.get("/replay/feed", dependencies=[AccessDep])
    def replay_feed():
        """Rolling 20-row window over held-out test rows, scored by the loaded model. This is a replay, not live traffic."""
        st = require_model()
        if st.replay is None or not len(st.replay_proba):
            return {"transactions": [], "count": 0, "source": "replay_of_held_out_test_data"}
        n = len(st.replay_proba)
        size = min(20, n)
        with st.replay_lock:
            start = st.replay_offset
            st.replay_offset = (start + min(5, n)) % n
        rows = []
        for pos, i in enumerate([(start + k) % n for k in range(size)]):
            p = float(st.replay_proba[i])
            r = st.replay.iloc[i]
            rows.append({
                "id": f"REPLAY-{i:05d}", "row_index": i, "amount": round(float(r["Amount"]), 2),
                "time_seconds": float(r["Time"]), "fraud_probability": round(p, 6),
                "risk_level": risk_level(p, st.bundle.threshold),
                "is_fraud": p >= st.bundle.threshold, "true_label": int(r["Class"]),
            })
        return {"transactions": rows, "count": len(rows), "source": "replay_of_held_out_test_data"}

    @app.get("/replay/threshold", dependencies=[AccessDep])
    def replay_threshold(value: float = Query(..., ge=0.0, le=1.0)):
        """Real precision/recall/cost at an arbitrary threshold, measured on the labelled replay sample."""
        st = require_model()
        if st.replay is None:
            raise HTTPException(404, "No replay sample available")
        y = st.replay["Class"].values.astype(int)
        flagged = st.replay_proba >= value
        data = st.bundle.metadata["data"]
        test_frauds = data["split_frauds"]["test"]
        test_legit = data["split_rows"]["test"] - test_frauds
        legit = y == 0

        if "weight" in st.replay.columns:
            wts = st.replay["weight"].values.astype(float)
        else:  # older replay file: uniform legit weight
            n_legit = int(legit.sum())
            wts = np.where(legit, test_legit / n_legit if n_legit else 1.0, 1.0)
        weighted = int(y.sum()) == test_frauds
        if not weighted:
            wts = np.ones(len(y))

        tp, fn = int((flagged & (y == 1)).sum()), int((~flagged & (y == 1)).sum())
        fp_s = int((flagged & legit).sum())
        fp = float(wts[flagged & legit].sum())
        tn = float(wts[~flagged & legit].sum())
        hard = legit & (wts == 1.0)
        exact_from = float(st.replay_proba[hard].min()) if weighted and hard.any() else None
        exact = bool(weighted and exact_from is not None and value >= exact_from)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        costs = st.bundle.metadata["costs"]
        if exact:
            note = (f"Exact for the full held-out test split at this threshold (all {int(y.sum())} frauds and every "
                    f"legitimate row scoring >= {exact_from:.3f} are in the replay sample).")
        elif weighted:
            note = (f"Estimated for the full test split: below {exact_from if exact_from is not None else 0:.3f} the "
                    f"false-alarm count is scaled up from a random sample of easy rows, so it is noisy. "
                    f"Only {int(y.sum())} frauds: a guide, not a guarantee.")
        else:
            note = ("Measured on the raw replay sample (not prevalence-corrected); precision and cost are not "
                    "comparable to metrics.json.")
        return {"threshold": value, "model_threshold": st.bundle.threshold, "rows": len(y), "frauds": int(y.sum()),
                "weighted": weighted, "exact": exact, "exact_from": exact_from,
                "tp": tp, "fp": int(round(fp)), "fn": fn, "tn": int(round(tn)), "fp_sampled": fp_s,
                "precision": round(precision, 4), "recall": round(recall, 4),
                "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
                "expected_cost": round(fn * costs["false_negative"] + fp * costs["false_positive"], 2),
                "note": note}

    @app.get("/replay/explain", dependencies=[AccessDep])
    def replay_explain(flagged: bool = Query(True)):
        """Real TreeSHAP explanation for the next flagged (or unflagged) replay row."""
        st = require_model()
        if st.replay is None:
            raise HTTPException(404, "No replay sample available")
        mask = st.replay_proba >= st.bundle.threshold
        pool = np.where(mask if flagged else ~mask)[0]
        if not len(pool):
            raise HTTPException(404, "No matching rows in the replay sample")
        with st.replay_lock:
            i = int(pool[st.explain_offset % len(pool)])
            st.explain_offset += 1
        sub = st.replay.iloc[[i]][RAW_FEATURES]
        contrib, reasons = explain_rows(st, sub)
        names = st.bundle.feature_names
        order = np.argsort(-np.abs(contrib[0]))[:10]
        p = float(st.replay_proba[i])
        return {"row_index": i, "amount": float(sub["Amount"].iloc[0]), "fraud_probability": round(p, 6),
                "is_fraud": p >= st.bundle.threshold, "true_label": int(st.replay["Class"].iloc[i]),
                "risk_level": risk_level(p, st.bundle.threshold),
                "contributions": [{"feature": names[j], "contribution": round(float(contrib[0][j]), 4)} for j in order],
                "reasons": reasons[0], "note": DISCLAIMER}

    # ------------------------------------------------------------ monitoring
    @app.get("/monitoring/metrics", dependencies=[AccessDep])
    def monitoring_metrics():
        require_model()
        return state().monitor.snapshot()

    @app.get("/monitoring/drift", dependencies=[AccessDep])
    def monitoring_drift():
        require_model()
        return state().monitor.drift_report()

    return app


app = create_app()
