"""In-memory serving metrics + input drift buffer."""

import logging
import threading
import time
from collections import deque

import numpy as np

from src.drift import psi, psi_status

logger = logging.getLogger("sentinel.monitoring")


class Monitor:
    def __init__(self, reference: dict, window: int, min_samples: int, log_every: int):
        self.reference = reference or {}
        self.min_samples = min_samples
        self.log_every = max(1, log_every)
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.requests = 0
        self.predictions = 0
        self.fraud_flags = 0
        self.latencies_ms = deque(maxlen=2000)
        self._drift_rows = deque(maxlen=window)  # dict(feature -> value)

    def record(self, n_rows: int, n_flagged: int, latency_ms: float, drift_rows):
        with self._lock:
            before = self.predictions
            self.requests += 1
            self.predictions += n_rows
            self.fraud_flags += n_flagged
            self.latencies_ms.append(latency_ms)
            self._drift_rows.extend(drift_rows)
            crossed = self.predictions // self.log_every > before // self.log_every
            snap = self._snapshot_locked() if crossed else None
        if snap:
            logger.info("serving stats: predictions=%d fraud_rate=%.4f p50=%.1fms p95=%.1fms p99=%.1fms",
                        snap["predictions"], snap["flag_rate"], snap["latency_ms"]["p50"],
                        snap["latency_ms"]["p95"], snap["latency_ms"]["p99"])

    def _snapshot_locked(self) -> dict:
        lat = np.array(self.latencies_ms) if self.latencies_ms else np.array([0.0])
        return {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "requests": self.requests,
            "predictions": self.predictions,
            "flagged": self.fraud_flags,
            "flag_rate": (self.fraud_flags / self.predictions) if self.predictions else 0.0,
            "latency_ms": {
                "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                "p99": float(np.percentile(lat, 99)), "mean": float(lat.mean()),
                "samples": len(self.latencies_ms),
            },
        }

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_locked()

    def drift_report(self) -> dict:
        with self._lock:
            rows = list(self._drift_rows)
        n = len(rows)
        base = {"window_samples": n, "min_samples": self.min_samples, "reference": "training split"}
        if n < self.min_samples:
            return {**base, "status": "insufficient_data", "features": {}}
        out, worst = {}, 0.0
        for feat, ref in self.reference.items():
            vals = [r[feat] for r in rows if feat in r]
            if not vals:
                continue
            value = psi(ref, vals)
            worst = max(worst, value)
            out[feat] = {"psi": round(value, 4), "status": psi_status(value)}
        return {**base, "status": psi_status(worst), "max_psi": round(worst, 4), "features": out,
                "note": "PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant. Input drift only; cannot detect label drift."}
