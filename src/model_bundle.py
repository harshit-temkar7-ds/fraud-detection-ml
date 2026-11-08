"""Versioned, pickle-free model bundle.

Artifacts in model/:
  xgb_model.json     XGBoost native JSON
  preprocess.json    fitted scaler parameters
  calibrator.json    probability calibrator (isotonic / sigmoid / none)
  metadata.json      version, feature order, threshold, costs, library versions,
                     SHA-256 of every artifact above
  metrics.json       evaluation results (written by training)
  reference_stats.json  feature stats + PSI reference bins (written by training)
"""

import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import sha256_file
from .features import SCALED_FEATURES, make_features

logger = logging.getLogger(__name__)

ARTIFACT_FILES = ("xgb_model.json", "preprocess.json", "calibrator.json")
SCHEMA_VERSION = 2


class ArtifactError(RuntimeError):
    """Model artifacts are missing, corrupted, or incompatible."""


def library_versions() -> dict:
    import numpy
    import pandas
    return {
        "python": platform.python_version(),
        "xgboost": xgb.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }


class Calibrator:
    """Pure-numpy probability calibrator, serialisable to JSON."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.method = spec["method"]

    @classmethod
    def identity(cls):
        return cls({"method": "none"})

    @classmethod
    def isotonic(cls, x_thresholds, y_thresholds):
        return cls({"method": "isotonic", "x": [float(v) for v in x_thresholds],
                    "y": [float(v) for v in y_thresholds]})

    @classmethod
    def sigmoid(cls, a: float, b: float):
        return cls({"method": "sigmoid", "a": float(a), "b": float(b)})

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == "isotonic":
            return np.interp(p, self.spec["x"], self.spec["y"])
        if self.method == "sigmoid":
            clipped = np.clip(p, 1e-7, 1 - 1e-7)
            z = np.log(clipped / (1 - clipped))
            return 1.0 / (1.0 + np.exp(-(self.spec["a"] * z + self.spec["b"])))
        return p


@dataclass
class ModelBundle:
    metadata: dict
    clf: xgb.XGBClassifier
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    calibrator: Calibrator

    # ---- properties -------------------------------------------------------
    @property
    def feature_names(self) -> list:
        return self.metadata["feature_names"]

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    # ---- inference --------------------------------------------------------
    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Raw transactions -> scaled feature frame in the exact training order."""
        x = make_features(raw)
        for i, col in enumerate(SCALED_FEATURES):
            x[col] = (x[col] - self.scaler_mean[i]) / self.scaler_scale[i]
        return x[self.feature_names]

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        """Calibrated fraud probabilities."""
        return self.calibrator(self.predict_raw(self.prepare(raw)))

    def predict_raw(self, prepared: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(prepared)[:, 1]

    def contributions(self, raw: pd.DataFrame):
        """TreeSHAP contributions (log-odds). Returns (prepared_features, contribs[n, n_features])."""
        prepared = self.prepare(raw)
        booster = self.clf.get_booster()
        dm = xgb.DMatrix(prepared, feature_names=self.feature_names)
        contribs = booster.predict(dm, pred_contribs=True)[:, :-1]  # last column = bias
        return prepared, contribs

    # ---- persistence ------------------------------------------------------
    @classmethod
    def load(cls, model_dir, strict_versions: bool = False) -> "ModelBundle":
        d = Path(model_dir)
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            raise ArtifactError(f"metadata.json not found in {d}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactError(
                f"Unsupported artifact schema {metadata.get('schema_version')} (expected {SCHEMA_VERSION})")

        for name in ARTIFACT_FILES:
            path = d / name
            if not path.exists():
                raise ArtifactError(f"Missing artifact: {path}")
            expected = metadata.get("artifact_sha256", {}).get(name)
            if expected != sha256_file(path):
                raise ArtifactError(f"Checksum mismatch for {name}: artifact is corrupted or was modified")

        trained = metadata.get("library_versions", {})
        current = library_versions()
        for lib in ("xgboost",):
            t, c = trained.get(lib, ""), current[lib]
            if t.split(".")[:2] != c.split(".")[:2]:
                message = f"{lib} version mismatch: trained with {t}, running {c}"
                if strict_versions:
                    raise ArtifactError(message)
                logger.warning(message)

        clf = xgb.XGBClassifier()
        clf.load_model(str(d / "xgb_model.json"))
        pre = json.loads((d / "preprocess.json").read_text(encoding="utf-8"))
        cal = Calibrator(json.loads((d / "calibrator.json").read_text(encoding="utf-8")))
        bundle = cls(metadata, clf, np.array(pre["mean"], float), np.array(pre["scale"], float), cal)
        if list(clf.feature_names_in_) != bundle.feature_names:
            raise ArtifactError("Feature order in xgb_model.json does not match metadata.json")
        return bundle


def save_bundle(model_dir, clf, scaler_mean, scaler_scale, calibrator: Calibrator, metadata: dict) -> dict:
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(d / "xgb_model.json"))
    (d / "preprocess.json").write_text(json.dumps({
        "scaled_features": SCALED_FEATURES,
        "mean": [float(v) for v in scaler_mean],
        "scale": [float(v) for v in scaler_scale],
    }, indent=2), encoding="utf-8")
    (d / "calibrator.json").write_text(json.dumps(calibrator.spec), encoding="utf-8")

    metadata = dict(metadata)
    metadata["schema_version"] = SCHEMA_VERSION
    metadata["feature_names"] = [str(c) for c in clf.feature_names_in_]
    metadata["library_versions"] = library_versions()
    metadata["artifact_sha256"] = {n: sha256_file(d / n) for n in ARTIFACT_FILES}
    (d / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
