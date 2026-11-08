"""Training CLI.

    python -m src.train --config config/train.yaml            # real data/creditcard.csv
    python -m src.train --config config/train.yaml --synthetic # offline demo data (NOT real)

Pipeline: time-ordered split -> 5-fold CV model selection on TRAIN -> refit ->
calibrate on VAL -> cost-optimal threshold on VAL -> evaluate once on TEST (bootstrap CIs).
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import __version__
from .config import ROOT, load_config, sha256_file
from .decision import optimal_threshold, risk_band_edges
from .drift import reference_bins
from .features import MODEL_FEATURES, RAW_FEATURES, SCALED_FEATURES, make_features
from .metrics import bootstrap_ci, confusion, point_metrics, recall_at_precision
from .model_bundle import Calibrator, ModelBundle, save_bundle

logger = logging.getLogger("train")


# ----------------------------------------------------------------------------- data
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = RAW_FEATURES + ["Class"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    if df[needed].isnull().any().any():
        raise SystemExit("Dataset contains missing values; clean it first.")
    if not set(df["Class"].unique()) <= {0, 1}:
        raise SystemExit("Class must be 0/1.")
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    n = len(df)
    i, j = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()


# ----------------------------------------------------------------------------- model
def build_pipeline(strategy: str, cfg: dict, y_train) -> ImbPipeline:
    params = dict(cfg["xgb"])
    params.update(random_state=cfg["seed"], eval_metric="aucpr")
    if strategy == "scale_pos_weight":
        pos = max(int(np.sum(y_train)), 1)
        params["scale_pos_weight"] = float((len(y_train) - pos) / pos)
    prep = ColumnTransformer([("amount", StandardScaler(), SCALED_FEATURES)],
                             remainder="passthrough", verbose_feature_names_out=False)
    prep.set_output(transform="pandas")
    steps = [("prep", prep)]
    if strategy == "smote":
        steps.append(("smote", SMOTE(sampling_strategy=cfg["smote_sampling_strategy"],
                                     random_state=cfg["seed"])))
    steps.append(("clf", XGBClassifier(**params)))
    return ImbPipeline(steps)


def cross_validate(strategy, cfg, X, y):
    skf = StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=cfg["seed"])
    ap, rap = [], []
    for tr, va in skf.split(X, y):
        pipe = build_pipeline(strategy, cfg, y.iloc[tr])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[va])[:, 1]
        ap.append(average_precision_score(y.iloc[va], p))
        rap.append(recall_at_precision(y.iloc[va], p, cfg["cv"]["min_precision"]))
    return {"pr_auc_mean": float(np.mean(ap)), "pr_auc_std": float(np.std(ap)),
            "recall_at_precision_mean": float(np.mean(rap)),
            "pr_auc_folds": [float(a) for a in ap]}


def fit_calibrator(method: str, raw_val, y_val, min_pos: int):
    n_pos = int(np.sum(y_val))
    if method == "auto":
        method = "isotonic" if n_pos >= min_pos else "sigmoid"
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_val, y_val)
        return Calibrator.isotonic(iso.X_thresholds_, iso.y_thresholds_), method
    if method == "sigmoid":
        p = np.clip(raw_val, 1e-7, 1 - 1e-7)
        z = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y_val)
        return Calibrator.sigmoid(lr.coef_[0][0], lr.intercept_[0]), method
    return Calibrator.identity(), "none"


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


# ----------------------------------------------------------------------------- figures
def save_figures(docs_dir: Path, y_test, raw_test, cal_test, contrib_importance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve

    fig_dir = docs_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, p in (("raw model score", raw_test), ("calibrated", cal_test)):
        frac, mean = calibration_curve(y_test, p, n_bins=8, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Reliability curve (test split)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "reliability.png", dpi=120); plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_test, cal_test)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec, prec)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall (test split)")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(fig_dir / "pr_curve.png", dpi=120); plt.close(fig)

    top = sorted(contrib_importance.items(), key=lambda kv: kv[1])[-15:]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([k for k, _ in top], [v for _, v in top])
    ax.set_xlabel("mean |TreeSHAP| (log-odds)"); ax.set_title("Global feature importance")
    fig.tight_layout(); fig.savefig(fig_dir / "feature_importance.png", dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------- main
def build_replay(test: pd.DataFrame, n_rows: int, seed: int, scores=None, hard_floor: float = 0.01) -> pd.DataFrame:
    """Replay sample with a per-row `weight`, so the dashboard can reconstruct the FULL test split.

    * every test fraud, weight 1
    * 'hard' legitimate rows (model score >= hard_floor, at most half the budget), weight 1  -> false alarms at any
      threshold >= the lowest hard score are counted EXACTLY, not estimated
    * a random sample of the remaining easy legitimate rows, weight = easy_total / easy_sampled

    Without `scores` there are no hard rows and all legitimate rows are a uniform random sample. A plain 'last N rows'
    slice is not used: on real data (~0.17% fraud) it can hold zero frauds.
    """
    frauds = test[test["Class"] == 1]
    legit = test[test["Class"] == 0]
    budget = min(max(n_rows - len(frauds), 0), len(legit))
    if scores is not None:
        s = pd.Series(np.asarray(scores, dtype=float), index=test.index).loc[legit.index]
        hard_idx = s[s >= hard_floor].sort_values(ascending=False).index[: budget // 2]
    else:
        hard_idx = legit.index[:0]
    hard = legit.loc[hard_idx]
    rest = legit.drop(hard_idx)
    n_easy = min(budget - len(hard), len(rest))
    easy = rest.sample(n=n_easy, random_state=seed)
    w_easy = len(rest) / n_easy if n_easy else 1.0

    parts = [frauds.assign(weight=1.0), hard.assign(weight=1.0), easy.assign(weight=w_easy)]
    sample = pd.concat(parts).sort_values("Time", kind="stable")
    out = sample[RAW_FEATURES + ["Class", "weight"]].copy()
    v_cols = [c for c in RAW_FEATURES if c.startswith("V")]
    out[v_cols] = out[v_cols].round(5)
    out["weight"] = out["weight"].round(6)
    return out.reset_index(drop=True)


def run(cfg: dict, data_path: Path, model_dir: Path, docs_dir: Path, synthetic: bool,
        update_readme: bool = True, readme_path: Path = None, n_boot=None) -> dict:
    seed = cfg["seed"]
    df = load_dataset(data_path)
    train, val, test = time_split(df, cfg["split"]["train_frac"], cfg["split"]["val_frac"])
    logger.info("rows train/val/test = %d/%d/%d; frauds = %d/%d/%d", len(train), len(val), len(test),
                train.Class.sum(), val.Class.sum(), test.Class.sum())

    X_tr, y_tr = make_features(train), train["Class"]
    X_va, y_va = make_features(val), val["Class"]
    X_te, y_te = make_features(test), test["Class"]

    # 1) model selection by stratified CV on TRAIN only
    cv_results = {}
    for strategy in cfg["strategies"]:
        logger.info("CV strategy: %s", strategy)
        cv_results[strategy] = cross_validate(strategy, cfg, X_tr, y_tr)
        logger.info("  PR-AUC %.4f +- %.4f", cv_results[strategy]["pr_auc_mean"], cv_results[strategy]["pr_auc_std"])
    best = max(cv_results, key=lambda s: cv_results[s]["pr_auc_mean"])
    logger.info("selected strategy (by CV PR-AUC): %s", best)

    # 2) refit on all TRAIN
    pipe = build_pipeline(best, cfg, y_tr)
    pipe.fit(X_tr, y_tr)
    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps["prep"].named_transformers_["amount"]

    # 3) calibrate + choose threshold on VAL
    raw_va = pipe.predict_proba(X_va)[:, 1]
    calibrator, cal_method = fit_calibrator(cfg["calibration"]["method"], raw_va, y_va,
                                            cfg["calibration"]["min_positives_for_isotonic"])
    cal_va = calibrator(raw_va)

    cfg_fn = cfg["costs"]["false_negative"]
    cost_fn = float(train.loc[train.Class == 1, "Amount"].mean()) if cfg_fn == "auto" else float(cfg_fn)
    cost_fp = float(cfg["costs"]["false_positive"])
    th = optimal_threshold(y_va, cal_va, cost_fn, cost_fp)
    threshold = th["threshold"]

    # 4) single evaluation on untouched TEST
    raw_te = pipe.predict_proba(X_te)[:, 1]
    cal_te = calibrator(raw_te)
    min_prec = cfg["cv"]["min_precision"]
    test_point = point_metrics(y_te, cal_te, threshold, cost_fn, cost_fp, min_prec)
    boot = bootstrap_ci(y_te, cal_te, threshold, cost_fn, cost_fp, min_prec,
                        n_boot=n_boot or cfg["bootstrap"]["n"], ci=cfg["bootstrap"]["ci"], seed=seed)
    test_metrics = {k: {"value": v, "ci95": boot.get(k)} for k, v in test_point.items()}
    cm = confusion(y_te, cal_te >= threshold)
    p_default = point_metrics(y_te, cal_te, 0.5, cost_fn, cost_fp, min_prec)
    baselines = {
        "cost_flag_nothing": float(y_te.sum() * cost_fn),
        "cost_at_threshold_0.5": p_default["expected_cost"],
        "cost_at_chosen_threshold": test_point["expected_cost"],
    }
    calibration_report = {
        "method": cal_method,
        "brier_raw_test": brier(y_te, raw_te),
        "brier_calibrated_test": brier(y_te, cal_te),
        "mean_pred_raw_test": float(raw_te.mean()),
        "mean_pred_calibrated_test": float(cal_te.mean()),
        "observed_fraud_rate_test": float(y_te.mean()),
    }

    # 5) global importance, reference stats for explanations + drift
    bundle_tmp_meta = {"feature_names": [str(c) for c in clf.feature_names_in_]}
    import xgboost as xgb
    contrib = clf.get_booster().predict(
        xgb.DMatrix(pipe.named_steps["prep"].transform(X_va), feature_names=bundle_tmp_meta["feature_names"]),
        pred_contribs=True)[:, :-1]
    mean_abs = np.abs(contrib).mean(axis=0)
    importance = dict(zip(bundle_tmp_meta["feature_names"], map(float, mean_abs)))
    top_v = [f for f, _ in sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
             if f.startswith("V")][:5]

    X_tr_unscaled = X_tr  # explanation z-scores use *unscaled* training stats
    feature_stats = {c: {"mean": float(X_tr_unscaled[c].mean()), "std": float(X_tr_unscaled[c].std() or 1.0)}
                     for c in MODEL_FEATURES}
    psi_features = {"Amount": reference_bins(train["Amount"].values)}
    for f in top_v:
        psi_features[f] = reference_bins(X_tr[f].values)
    reference = {"feature_stats": feature_stats, "psi_features": psi_features,
                 "n_reference_rows": int(len(train))}

    # 6) persist artifacts (pickle-free)
    now = datetime.now(timezone.utc)
    metadata = {
        "model_version": now.strftime("%Y%m%d-%H%M%S"),
        "sentinel_version": __version__,
        "trained_at": now.isoformat(),
        "strategy": best,
        "calibration_method": cal_method,
        "threshold": threshold,
        "risk_bands": risk_band_edges(threshold),
        "costs": {"false_negative": cost_fn, "false_positive": cost_fp,
                  "false_negative_source": "mean fraud amount in train split" if cfg_fn == "auto" else "config"},
        "raw_input_features": RAW_FEATURES,
        "data": {
            "source": "SYNTHETIC (offline demo - NOT real data)" if synthetic else "user-provided dataset",
            "synthetic": bool(synthetic),
            "file": str(data_path.name),
            "sha256": sha256_file(data_path),
            "rows": int(len(df)), "frauds": int(df.Class.sum()),
            "split_rows": {"train": len(train), "val": len(val), "test": len(test)},
            "split_frauds": {"train": int(train.Class.sum()), "val": int(val.Class.sum()),
                             "test": int(test.Class.sum())},
            "split_kind": "time-ordered (train earliest, test latest)",
        },
        "config": cfg,
    }
    metadata = save_bundle(model_dir, clf, scaler.mean_, scaler.scale_, calibrator, metadata)

    metrics = {
        "model_version": metadata["model_version"],
        "synthetic_data": bool(synthetic),
        "selected_strategy": best,
        "cv": cv_results,
        "threshold": threshold,
        "costs": metadata["costs"],
        "validation_cost_at_threshold": th["expected_cost"],
        "test": test_metrics,
        "test_confusion": cm,
        "test_cost_baselines": baselines,
        "calibration": calibration_report,
        "data": metadata["data"],
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (model_dir / "reference_stats.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")

    replay = build_replay(test, int(cfg.get("replay_rows", 3000)), seed, scores=cal_te)
    replay.to_csv(model_dir / "replay_sample.csv", index=False)

    # 7) sanity: the saved bundle must reproduce the in-memory pipeline exactly
    bundle = ModelBundle.load(model_dir)
    diff = float(np.max(np.abs(bundle.predict_proba(test[RAW_FEATURES]) - cal_te)))
    if diff > 1e-5:
        raise RuntimeError(f"Saved bundle disagrees with trained pipeline (max diff {diff})")
    logger.info("bundle/pipeline max abs diff = %.2e", diff)

    # 8) docs
    from .report import update_readme_block, write_model_card
    save_figures(docs_dir, y_te.values, raw_te, cal_te, importance)
    write_model_card(docs_dir / "MODEL_CARD.md", metrics, metadata, importance)
    if update_readme and readme_path and Path(readme_path).exists():
        update_readme_block(Path(readme_path), metrics, metadata)
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "train.yaml"))
    ap.add_argument("--data", help="override data path")
    ap.add_argument("--synthetic", action="store_true", help="use generated SYNTHETIC data (offline demo)")
    ap.add_argument("--model-dir", default=str(ROOT / "model"))
    ap.add_argument("--docs-dir", default=str(ROOT / "docs"))
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.synthetic:
        from .synthetic import generate
        path = Path(args.data or ROOT / cfg["data"]["synthetic_path"])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            generate().to_csv(path, index=False)
            logger.warning("Generated SYNTHETIC dataset at %s", path)
    else:
        path = Path(args.data or ROOT / cfg["data"]["path"])
        if not path.exists():
            sys.exit(f"Dataset not found: {path}\nGet the real data with:  python scripts/download_data.py\n"
                     "or run on offline demo data with:  python -m src.train --synthetic")
    run(cfg, path, Path(args.model_dir), Path(args.docs_dir), args.synthetic,
        update_readme=not args.no_readme, readme_path=ROOT / "README.md")


if __name__ == "__main__":
    main()
