"""Build a ready-to-upload Hugging Face *Docker Space* folder (default: hf_space/).

    python scripts/make_hf_space.py
    # then create a Space (SDK: Docker) on huggingface.co and push the folder; see the README section "Publish".

The public demo runs with AUTH_DISABLED=true (HF shows the app inside an iframe where the API-key popup would be blocked).
Everything it serves is held-out replay data plus scoring of whatever a visitor submits. Do not put anything private in it.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

README = """---
title: Sentinel Fraud Detection
emoji: 🛡️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Calibrated fraud scoring API and dashboard
---

# Sentinel — fraud detection demo

XGBoost fraud classifier with probability calibration, a cost-optimal decision threshold, TreeSHAP reason codes,
a FastAPI service and a dashboard. Source code: add your GitHub link here.

* Dashboard: this page. API docs: `/docs`.
* The dashboard shows a **replay of held-out test data** (not live traffic).
* Metrics are from a time-ordered held-out test split with only 75 frauds, so confidence intervals are wide.
  See the model card in the GitHub repository.
* Data: Kaggle *Credit Card Fraud Detection* (ULB / Worldline, DbCL v1.0). See `NOTICE.md`.
* Demo only: not for real fraud decisions.
"""

DOCKERFILE = """FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \\
    AUTH_DISABLED=true RATE_LIMIT_PER_MIN=600 ALLOWED_ORIGINS=*
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY frontend/ frontend/
COPY model/ model/

# Hugging Face runs Docker Spaces as UID 1000
RUN useradd -m -u 1000 user
USER user

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "hf_space"))
    a = ap.parse_args()
    out = Path(a.out)
    if not (ROOT / "model" / "metadata.json").exists():
        sys.exit("model/ has no trained artifacts. Run: python -m src.train")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("src", "api", "frontend", "model"):
        shutil.copytree(ROOT / d, out / d, ignore=ignore)
    for f in ("requirements.txt", "LICENSE", "NOTICE.md"):
        if (ROOT / f).exists():
            shutil.copy(ROOT / f, out / f)
    (out / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"Space folder ready: {out}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
