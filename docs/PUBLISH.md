# Publishing SENTINEL

## 1. GitHub (code)
```powershell
cd C:\Users\SAMSUNG\Downloads\sentinel
git init -b main
git config user.name  "Your Name"
git config user.email "you@example.com"
git add .
git status                       # CHECK: data/creditcard.csv, .venv and .env must NOT be listed
git commit -m "SENTINEL fraud detection"
```
Create an EMPTY repository on github.com (no README / .gitignore / licence), then:
```powershell
git remote add origin https://github.com/<USERNAME>/sentinel-fraud-detection.git
git push -u origin main
```
GitHub Actions (lint, tests, Docker build) starts automatically (Actions tab).

## 2. Hugging Face (Docker Space + model repo)
```powershell
pip install huggingface_hub
$env:HF_TOKEN="hf_xxx"            # WRITE token from https://huggingface.co/settings/tokens (keep it private)
python scripts/publish_hf.py --dry-run
python scripts/publish_hf.py
```
The Space runs with `AUTH_DISABLED=true` (HF shows apps in an iframe, where the API-key popup is blocked). It is a public demo.
Check `NOTICE.md` about redistributing `model/replay_sample.csv` (rows of the Kaggle dataset).
