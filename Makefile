.PHONY: install install-dev data-kaggle data-synthetic train train-synthetic test test-fast lint api up up-stream
PY ?= python

install:        ; pip install -r requirements.txt
install-dev:    ; pip install -r requirements-dev.txt
data-kaggle:    ; pip install kaggle && $(PY) scripts/download_data.py
data-synthetic: ; $(PY) scripts/make_synthetic.py
train:          ; $(PY) -m src.train                      # real data/creditcard.csv
train-synthetic:; $(PY) -m src.train --synthetic          # offline demo (NOT real data)
test:           ; $(PY) -m pytest -q
test-fast:      ; $(PY) -m pytest -q -m "not slow"
lint:           ; ruff check .
api:            ; API_KEYS=$${API_KEYS:-dev-key-change-me} uvicorn api.main:app --port 8000
up:             ; docker compose up --build
up-stream:      ; docker compose --profile streaming up --build
