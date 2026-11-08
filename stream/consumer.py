"""Kafka consumer: scores each transaction via the API, alerts on fraud, tracks live precision/recall
against the ground-truth labels carried by the replay, and sends unscoreable messages to a dead-letter topic.

    API_URL=http://localhost:8000 API_KEY=dev-key-change-me python -m stream.consumer
"""

import json
import logging
import time
from dataclasses import dataclass

import requests

from .settings import StreamSettings

logger = logging.getLogger("consumer")

FEATURE_KEYS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


class ScoringFailed(Exception):
    def __init__(self, reason: str, retryable: bool):
        super().__init__(reason)
        self.reason, self.retryable = reason, retryable


@dataclass
class Tally:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    unlabeled: int = 0
    dead_lettered: int = 0

    def update(self, flagged: bool, label):
        if label is None:
            self.unlabeled += 1
        elif flagged and label == 1:
            self.tp += 1
        elif flagged and label == 0:
            self.fp += 1
        elif not flagged and label == 1:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def scored(self) -> int:
        return self.tp + self.fp + self.fn + self.tn + self.unlabeled

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def summary(self) -> str:
        return (f"scored={self.scored} dead_lettered={self.dead_lettered} | TP={self.tp} FP={self.fp} "
                f"FN={self.fn} TN={self.tn} | precision={self.precision:.3f} recall={self.recall:.3f} f1={self.f1:.3f}")


def build_payload(txn: dict) -> dict:
    """Only the model's raw input fields; strict API schema rejects extras."""
    missing = [k for k in FEATURE_KEYS if k not in txn]
    if missing:
        raise ScoringFailed(f"message missing fields: {missing}", retryable=False)
    return {k: txn[k] for k in FEATURE_KEYS}


def score_with_retry(session, cfg: StreamSettings, payload: dict, sleep=time.sleep) -> dict:
    url = cfg.api_url.rstrip("/") + "/predict?explain=true"
    headers = {"X-API-Key": cfg.api_key} if cfg.api_key else {}
    last = "unknown error"
    for attempt in range(cfg.score_retries + 1):
        try:
            resp = session.post(url, json=payload, headers=headers, timeout=5)
        except requests.RequestException as exc:
            last, wait = f"request failed: {exc.__class__.__name__}", cfg.backoff_base_sec * 2 ** attempt
        else:
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (400, 401, 403, 404, 422):
                raise ScoringFailed(f"HTTP {resp.status_code}: {resp.text[:200]}", retryable=False)
            last = f"HTTP {resp.status_code}"
            wait = cfg.backoff_base_sec * 2 ** attempt
            if resp.status_code == 429:
                try:
                    wait = max(wait, float(resp.headers.get("Retry-After", wait)))
                except ValueError:
                    pass
        if attempt < cfg.score_retries:
            logger.warning("scoring attempt %d failed (%s); retrying in %.1fs", attempt + 1, last, wait)
            sleep(wait)
    raise ScoringFailed(f"gave up after {cfg.score_retries + 1} attempts: {last}", retryable=True)


def top_reason(result: dict) -> str:
    reasons = result.get("reasons") or []
    return reasons[0]["description"] if reasons else "n/a (no reason codes returned)"


def handle_message(txn: dict, session, cfg: StreamSettings, tally: Tally, dead_letter) -> None:
    try:
        result = score_with_retry(session, cfg, build_payload(txn))
    except ScoringFailed as exc:
        tally.dead_lettered += 1
        dead_letter({"error": exc.reason, "retryable": exc.retryable, "message": txn})
        logger.error("dead-lettered row_id=%s: %s", txn.get("row_id"), exc.reason)
        return
    flagged = bool(result.get("is_fraud"))
    tally.update(flagged, txn.get("label"))
    if flagged:
        logger.warning("FRAUD ALERT row=%s txn=%s p=%.3f risk=%s amount=%.2f true_label=%s | %s",
                       txn.get("row_id"), result.get("transaction_id"), result.get("fraud_probability", 0),
                       result.get("risk_level"), txn.get("Amount", 0), txn.get("label"), top_reason(result))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = StreamSettings()
    from kafka import KafkaConsumer, KafkaProducer

    from .producer import connect_with_retries

    consumer = connect_with_retries(
        lambda: KafkaConsumer(cfg.topic, bootstrap_servers=[cfg.broker], group_id=cfg.group_id,
                              value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                              auto_offset_reset=cfg.offset_reset, enable_auto_commit=True),
        cfg.connect_retries, cfg.backoff_base_sec, f"Kafka at {cfg.broker}")
    dlq = KafkaProducer(bootstrap_servers=[cfg.broker], value_serializer=lambda v: json.dumps(v).encode("utf-8"))

    def dead_letter(payload):
        dlq.send(cfg.dlq_topic, value=payload)

    session, tally = requests.Session(), Tally()
    logger.info("Listening on '%s'; scoring via %s; DLQ '%s'", cfg.topic, cfg.api_url, cfg.dlq_topic)
    try:
        for msg in consumer:
            handle_message(msg.value, session, cfg, tally, dead_letter)
            if (tally.scored + tally.dead_lettered) % cfg.summary_every == 0:
                logger.info("RUNNING %s", tally.summary())
    except KeyboardInterrupt:
        pass
    finally:
        dlq.flush()
        logger.info("FINAL %s", tally.summary())
        consumer.close()


if __name__ == "__main__":
    main()
