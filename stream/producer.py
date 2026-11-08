"""Kafka producer: replays REAL held-out transactions (with their true labels) at a fixed rate.

    KAFKA_BROKER=localhost:9092 python -m stream.producer
"""

import json
import logging
import time

import pandas as pd

from .settings import StreamSettings

logger = logging.getLogger("producer")


def load_replay(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError(f"{path} must contain the true label column 'Class'")
    return df


def build_message(row: pd.Series, row_id: int) -> dict:
    msg = {k: float(v) for k, v in row.items() if k not in ("Class", "weight")}
    msg["label"] = int(row["Class"])      # ground truth, used by the consumer for precision/recall
    msg["row_id"] = int(row_id)
    msg["source"] = "replay_of_held_out_test_data"
    return msg


def connect_with_retries(factory, retries: int, base: float, what: str):
    for attempt in range(1, retries + 1):
        try:
            return factory()
        except Exception as exc:  # kafka raises several exception types
            wait = min(base * 2 ** (attempt - 1), 30)
            logger.warning("%s not reachable (%s); retry %d/%d in %.1fs", what, exc, attempt, retries, wait)
            time.sleep(wait)
    raise SystemExit(f"Could not connect to {what} after {retries} attempts")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = StreamSettings()
    from kafka import KafkaProducer

    df = load_replay(cfg.replay_file)
    producer = connect_with_retries(
        lambda: KafkaProducer(bootstrap_servers=[cfg.broker], acks="all", retries=3,
                              value_serializer=lambda v: json.dumps(v).encode("utf-8")),
        cfg.connect_retries, cfg.backoff_base_sec, f"Kafka at {cfg.broker}")
    logger.info("Replaying %d held-out rows to topic '%s' every %.2fs", len(df), cfg.topic, cfg.send_interval_sec)

    sent = 0
    try:
        while True:
            for i, row in df.iterrows():
                producer.send(cfg.topic, value=build_message(row, i))
                sent += 1
                if sent % 50 == 0:
                    producer.flush()
                    logger.info("sent %d messages", sent)
                if cfg.max_messages and sent >= cfg.max_messages:
                    raise StopIteration
                time.sleep(cfg.send_interval_sec)
            if not cfg.replay_loop:
                break
    except (KeyboardInterrupt, StopIteration):
        pass
    finally:
        producer.flush()
        producer.close()
        logger.info("Stopped after %d messages", sent)


if __name__ == "__main__":
    main()
