"""
Kafka Producer — Simulates a real-time transaction stream
Sends random transactions (some seeded as fraud) to the Kafka topic 'transactions'
"""

import json
import time
import random
import numpy as np
from kafka import KafkaProducer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

KAFKA_BROKER  = "localhost:9092"
TOPIC         = "transactions"
SEND_RATE_SEC = 0.5    # one transaction every 0.5s
FRAUD_RATE    = 0.05   # 5% of messages are fraud-like


def generate_transaction(force_fraud: bool = False) -> dict:
    """Generate a synthetic transaction matching creditcard.csv schema"""
    np.random.seed(None)

    if force_fraud:
        # Fraud fingerprint — V1 very negative, high V3
        txn = {f"V{i}": np.random.randn() for i in range(1, 29)}
        txn["V1"]  -= random.uniform(2, 5)
        txn["V3"]  += random.uniform(2, 4)
        txn["V14"] -= random.uniform(2, 4)
        txn["Amount"] = random.uniform(1, 50)   # small amounts typical in fraud
    else:
        txn = {f"V{i}": np.random.randn() for i in range(1, 29)}
        txn["Amount"] = abs(np.random.exponential(88))

    # Round to 4 dp
    txn = {k: round(float(v), 4) for k, v in txn.items()}
    txn["timestamp"] = time.time()
    txn["is_simulated_fraud"] = force_fraud  # ground-truth label for evaluation
    return txn


def on_send_success(metadata):
    logger.info(f"✅ Sent to [{metadata.topic}] partition={metadata.partition} offset={metadata.offset}")


def on_send_error(exc):
    logger.error(f"❌ Send failed: {exc}")


def main():
    logger.info(f"Connecting to Kafka broker at {KAFKA_BROKER}...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3
        )
        logger.info(f"✅ Connected! Streaming to topic: [{TOPIC}]")
    except Exception as e:
        logger.error(f"❌ Cannot connect to Kafka: {e}")
        logger.info("💡 Start Kafka first:\n   docker-compose up kafka zookeeper")
        return

    count = 0
    try:
        while True:
            is_fraud = random.random() < FRAUD_RATE
            txn = generate_transaction(force_fraud=is_fraud)

            producer.send(TOPIC, value=txn).add_callback(on_send_success).add_errback(on_send_error)
            producer.flush()

            count += 1
            tag = "🚨 FRAUD" if is_fraud else "✅ LEGIT"
            logger.info(f"[#{count}] {tag} | Amount=${txn['Amount']:.2f}")

            time.sleep(SEND_RATE_SEC)

    except KeyboardInterrupt:
        logger.info(f"\n⛔ Stopped. Total sent: {count}")
        producer.close()


if __name__ == "__main__":
    main()
