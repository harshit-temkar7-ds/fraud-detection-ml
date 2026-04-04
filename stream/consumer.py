"""
Kafka Consumer — Reads live transactions and scores them via FastAPI
Logs fraud alerts in real time
"""

import json
import time
import requests
from kafka import KafkaConsumer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

KAFKA_BROKER = "localhost:9092"
TOPIC        = "transactions"
API_URL      = "http://localhost:8000/predict"
GROUP_ID     = "fraud-detection-group"


def score_transaction(txn: dict) -> dict | None:
    """Call FastAPI scoring endpoint for a single transaction"""
    payload = {k: v for k, v in txn.items()
               if k.startswith("V") or k == "Amount"}
    try:
        resp = requests.post(API_URL, json=payload, timeout=2)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None


def main():
    logger.info(f"Connecting to Kafka broker at {KAFKA_BROKER}...")
    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            group_id=GROUP_ID,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True
        )
        logger.info(f"✅ Listening on topic: [{TOPIC}]")
    except Exception as e:
        logger.error(f"❌ Cannot connect to Kafka: {e}")
        return

    total = fraud = 0
    try:
        for msg in consumer:
            txn = msg.value
            result = score_transaction(txn)

            total += 1
            if result and result.get("is_fraud"):
                fraud += 1
                logger.warning(
                    f"🚨 FRAUD ALERT! | TXN: {result['transaction_id']} "
                    f"| Prob: {result['fraud_probability']:.1%} "
                    f"| Risk: {result['risk_level']} "
                    f"| Amount: ${txn.get('Amount', 0):.2f}"
                )
                logger.warning(
                    f"   Top factor: {result['top_risk_factors'][0]['feature']} "
                    f"(SHAP={result['top_risk_factors'][0]['shap_value']})"
                )
            else:
                if total % 10 == 0:  # log every 10th legit
                    logger.info(f"✅ LEGIT | Total processed: {total} | Frauds caught: {fraud}")

    except KeyboardInterrupt:
        logger.info(f"\n⛔ Stopped. Total={total} | Frauds={fraud} | Rate={fraud/max(total,1)*100:.1f}%")
        consumer.close()


if __name__ == "__main__":
    main()
