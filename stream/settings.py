import os
from dataclasses import dataclass, field


@dataclass
class StreamSettings:
    broker: str = field(default_factory=lambda: os.getenv("KAFKA_BROKER", "localhost:9092"))
    topic: str = field(default_factory=lambda: os.getenv("KAFKA_TOPIC", "transactions"))
    dlq_topic: str = field(default_factory=lambda: os.getenv("KAFKA_DLQ_TOPIC", "transactions.dlq"))
    offset_reset: str = field(default_factory=lambda: os.getenv("KAFKA_OFFSET_RESET", "earliest"))
    group_id: str = field(default_factory=lambda: os.getenv("KAFKA_GROUP_ID", "fraud-detection-group"))
    api_url: str = field(default_factory=lambda: os.getenv("API_URL", "http://localhost:8000"))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    replay_file: str = field(default_factory=lambda: os.getenv("REPLAY_FILE", "model/replay_sample.csv"))
    send_interval_sec: float = field(default_factory=lambda: float(os.getenv("SEND_INTERVAL_SEC", "0.5")))
    replay_loop: bool = field(default_factory=lambda: os.getenv("REPLAY_LOOP", "true").lower() == "true")
    max_messages: int = field(default_factory=lambda: int(os.getenv("MAX_MESSAGES", "0")))  # 0 = unlimited
    connect_retries: int = field(default_factory=lambda: int(os.getenv("CONNECT_RETRIES", "10")))
    score_retries: int = field(default_factory=lambda: int(os.getenv("SCORE_RETRIES", "4")))
    backoff_base_sec: float = field(default_factory=lambda: float(os.getenv("BACKOFF_BASE_SEC", "0.5")))
    summary_every: int = field(default_factory=lambda: int(os.getenv("SUMMARY_EVERY", "50")))
