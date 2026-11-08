import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "500"))


@dataclass
class Settings:
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", ROOT / "model")))
    frontend_path: Path = field(default_factory=lambda: ROOT / "frontend" / "dashboard.html")
    api_keys: tuple = field(default_factory=lambda: tuple(
        k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()))
    auth_disabled: bool = field(default_factory=lambda: _bool("AUTH_DISABLED"))
    rate_limit_per_min: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MIN", "120")))
    max_batch_size: int = MAX_BATCH_SIZE
    allowed_origins: tuple = field(default_factory=lambda: tuple(
        o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",") if o.strip()))
    strict_versions: bool = field(default_factory=lambda: _bool("STRICT_VERSIONS"))
    drift_window: int = field(default_factory=lambda: int(os.getenv("DRIFT_WINDOW", "5000")))
    drift_min_samples: int = field(default_factory=lambda: int(os.getenv("DRIFT_MIN_SAMPLES", "200")))
    log_every: int = field(default_factory=lambda: int(os.getenv("LOG_EVERY_N_PREDICTIONS", "100")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
