import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sha256_file(path, chunk=1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()
