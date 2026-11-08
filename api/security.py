"""API-key auth, in-memory sliding-window rate limiting, request-ID logging."""

import contextvars
import hashlib
import logging
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import Depends, Header, HTTPException, Request

request_id_var = contextvars.ContextVar("request_id", default="-")
_RID_OK = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str):
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)


def new_request_id(incoming):
    return incoming if incoming and _RID_OK.match(incoming) else uuid.uuid4().hex[:16]


class SlidingWindowLimiter:
    """Per-identity limiter. In-memory => per process; use a shared store for multi-replica deployments."""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit, self.window = limit, window
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, identity: str):
        """Returns (allowed, retry_after_seconds)."""
        if self.limit <= 0:
            return True, 0
        now = time.monotonic()
        with self._lock:
            q = self._hits[identity]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, max(1, int(self.window - (now - q[0])) + 1)
            q.append(now)
            if len(self._hits) > 10_000:  # drop idle identities
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > self.window]:
                    self._hits.pop(k, None)
            return True, 0


def _identity(key, request: Request) -> str:
    if key:
        return "key:" + hashlib.sha256(key.encode()).hexdigest()[:12]
    return "ip:" + (request.client.host if request.client else "unknown")


def require_access(request: Request, x_api_key: str = Header(default=None, alias="X-API-Key")):
    """Dependency: authenticate (unless AUTH_DISABLED) then rate-limit."""
    settings = request.app.state.settings
    key = None
    if not settings.auth_disabled:
        if not settings.api_keys:
            raise HTTPException(503, "API authentication is not configured (set API_KEYS)")
        if not x_api_key or not any(secrets.compare_digest(x_api_key, k) for k in settings.api_keys):
            raise HTTPException(401, "Invalid or missing API key", headers={"WWW-Authenticate": "ApiKey"})
        key = x_api_key
    allowed, retry = request.app.state.limiter.check(_identity(key, request))
    if not allowed:
        raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": str(retry)})
    return key


AccessDep = Depends(require_access)
