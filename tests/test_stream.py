import pytest
import requests

from stream.consumer import ScoringFailed, Tally, build_payload, handle_message, score_with_retry
from stream.producer import build_message
from stream.settings import StreamSettings

FEATURES = {**{f"V{i}": 0.1 for i in range(1, 29)}, "Amount": 10.0, "Time": 100.0}


class FakeResp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._b, self.headers, self.text = status, body or {}, headers or {}, str(body)

    def json(self):
        return self._b


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def post(self, *a, **k):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def cfg(**kw):
    return StreamSettings(api_url="http://x", api_key="k", score_retries=3, backoff_base_sec=0.0, **kw)


def test_tally_precision_recall():
    t = Tally()
    for flagged, label in [(True, 1), (True, 1), (True, 0), (False, 1), (False, 0), (False, 0)]:
        t.update(flagged, label)
    assert (t.tp, t.fp, t.fn, t.tn) == (2, 1, 1, 2)
    assert t.precision == pytest.approx(2 / 3) and t.recall == pytest.approx(2 / 3)
    assert "precision=0.667" in t.summary()


def test_build_payload_strips_extras_and_checks_fields():
    payload = build_payload({**FEATURES, "label": 1, "row_id": 3, "source": "x"})
    assert set(payload) == set(FEATURES)
    with pytest.raises(ScoringFailed) as e:
        build_payload({"V1": 1})
    assert not e.value.retryable


def test_retry_then_success():
    s = FakeSession([FakeResp(503), requests.ConnectionError(), FakeResp(200, {"is_fraud": False})])
    assert score_with_retry(s, cfg(), FEATURES, sleep=lambda _: None) == {"is_fraud": False}
    assert s.calls == 3


def test_429_honours_retry_after():
    waits = []
    s = FakeSession([FakeResp(429, headers={"Retry-After": "7"}), FakeResp(200, {"ok": 1})])
    score_with_retry(s, cfg(), FEATURES, sleep=waits.append)
    assert waits == [7.0]


def test_422_not_retried():
    s = FakeSession([FakeResp(422, {"detail": "bad"})])
    with pytest.raises(ScoringFailed) as e:
        score_with_retry(s, cfg(), FEATURES, sleep=lambda _: None)
    assert s.calls == 1 and not e.value.retryable


def test_gives_up_after_retries():
    s = FakeSession([FakeResp(500)] * 4)
    with pytest.raises(ScoringFailed) as e:
        score_with_retry(s, cfg(), FEATURES, sleep=lambda _: None)
    assert s.calls == 4 and e.value.retryable


def test_dead_letter_on_failure_and_no_crash_without_reasons():
    dlq, tally = [], Tally()
    bad = FakeSession([FakeResp(422, {"detail": "x"})])
    handle_message({**FEATURES, "label": 0, "row_id": 1}, bad, cfg(), tally, dlq.append)
    assert tally.dead_lettered == 1 and dlq[0]["message"]["row_id"] == 1

    ok = FakeSession([FakeResp(200, {"is_fraud": True, "transaction_id": "t", "fraud_probability": 0.9,
                                      "risk_level": "HIGH"})])          # no 'reasons' key at all
    handle_message({**FEATURES, "label": 1, "row_id": 2}, ok, cfg(), tally, dlq.append)
    assert tally.tp == 1


def test_producer_message_carries_label():
    import pandas as pd
    row = pd.Series({**FEATURES, "Class": 1})
    msg = build_message(row, 7)
    assert msg["label"] == 1 and msg["row_id"] == 7 and "Class" not in msg
