import copy

from tests.conftest import HEADERS, KEY


def test_missing_field_is_422(client, legit_row):
    body = copy.deepcopy(legit_row)
    body.pop("V5")
    assert client.post("/predict", json=body, headers=HEADERS).status_code == 422


def test_empty_body_is_422_not_silent_zeros(client):
    assert client.post("/predict", json={}, headers=HEADERS).status_code == 422


def test_extra_field_is_422(client, legit_row):
    assert client.post("/predict", json={**legit_row, "Class": 1}, headers=HEADERS).status_code == 422


def test_out_of_range_and_negative_amount(client, legit_row):
    assert client.post("/predict", json={**legit_row, "V3": 1e6}, headers=HEADERS).status_code == 422
    assert client.post("/predict", json={**legit_row, "Amount": -5}, headers=HEADERS).status_code == 422


def test_nan_rejected(client, legit_row):
    raw = str({**legit_row, "V1": "NaN"}).replace("'", '"').replace('"NaN"', "NaN")
    r = client.post("/predict", content=raw, headers={**HEADERS, "Content-Type": "application/json"})
    assert r.status_code == 422


def test_batch_limits(client, legit_row):
    assert client.post("/predict/batch", json={"transactions": []}, headers=HEADERS).status_code == 422
    too_many = {"transactions": [legit_row] * 501}
    assert client.post("/predict/batch", json=too_many, headers=HEADERS).status_code == 422


def test_auth_required_and_wrong_key(client, legit_row):
    assert client.post("/predict", json=legit_row).status_code == 401
    assert client.post("/predict", json=legit_row, headers={"X-API-Key": "nope"}).status_code == 401
    assert client.get("/stats/summary").status_code == 401


def test_probes_are_open(client):
    assert client.get("/live").status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.get("/health").status_code == 200


def test_auth_not_configured_is_503(make_client, legit_row):
    c = make_client(api_keys=())
    assert c.post("/predict", json=legit_row, headers={"X-API-Key": KEY}).status_code == 503


def test_rate_limit(make_client, legit_row):
    c = make_client(rate_limit_per_min=3)
    codes = [c.post("/predict", json=legit_row, headers=HEADERS).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429
    r = c.post("/predict", json=legit_row, headers=HEADERS)
    assert r.status_code == 429 and "retry-after" in r.headers
