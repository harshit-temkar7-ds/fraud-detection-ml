"""Turn TreeSHAP contributions into human-readable reason codes.

Honest scope: V1..V28 are anonymised PCA components, so a reason such as
"V14 is unusually low" explains the *model*, not a real-world cause.
"""

GROUPS = {"hour_sin": "time_of_day", "hour_cos": "time_of_day", "log_amount": "amount"}

DISCLAIMER = (
    "Reason codes are model attributions (TreeSHAP, log-odds scale) over anonymised "
    "features. They describe what drove the model's score, not the real-world cause of fraud."
)


def build_reasons(feature_names, contributions, feature_values, feature_stats, top_k=3, hour=None):
    """contributions/feature_values: 1-D arrays aligned with feature_names."""
    grouped = {}
    for name, c in zip(feature_names, contributions):
        key = GROUPS.get(name, name)
        grouped.setdefault(key, []).append((name, float(c)))

    scored = []
    for key, items in grouped.items():
        total = sum(c for _, c in items)
        if total > 0:
            scored.append((key, total, items))
    scored.sort(key=lambda r: r[1], reverse=True)

    values = dict(zip(feature_names, feature_values))
    reasons = []
    for key, total, items in scored[:top_k]:
        if key == "time_of_day":
            hr = "unknown" if hour is None else f"{int(hour):02d}:00"
            reasons.append({
                "code": "R_TIME_OF_DAY", "feature": "Time", "direction": None,
                "contribution": round(total, 4),
                "description": f"Transaction hour ({hr}) is associated with higher fraud risk by the model",
            })
            continue
        name = items[0][0]
        stats = feature_stats.get(name, {})
        std = stats.get("std") or 1.0
        z = (values[name] - stats.get("mean", 0.0)) / std
        direction = "HIGH" if z >= 0 else "LOW"
        if key == "amount":
            desc = f"Transaction amount is unusually {direction.lower()} (z={z:+.1f})"
            code, feat = f"R_AMOUNT_{direction}", "Amount"
        else:
            desc = f"Anonymised component {name} is {direction.lower()} vs. typical (z={z:+.1f})"
            code, feat = f"R_{name}_{direction}", name
        reasons.append({"code": code, "feature": feat, "direction": direction.lower(),
                        "contribution": round(total, 4), "description": desc})
    return reasons
