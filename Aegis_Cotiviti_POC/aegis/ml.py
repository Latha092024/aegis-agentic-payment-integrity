"""
ml.py
-----
The three ML-backed tools that complete the Topic-2 technique list. Each returns
the same result-dict shape as the rule tools so they drop straight into the
agent's trace and reasoning.

  predict_claim_risk       -> PREDICTION      (trained GradientBoosting classifier)
  provider_peer_outlier    -> CLUSTERING      (KMeans peer groups + distance outlier)
  detect_temporal_anomaly  -> TIME-SERIES     (rolling z-score on daily claim volume)
"""
import json
import os
from functools import lru_cache

import joblib
import numpy as np

from .features import claim_features

_ROOT = os.path.dirname(os.path.dirname(__file__))
_MODELS = os.path.join(_ROOT, "models")
_DATA = os.path.join(_ROOT, "data")


@lru_cache(maxsize=1)
def _risk():
    model = joblib.load(os.path.join(_MODELS, "risk_model.joblib"))
    meta = json.load(open(os.path.join(_MODELS, "risk_meta.json")))
    return model, meta


@lru_cache(maxsize=1)
def _providers():
    d = json.load(open(os.path.join(_DATA, "providers.json")))
    return {p["provider_id"]: p for p in d["providers"]}


@lru_cache(maxsize=1)
def _timeseries():
    return json.load(open(os.path.join(_DATA, "provider_timeseries.json")))


# --------------------------------------------------------------- PREDICTION --
def predict_claim_risk(claim):
    model, meta = _risk()
    x = np.array([claim_features(claim)], dtype=float)
    prob = float(model.predict_proba(x)[0, 1])
    level = "high" if prob >= 0.66 else ("elevated" if prob >= 0.4 else "low")
    return {
        "tool": "predict_claim_risk",
        "finding": "anomaly" if prob >= 0.66 else ("info" if prob >= 0.4 else "clean"),
        "detail": (f"Predicted problematic-claim probability {prob:.0%} ({level} risk), "
                   f"from a gradient-boosted model (holdout AUC {meta['auc']})."),
        "rule_id": "ML-RISK",
        "reference": f"Gradient-boosted risk classifier (AUC {meta['auc']}, n={meta['n_train']+meta['n_test']})",
        "data": {"probability": round(prob, 3), "level": level, "auc": meta["auc"]},
    }


# --------------------------------------------------------------- CLUSTERING --
def provider_peer_outlier(provider_id):
    prov = _providers().get(provider_id)
    if not prov:
        return {"tool": "provider_peer_outlier", "finding": "info",
                "detail": f"No peer-group profile on file for provider {provider_id}.",
                "rule_id": None, "reference": None, "data": {"provider_id": provider_id}}
    is_out = prov["is_outlier"]
    return {
        "tool": "provider_peer_outlier",
        "finding": "anomaly" if is_out else "clean",
        "detail": (f"Provider {provider_id} sits in peer group '{prov['cluster_name']}' "
                   + ("and is a peer-relative OUTLIER (distance "
                      f"{prov['centroid_distance']} from the group centroid) - billing profile "
                      "diverges sharply from similar providers."
                      if is_out else
                      f"with a typical profile (distance {prov['centroid_distance']} from centroid).")),
        "rule_id": "CLUSTER-OUTLIER",
        "reference": "KMeans peer-group clustering (distance-to-centroid outlier)",
        "data": {"provider_id": provider_id, "cluster": prov["cluster_name"],
                 "distance": prov["centroid_distance"], "is_outlier": is_out},
    }


# --------------------------------------------------------------- TIME-SERIES -
def detect_temporal_anomaly(provider_id, window=30, z_threshold=3.0, recent=7):
    ts = _timeseries().get(provider_id)
    if not ts:
        return {"tool": "detect_temporal_anomaly", "finding": "info",
                "detail": f"No time series on file for provider {provider_id}.",
                "rule_id": None, "reference": None, "data": {"provider_id": provider_id}}
    counts = np.array(ts["counts"], dtype=float)
    dates = ts["dates"]
    baseline = counts[:-recent]                       # history excluding the recent window
    mu, sd = baseline[-window:].mean(), baseline[-window:].std() or 1.0
    recent_z = (counts[-recent:] - mu) / sd
    i = int(np.argmax(recent_z))
    peak_z = float(recent_z[i])
    is_anom = peak_z >= z_threshold
    return {
        "tool": "detect_temporal_anomaly",
        "finding": "anomaly" if is_anom else "clean",
        "detail": (f"Daily claim volume for {provider_id}: recent peak on {dates[-recent + i]} "
                   f"reached {int(counts[-recent + i])} vs baseline mean {mu:.1f} "
                   f"(z-score {peak_z:+.1f}). "
                   + ("Temporal spike exceeds the anomaly threshold - abrupt change in billing behavior."
                      if is_anom else "No temporal spike detected.")),
        "rule_id": "TS-ANOM",
        "reference": "Rolling z-score temporal anomaly detection",
        "data": {"provider_id": provider_id, "peak_z": round(peak_z, 2),
                 "baseline_mean": round(float(mu), 2), "is_anomaly": is_anom,
                 "peak_date": dates[-recent + i]},
    }
