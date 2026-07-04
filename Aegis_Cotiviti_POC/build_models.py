"""
build_models.py
---------------
One-shot builder. Run once (or to refresh):  python build_models.py

Produces:
  models/risk_model.joblib      - trained GradientBoosting risk classifier
  models/risk_meta.json         - AUC/accuracy/feature importances (for the UI)
  data/providers.json           - provider features + KMeans peer cluster + outlier flag + 2D coords
  data/provider_timeseries.json - per-provider daily claim counts
"""
import json
import os

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from aegis.features import (FEATURE_NAMES, PROVIDER_FEATURES, claim_features,
                            generate_labeled_claims, generate_provider_population,
                            generate_provider_timeseries)

ROOT = os.path.dirname(__file__)
MODELS = os.path.join(ROOT, "models"); os.makedirs(MODELS, exist_ok=True)
DATA = os.path.join(ROOT, "data")

# ----------------------------------------------------- 1) risk classifier ----
print("Training claim-risk classifier...")
X, y = generate_labeled_claims(1600)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
clf = GradientBoostingClassifier(random_state=0)
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)[:, 1]
auc = roc_auc_score(yte, proba)
acc = accuracy_score(yte, clf.predict(Xte))
joblib.dump(clf, os.path.join(MODELS, "risk_model.joblib"))
meta = {"auc": round(float(auc), 3), "accuracy": round(float(acc), 3),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "positive_rate": round(float(y.mean()), 3),
        "feature_names": FEATURE_NAMES,
        "importances": [round(float(i), 3) for i in clf.feature_importances_]}
json.dump(meta, open(os.path.join(MODELS, "risk_meta.json"), "w"), indent=2)
print(f"  risk model: AUC={auc:.3f}  accuracy={acc:.3f}  (train {len(ytr)}, test {len(yte)})")

# ----------------------------------------------------- 2) peer clustering ----
print("Fitting provider peer-group clustering (KMeans)...")
providers = generate_provider_population()
M = np.array([[p[f] for f in PROVIDER_FEATURES] for p in providers], dtype=float)
scaler = StandardScaler().fit(M)
Ms = scaler.transform(M)
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Ms)
labels = km.labels_
dist = np.linalg.norm(Ms - km.cluster_centers_[labels], axis=1)   # distance to own centroid
thresh = float(dist.mean() + 2 * dist.std())                       # peer-relative outlier band
coords = PCA(n_components=2, random_state=0).fit_transform(Ms)     # 2D for the scatter plot

# human-readable cluster names by nearest archetype centroid feel (dominant trait)
def cluster_name(center_scaled):
    c = scaler.inverse_transform(center_scaled.reshape(1, -1))[0]
    d = dict(zip(PROVIDER_FEATURES, c))
    if d["pct_procedures"] > 0.32:
        return "Procedure-heavy"
    if d["claim_volume"] > 450:
        return "High-volume lab"
    if d["avg_billed"] > 190:
        return "High-charge"
    return "Primary care"

# guarantee unique display names even if two centroids share a trait
_raw = {i: cluster_name(km.cluster_centers_[i]) for i in range(4)}
_seen, cnames = {}, {}
for i in range(4):
    base = _raw[i]
    _seen[base] = _seen.get(base, 0) + 1
    cnames[i] = base if _seen[base] == 1 else f"{base} {_seen[base]}"
for i, p in enumerate(providers):
    p["cluster"] = int(labels[i])
    p["cluster_name"] = cnames[int(labels[i])]
    p["centroid_distance"] = round(float(dist[i]), 3)
    p["is_outlier"] = bool(dist[i] > thresh)
    p["viz_x"] = round(float(coords[i, 0]), 3)
    p["viz_y"] = round(float(coords[i, 1]), 3)

joblib.dump({"kmeans": km, "scaler": scaler, "threshold": thresh, "cluster_names": cnames},
            os.path.join(MODELS, "provider_kmeans.joblib"))
json.dump({"threshold": round(thresh, 3), "providers": providers},
          open(os.path.join(DATA, "providers.json"), "w"), indent=2)
outliers = [p["provider_id"] for p in providers if p["is_outlier"]]
print(f"  clusters: { {cnames[i]: int((labels==i).sum()) for i in range(4)} }")
print(f"  peer outliers ({len(outliers)}): {outliers}")

# ----------------------------------------------------- 3) time series --------
print("Generating provider daily time series...")
ts = generate_provider_timeseries(providers)
json.dump(ts, open(os.path.join(DATA, "provider_timeseries.json"), "w"))
print(f"  wrote {len(ts)} provider series (90 days each)")
print("Done. Artifacts in models/ and data/.")
