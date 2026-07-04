"""
features.py
-----------
Shared engineering used by BOTH model training (build_models.py) and live
inference (ml.py), so the features are guaranteed identical in train and serve.

Also holds the synthetic-data generators for:
  - a large labelled claim set (to train the risk classifier)
  - a provider population with billing profiles (to cluster into peer groups)
  - per-provider daily time series (for temporal anomaly detection)

All synthetic. No PHI.
"""
import numpy as np

from .tools import load_kb

# Ordered feature vector for the risk model. Order is load-bearing (train==serve).
FEATURE_NAMES = [
    "total_billed", "n_lines", "n_distinct_cpts", "n_modifiers",
    "max_amount_z", "mean_amount_z", "has_ncci_pair",
    "is_high_level_em", "has_high_cost_proc", "billed_per_line",
    "proc_dx_mismatch",
]

HIGH_LEVEL_EM = {"99215", "99205"}
HIGH_COST_PROC = {"20610", "93307"}


def claim_features(claim):
    """Turn a claim dict into the model's feature vector (list, in FEATURE_NAMES order)."""
    kb = load_kb()
    stats = kb["amount_stats"]
    lines = claim["lines"]
    cpts = [l["cpt"] for l in lines]
    amounts = [float(l["billed_amount"]) for l in lines]

    zs = []
    for l in lines:
        s = stats.get(l["cpt"])
        if s and s["std"]:
            zs.append(abs((l["billed_amount"] - s["mean"]) / s["std"]))
    max_z = max(zs) if zs else 0.0
    mean_z = float(np.mean(zs)) if zs else 0.0

    cset = set(cpts)
    has_ncci = any(e["column1"] in cset and e["column2"] in cset for e in kb["ncci_edits"])
    n_lines = len(lines)
    total = sum(amounts)

    # diagnosis-support mismatch: a procedure with a medical-necessity policy billed
    # without any supported diagnosis (a genuine predictive signal for denials).
    mn = {r["cpt"]: r["supported_icd10_prefixes"] for r in kb["medical_necessity"]}
    mismatch = 0.0
    for l in lines:
        prefixes = mn.get(l["cpt"])
        if prefixes:
            ok = any(dx.replace(".", "").upper().startswith(p.replace(".", "").upper())
                     for dx in l["icd10"] for p in prefixes)
            if not ok:
                mismatch = 1.0

    return [
        total,
        n_lines,
        len(cset),
        sum(len(l["modifiers"]) for l in lines),
        max_z,
        mean_z,
        1.0 if has_ncci else 0.0,
        1.0 if cset & HIGH_LEVEL_EM else 0.0,
        1.0 if cset & HIGH_COST_PROC else 0.0,
        total / n_lines if n_lines else 0.0,
        mismatch,
    ]


# --------------------------------------------------------------------------- #
# Synthetic labelled claims for training the risk model
# --------------------------------------------------------------------------- #
_BENIGN_DX = ["E11.9", "I10", "E78.5", "K21.9", "J45.909"]
_PROC_SUPPORTED = {"20610": ["M17.0", "M25.561", "M19.90"],
                   "93307": ["I50.9", "I25.10", "I48.91"]}
_LAB_EM = ["99213", "99214", "80053", "80048", "36415", "85025"]


def generate_labeled_claims(n=1600, seed=7):
    """Generate n synthetic claims with a realistic (noisy) 'problematic' label.

    Issues are injected probabilistically; label noise + legitimately-high-cost
    clean claims create overlap so the model can't separate perfectly (AUC ~0.9,
    not 1.0), which is what a believable claims-risk model looks like.
    """
    rng = np.random.default_rng(seed)
    kb = load_kb()
    stats = kb["amount_stats"]
    X, y = [], []

    def amt(cpt):
        s = stats[cpt]
        return max(1.0, float(rng.normal(s["mean"], s["std"])))

    for _ in range(n):
        n_lines = int(rng.choice([1, 1, 2, 2, 3]))
        cpts = list(rng.choice(_LAB_EM, size=n_lines, replace=True))
        lines = [{"cpt": c, "modifiers": [], "icd10": [rng.choice(_BENIGN_DX)],
                  "billed_amount": amt(c)} for c in cpts]
        problematic = False

        r = rng.random()
        if r < 0.15:                                   # unbundling
            lines = [{"cpt": "80053", "modifiers": [], "icd10": ["E11.9"], "billed_amount": amt("80053")},
                     {"cpt": "80048", "modifiers": [], "icd10": ["E11.9"], "billed_amount": amt("80048")}]
            problematic = True
        elif r < 0.30:                                 # amount anomaly
            lines[0]["billed_amount"] *= float(rng.uniform(3.0, 6.0))
            problematic = True
        elif r < 0.42:                                 # medical necessity fail
            proc = rng.choice(list(_PROC_SUPPORTED))
            lines[0] = {"cpt": proc, "modifiers": [], "icd10": [rng.choice(["J06.9", "Z00.00"])],
                        "billed_amount": amt(proc)}
            problematic = True
        elif r < 0.52:                                 # upcoding
            lines[0] = {"cpt": "99215", "modifiers": [], "icd10": ["Z00.00"], "billed_amount": amt("99215")}
            problematic = True
        else:
            # clean, but sometimes legitimately pricey -> overlap with anomalies
            if rng.random() < 0.12:
                lines[0]["billed_amount"] *= float(rng.uniform(1.8, 2.6))
            # sometimes a high-cost procedure that IS supported, so the model learns
            # dx-support matters (not merely that a high-cost code is present)
            if rng.random() < 0.18:
                proc = rng.choice(list(_PROC_SUPPORTED))
                lines[0] = {"cpt": proc, "modifiers": [],
                            "icd10": [rng.choice(_PROC_SUPPORTED[proc])], "billed_amount": amt(proc)}

        claim = {"lines": lines}
        label = 1 if problematic else 0
        if rng.random() < 0.10:                        # label noise
            label = 1 - label
        X.append(claim_features(claim))
        y.append(label)

    return np.array(X, dtype=float), np.array(y, dtype=int)


# --------------------------------------------------------------------------- #
# Provider population (for clustering into peer groups)
# --------------------------------------------------------------------------- #
PROVIDER_FEATURES = ["avg_billed", "claim_volume", "pct_high_em", "pct_procedures", "avg_lines"]

# Four normal billing archetypes (centers) + spread.
_ARCHETYPES = {
    "Primary care":  {"avg_billed": 140, "claim_volume": 300, "pct_high_em": 0.05, "pct_procedures": 0.10, "avg_lines": 2.2},
    "Specialist":    {"avg_billed": 320, "claim_volume": 150, "pct_high_em": 0.15, "pct_procedures": 0.40, "avg_lines": 1.5},
    "Lab / imaging": {"avg_billed": 45,  "claim_volume": 600, "pct_high_em": 0.01, "pct_procedures": 0.05, "avg_lines": 3.0},
    "Mixed group":   {"avg_billed": 205, "claim_volume": 250, "pct_high_em": 0.10, "pct_procedures": 0.25, "avg_lines": 2.0},
}

# The demo claims' providers, given controlled profiles so the tools return
# meaningful results on camera. P-213 and P-217 are deliberate outliers.
_DEMO_PROVIDERS = {
    "P-210": "Primary care", "P-211": "Primary care", "P-212": "Specialist",
    "P-214": "Primary care", "P-215": "Specialist", "P-216": "Specialist",
    "P-500": "Primary care",
    "P-213": {"avg_billed": 430, "claim_volume": 520, "pct_high_em": 0.46, "pct_procedures": 0.20, "avg_lines": 1.4},  # outlier + will spike
    "P-217": {"avg_billed": 505, "claim_volume": 180, "pct_high_em": 0.30, "pct_procedures": 0.62, "avg_lines": 1.3},  # outlier
}


def generate_eval_claims(n=300, seed=99):
    """Full synthetic claims + ground-truth 'problematic' labels, for evaluating the
    WHOLE pipeline (rules + ML + governance) end-to-end, not just the model.
    Providers are outside the demo population so provider tools stay neutral."""
    rng = np.random.default_rng(seed)
    kb = load_kb()
    stats = kb["amount_stats"]

    def amt(cpt):
        s = stats[cpt]
        return round(max(1.0, float(rng.normal(s["mean"], s["std"]))), 2)

    out = []
    for i in range(n):
        n_lines = int(rng.choice([1, 1, 2, 2, 3]))
        cpts = list(rng.choice(_LAB_EM, size=n_lines, replace=True))
        lines = [{"cpt": c, "modifiers": [], "icd10": [str(rng.choice(_BENIGN_DX))],
                  "billed_amount": amt(c)} for c in cpts]
        problematic = False
        r = rng.random()
        if r < 0.15:
            lines = [{"cpt": "80053", "modifiers": [], "icd10": ["E11.9"], "billed_amount": amt("80053")},
                     {"cpt": "80048", "modifiers": [], "icd10": ["E11.9"], "billed_amount": amt("80048")}]
            problematic = True
        elif r < 0.30:
            lines[0]["billed_amount"] = round(lines[0]["billed_amount"] * float(rng.uniform(3, 6)), 2)
            problematic = True
        elif r < 0.42:
            proc = str(rng.choice(list(_PROC_SUPPORTED)))
            lines[0] = {"cpt": proc, "modifiers": [], "icd10": [str(rng.choice(["J06.9", "Z00.00"]))],
                        "billed_amount": amt(proc)}
            problematic = True
        elif r < 0.52:
            lines[0] = {"cpt": "99215", "modifiers": [], "icd10": ["Z00.00"], "billed_amount": amt("99215")}
            problematic = True
        else:
            if rng.random() < 0.12:
                lines[0]["billed_amount"] = round(lines[0]["billed_amount"] * float(rng.uniform(1.8, 2.6)), 2)
            if rng.random() < 0.18:
                proc = str(rng.choice(list(_PROC_SUPPORTED)))
                lines[0] = {"cpt": proc, "modifiers": [], "icd10": [str(rng.choice(_PROC_SUPPORTED[proc]))],
                            "billed_amount": amt(proc)}
        label = 1 if problematic else 0
        if rng.random() < 0.10:
            label = 1 - label
        claim = {"claim_id": f"EVAL-{i:04d}", "member_id": f"M-{i:05d}",
                 "provider_id": f"P-E{i % 40}", "npi": "0000000000",
                 "date_of_service": "2026-06-15", "place_of_service": "11",
                 "lines": lines, "total_billed": round(sum(l["billed_amount"] for l in lines), 2)}
        out.append((claim, label))
    return out


# --------------------------------------------------------------------------- #
# Provider population (for clustering into peer groups)
# --------------------------------------------------------------------------- #
def generate_provider_population(seed=11):
    rng = np.random.default_rng(seed)
    providers = []

    def jitter(center):
        return {
            "avg_billed": max(10, rng.normal(center["avg_billed"], center["avg_billed"] * 0.12)),
            "claim_volume": max(20, rng.normal(center["claim_volume"], center["claim_volume"] * 0.15)),
            "pct_high_em": float(np.clip(rng.normal(center["pct_high_em"], 0.02), 0, 1)),
            "pct_procedures": float(np.clip(rng.normal(center["pct_procedures"], 0.04), 0, 1)),
            "avg_lines": max(1.0, rng.normal(center["avg_lines"], 0.2)),
        }

    # demo providers first (fixed identities)
    for pid, spec in _DEMO_PROVIDERS.items():
        center = _ARCHETYPES[spec] if isinstance(spec, str) else spec
        f = jitter(center) if isinstance(spec, str) else dict(spec)
        providers.append({"provider_id": pid, **f})

    # fill out the population with normal providers across archetypes
    names = list(_ARCHETYPES)
    for i in range(50):
        arch = names[i % len(names)]
        providers.append({"provider_id": f"P-{600 + i}", **jitter(_ARCHETYPES[arch])})

    # a few extra random outliers so the outlier band isn't only demo providers
    for j in range(3):
        providers.append({"provider_id": f"P-9{j}0",
                          "avg_billed": rng.uniform(550, 720), "claim_volume": rng.uniform(400, 700),
                          "pct_high_em": rng.uniform(0.4, 0.6), "pct_procedures": rng.uniform(0.5, 0.75),
                          "avg_lines": rng.uniform(1.2, 1.6)})
    return providers


# --------------------------------------------------------------------------- #
# Per-provider daily time series (for temporal anomaly detection)
# --------------------------------------------------------------------------- #
def generate_provider_timeseries(providers, days=90, spike_providers=("P-213", "P-91 0"), seed=13):
    rng = np.random.default_rng(seed)
    import datetime
    start = datetime.date(2026, 4, 1)
    dates = [(start + datetime.timedelta(d)).isoformat() for d in range(days)]
    series = {}
    spike_set = {"P-213", "P-910"}
    for p in providers:
        pid = p["provider_id"]
        base = max(1.0, p["claim_volume"] / days)
        lam = np.full(days, base)
        if pid in spike_set:                     # inject a recent change-point (last 10 days)
            lam[-10:] = base * 3.2
        counts = rng.poisson(lam).astype(int)
        series[pid] = {"dates": dates, "counts": counts.tolist()}
    return series
