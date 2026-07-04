"""
real_data.py
------------
Bridges REAL public CMS data into Aegis.

Right now it powers the billed-amount anomaly check with REAL price baselines
computed from the CMS "Medicare Physician & Other Practitioners - by Provider
and Service" dataset (public domain, US government works).

  - data/real/cms_charges_sample.csv : a small real sample shipped with the repo
                                       (so real baselines work out-of-the-box).
  - data/real/cms_charges_full.csv   : the fuller pull produced by fetch_real_data.py
                                       (preferred automatically when present).

For any code NOT covered by the real files, it falls back to the synthetic
baseline in policy_kb.json - so the app always runs, and coverage grows as you
add real data.
"""
import glob
import json
import os
import statistics
from functools import lru_cache

_ROOT = os.path.dirname(os.path.dirname(__file__))
_DATA = os.path.join(_ROOT, "data")
_REAL = os.path.join(_DATA, "real")

# CMS "average submitted charge" column (real dataset schema).
_CHARGE_COL = "Avg_Sbmtd_Chrg"
_CODE_COL = "HCPCS_Cd"


@lru_cache(maxsize=1)
def _synthetic_baselines():
    with open(os.path.join(_DATA, "policy_kb.json")) as f:
        return json.load(f)["amount_stats"]


def _read_csv(path):
    rows = []
    with open(path, newline="") as f:
        import csv
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


@lru_cache(maxsize=1)
def _real_rows():
    """Load real CMS charge rows, preferring the full pull over the shipped sample."""
    if not os.path.isdir(_REAL):
        return []
    full = os.path.join(_REAL, "cms_charges_full.csv")
    files = [full] if os.path.exists(full) else sorted(glob.glob(os.path.join(_REAL, "*.csv")))
    rows = []
    for path in files:
        try:
            rows.extend(_read_csv(path))
        except Exception:
            continue
    return rows


@lru_cache(maxsize=1)
def get_amount_baselines():
    """Return {cpt: {"mean","std","n","source"}} using REAL charges where available
    (n >= 2), otherwise the synthetic baseline."""
    by_code = {}
    for r in _real_rows():
        code = r.get(_CODE_COL)
        try:
            val = float(r.get(_CHARGE_COL, ""))
        except (TypeError, ValueError):
            continue
        by_code.setdefault(code, []).append(val)

    baselines = {}
    # real first
    for code, vals in by_code.items():
        if len(vals) >= 2:
            mean = statistics.fmean(vals)
            std = statistics.pstdev(vals) or max(1.0, 0.1 * mean)
            baselines[code] = {"mean": round(mean, 2), "std": round(std, 2),
                               "n": len(vals), "source": "CMS Medicare (real)"}
    # synthetic fallback for anything not covered
    for code, s in _synthetic_baselines().items():
        if code not in baselines:
            baselines[code] = {"mean": s["mean"], "std": s["std"],
                               "n": None, "source": "synthetic"}
    return baselines


def coverage_summary():
    b = get_amount_baselines()
    real = {k: v for k, v in b.items() if v["source"].startswith("CMS")}
    return {"codes_total": len(b), "codes_real": len(real),
            "real_codes": sorted(real.keys()),
            "real_rows_loaded": len(_real_rows())}
