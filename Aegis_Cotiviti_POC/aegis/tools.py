"""
tools.py
--------
The "hands" of the agent. Each function is a self-contained payment-integrity
check that reads the policy knowledge base and returns a structured, auditable
result (a plain dict). The agent (LLM or rule-based) decides *which* of these to
call and in *what order*; these functions never make the final decision.

Every result dict follows a common shape:
    {
        "tool": <tool name>,
        "finding": "violation" | "anomaly" | "clean" | "info",
        "detail": <human-readable explanation>,
        "rule_id": <id or None>,
        "reference": <policy citation or None>,
        "data": {...}          # raw values for the audit trail
    }
"""

import json
import math
import os
from functools import lru_cache

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@lru_cache(maxsize=1)
def load_kb():
    with open(os.path.join(_DATA_DIR, "policy_kb.json")) as f:
        return json.load(f)


def _icd_matches(icd10_code, prefixes):
    code = icd10_code.replace(".", "").upper()
    return any(code.startswith(p.replace(".", "").upper()) for p in prefixes)


# --------------------------------------------------------------------------- #
# Tool 1: NCCI unbundling / procedure-to-procedure edits
# --------------------------------------------------------------------------- #
def check_unbundling(cpt_codes, modifiers_by_code=None):
    """Detect NCCI column1/column2 edits among the CPT codes on a claim.

    modifiers_by_code: optional {cpt: [modifiers]} used to evaluate whether an
    approved distinct-service modifier legitimately bypasses an edit.
    """
    kb = load_kb()
    modifiers_by_code = modifiers_by_code or {}
    codes = set(cpt_codes)
    bypass_mods = {"59", "XE", "XS", "XP", "XU"}

    for edit in kb["ncci_edits"]:
        c1, c2 = edit["column1"], edit["column2"]
        if c1 in codes and c2 in codes:
            applied = set(modifiers_by_code.get(c2, [])) | set(modifiers_by_code.get(c1, []))
            has_bypass = bool(applied & bypass_mods)
            if edit["modifier_bypass_allowed"] and has_bypass:
                return {
                    "tool": "check_unbundling",
                    "finding": "clean",
                    "detail": (f"Edit between {c1} and {c2} is present, but a distinct-service "
                               f"modifier ({', '.join(sorted(applied & bypass_mods))}) is applied and "
                               f"permitted for this pair, so separate payment is allowed."),
                    "rule_id": edit["rule_id"],
                    "reference": edit["reference"],
                    "data": {"column1": c1, "column2": c2, "modifiers_applied": sorted(applied),
                             "bypass_allowed": True},
                }
            return {
                "tool": "check_unbundling",
                "finding": "violation",
                "detail": (f"{c2} is a component of {c1} and is not separately payable "
                           f"({'no' if not has_bypass else 'no permitted'} distinct-service modifier). "
                           f"{edit['rationale']}"),
                "rule_id": edit["rule_id"],
                "reference": edit["reference"],
                "data": {"column1": c1, "column2": c2, "denied_code": c2,
                         "modifiers_applied": sorted(applied), "bypass_allowed": False},
            }

    return {
        "tool": "check_unbundling", "finding": "clean",
        "detail": "No NCCI procedure-to-procedure edits found among the billed codes.",
        "rule_id": None, "reference": None, "data": {"codes_checked": sorted(codes)},
    }


# --------------------------------------------------------------------------- #
# Tool 2: Medical necessity (procedure <-> diagnosis linkage)
# --------------------------------------------------------------------------- #
def check_medical_necessity(cpt, icd10_list):
    kb = load_kb()
    for rule in kb["medical_necessity"]:
        if rule["cpt"] == cpt:
            supported = any(_icd_matches(dx, rule["supported_icd10_prefixes"]) for dx in icd10_list)
            if supported:
                return {
                    "tool": "check_medical_necessity", "finding": "clean",
                    "detail": f"{cpt} is supported by at least one billed diagnosis ({', '.join(icd10_list)}).",
                    "rule_id": rule["rule_id"], "reference": rule["reference"],
                    "data": {"cpt": cpt, "diagnoses": icd10_list, "supported": True},
                }
            return {
                "tool": "check_medical_necessity", "finding": "violation",
                "detail": (f"{cpt} is not supported by any billed diagnosis ({', '.join(icd10_list)}). "
                           f"{rule['rationale']}"),
                "rule_id": rule["rule_id"], "reference": rule["reference"],
                "data": {"cpt": cpt, "diagnoses": icd10_list, "supported": False,
                         "expected_prefixes": rule["supported_icd10_prefixes"]},
            }
    return {
        "tool": "check_medical_necessity", "finding": "info",
        "detail": f"No restrictive medical-necessity policy on file for {cpt}; not flagged.",
        "rule_id": None, "reference": None, "data": {"cpt": cpt, "diagnoses": icd10_list},
    }


# --------------------------------------------------------------------------- #
# Tool 3: Billed-amount anomaly detection (z-score vs. historical distribution)
# --------------------------------------------------------------------------- #
def detect_amount_anomaly(cpt, billed_amount, z_threshold=3.0):
    from .real_data import get_amount_baselines
    stats = get_amount_baselines().get(cpt)
    if not stats:
        return {
            "tool": "detect_amount_anomaly", "finding": "info",
            "detail": f"No billed-amount baseline for {cpt}; anomaly check skipped.",
            "rule_id": None, "reference": None, "data": {"cpt": cpt, "billed_amount": billed_amount},
        }
    z = (billed_amount - stats["mean"]) / stats["std"] if stats["std"] else 0.0
    is_anom = abs(z) >= z_threshold
    src = stats["source"]
    src_txt = (f"real CMS Medicare data, n={stats['n']}" if src.startswith("CMS")
               else "synthetic baseline")
    return {
        "tool": "detect_amount_anomaly",
        "finding": "anomaly" if is_anom else "clean",
        "detail": (f"{cpt} billed at ${billed_amount:.2f} vs expected ${stats['mean']:.2f} "
                   f"(sd ${stats['std']:.2f}, {src_txt}); z-score {z:+.1f}. "
                   + ("Statistical outlier - exceeds the anomaly threshold."
                      if is_anom else "Within normal range.")),
        "rule_id": "STAT-ANOM-Z",
        "reference": ("CMS Medicare Physician & Other Practitioners charges (real)"
                      if src.startswith("CMS") else "Statistical outlier detection (z-score)"),
        "data": {"cpt": cpt, "billed_amount": billed_amount, "mean": stats["mean"],
                 "std": stats["std"], "z_score": round(z, 2), "threshold": z_threshold,
                 "baseline_source": src, "baseline_n": stats["n"]},
    }


# --------------------------------------------------------------------------- #
# Tool 4: Duplicate claim detection
# --------------------------------------------------------------------------- #
def check_duplicate(claim, history):
    """A duplicate = same member + provider + date_of_service + CPT set as a prior paid claim."""
    this_codes = frozenset(l["cpt"] for l in claim["lines"])
    for prior in history:
        same_key = (prior["member_id"] == claim["member_id"]
                    and prior["provider_id"] == claim["provider_id"]
                    and prior["date_of_service"] == claim["date_of_service"])
        prior_codes = frozenset(l["cpt"] for l in prior["lines"])
        if same_key and prior_codes == this_codes:
            return {
                "tool": "check_duplicate", "finding": "violation",
                "detail": (f"Matches previously adjudicated claim {prior['claim_id']} "
                           f"(same member, provider, date of service, and procedure codes). "
                           f"Duplicate services are non-payable."),
                "rule_id": "DUP-001", "reference": "Plan payment policy (duplicate)",
                "data": {"matched_claim": prior["claim_id"], "codes": sorted(this_codes)},
            }
    return {
        "tool": "check_duplicate", "finding": "clean",
        "detail": "No matching prior claim found in the recent claim history.",
        "rule_id": None, "reference": None, "data": {"codes": sorted(this_codes)},
    }


# --------------------------------------------------------------------------- #
# Tool 5: Possible-upcoding heuristic (signal only, never auto-denies)
# --------------------------------------------------------------------------- #
def check_upcoding(cpt, icd10_list):
    kb = load_kb()
    watch = kb["upcoding_watch"]
    if cpt in watch["high_level_em_codes"]:
        all_low = all(_icd_matches(dx, watch["low_acuity_icd10_prefixes"]) for dx in icd10_list)
        if all_low:
            return {
                "tool": "check_upcoding", "finding": "anomaly",
                "detail": (f"High-complexity E/M {cpt} billed with only routine/low-acuity "
                           f"diagnoses ({', '.join(icd10_list)}). Possible up-coding signal that "
                           f"cannot be confirmed from claim data; medical-record review advised."),
                "rule_id": "UPCODE-EM", "reference": "Coding-intensity policy (illustrative)",
                "data": {"cpt": cpt, "diagnoses": icd10_list},
            }
    return {
        "tool": "check_upcoding", "finding": "clean",
        "detail": f"No up-coding signal for {cpt} given the billed diagnoses.",
        "rule_id": None, "reference": None, "data": {"cpt": cpt, "diagnoses": icd10_list},
    }


# --------------------------------------------------------------------------- #
# Tool 6: Policy lookup (lightweight retrieval for citations / explanation)
# --------------------------------------------------------------------------- #
def lookup_policy(topic):
    """Retrieve the most relevant policy passage for a topic/question (RAG)."""
    from .rag import retrieve
    res = retrieve(topic, k=1)
    if res:
        top = res[0]
        return {"tool": "lookup_policy", "finding": "info", "detail": top["text"],
                "rule_id": top["id"], "reference": top["source"],
                "data": {"query": topic, "score": top["score"], "retrieval": "RAG"}}
    return {"tool": "lookup_policy", "finding": "info",
            "detail": f"No policy passage retrieved for '{topic}'.",
            "rule_id": None, "reference": None, "data": {"query": topic}}


# Registry used by the rule-based agent and to build LLM tool schemas.
TOOL_REGISTRY = {
    "check_unbundling": check_unbundling,
    "check_medical_necessity": check_medical_necessity,
    "detect_amount_anomaly": detect_amount_anomaly,
    "check_duplicate": check_duplicate,
    "check_upcoding": check_upcoding,
    "lookup_policy": lookup_policy,
}
