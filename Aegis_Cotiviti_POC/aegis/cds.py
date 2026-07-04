"""
cds.py - Clinical Decision Support (the Treatment axis of TPO)
-------------------------------------------------------------
Same agentic pattern as the payment side (plan -> tools -> reason -> recommend),
but it answers a clinical question instead of a payment one: given a patient's
conditions and what's being ordered, is the care consistent with guidelines?

Tools (identical result-dict shape as the payment tools, so they render the same):
  check_care_gaps          - guideline-recommended actions not yet done
  check_contraindication   - an ordered drug/procedure unsafe for the conditions
  check_guideline_alignment- is the ordered procedure appropriate for the diagnosis

The recommendation NEVER changes the payment verdict - clinical quality and
payment correctness are separate axes. High-severity findings escalate to a
clinician (the same human-in-the-loop governance philosophy).
"""
import json
import os
from functools import lru_cache

_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@lru_cache(maxsize=1)
def load_clinical_kb():
    return json.load(open(os.path.join(_DATA, "clinical_kb.json")))


def load_patient_cases():
    return load_clinical_kb()["patient_cases"]


def _match(dx, prefixes):
    d = dx.replace(".", "").upper()
    return any(d.startswith(p.replace(".", "").upper()) for p in prefixes)


def _any_condition(conditions, prefixes):
    return any(_match(dx, prefixes) for dx in conditions)


# --------------------------------------------------------------- care gaps ---
def check_care_gaps(conditions, history, current_meds):
    kb = load_clinical_kb()
    hist_cpts = {(h["cpt"], h.get("days_ago", 9999)) for h in history}
    meds = {m.lower() for m in current_meds}
    gaps, reminders = [], []
    for g in kb["care_guidelines"]:
        if not _any_condition(conditions, g["condition_prefixes"]):
            continue
        rec = g["recommended"]
        if g.get("type") == "reminder":
            reminders.append(rec["label"]); continue
        satisfied = False
        if "cpt" in rec:
            satisfied = any(c == rec["cpt"] and d <= rec.get("lookback_days", 9999)
                            for c, d in hist_cpts)
        elif "med" in rec:
            satisfied = rec["med"].lower() in meds
        if not satisfied:
            gaps.append({"label": rec["label"], "rule_id": g["rule_id"], "reference": g["reference"]})
    if gaps:
        return {"tool": "check_care_gaps", "finding": "anomaly",
                "detail": "Open care gap(s): " + "; ".join(x["label"] for x in gaps) + ".",
                "rule_id": gaps[0]["rule_id"], "reference": gaps[0]["reference"],
                "data": {"gaps": gaps, "reminders": reminders}}
    return {"tool": "check_care_gaps", "finding": "clean",
            "detail": "No open guideline care gaps for the patient's conditions."
                      + (f" Monitoring reminders: {'; '.join(reminders)}." if reminders else ""),
            "rule_id": None, "reference": None, "data": {"gaps": [], "reminders": reminders}}


# --------------------------------------------------------- contraindication --
def check_contraindication(ordered, conditions):
    kb = load_clinical_kb()
    med_map = kb["med_intervention_map"]
    interventions = []
    for o in ordered:
        if o.get("type") == "med":
            interventions.append(med_map.get(o["name"].lower(), o["name"].lower()))
    for ci in kb["contraindications"]:
        if ci["intervention"] in interventions and _any_condition(conditions, ci["condition_prefixes"]):
            return {"tool": "check_contraindication",
                    "finding": "violation" if ci["severity"] == "high" else "anomaly",
                    "detail": ci["detail"], "rule_id": ci["rule_id"], "reference": ci["reference"],
                    "data": {"intervention": ci["intervention"], "severity": ci["severity"]}}
    return {"tool": "check_contraindication", "finding": "clean",
            "detail": "No drug-condition contraindications detected among the ordered items.",
            "rule_id": None, "reference": None, "data": {"severity": "none"}}


# ------------------------------------------------------ guideline alignment --
def check_guideline_alignment(ordered, conditions):
    kb = load_clinical_kb()
    proc_cpts = {o["cpt"] for o in ordered if o.get("type") == "procedure" and o.get("cpt")}
    for a in kb["alignment"]:
        if a["procedure_cpt"] in proc_cpts and _any_condition(conditions, a["misaligned_condition_prefixes"]):
            return {"tool": "check_guideline_alignment", "finding": "anomaly",
                    "detail": a["detail"], "rule_id": a["rule_id"], "reference": a["reference"],
                    "data": {"procedure": a["procedure_cpt"], "severity": a["severity"]}}
    return {"tool": "check_guideline_alignment", "finding": "clean",
            "detail": "Ordered procedure(s) are consistent with guidelines for the diagnosis.",
            "rule_id": None, "reference": None, "data": {"severity": "none"}}


# ------------------------------------------------- deterministic CDS reasoner -
class CDSAgent:
    """Runs the CDS tools and produces a clinical recommendation + trace."""
    name = "cds_rule_based"

    def review(self, case):
        conditions = case["conditions"]
        ordered = case["ordered"]
        trace = [{"type": "plan",
                  "content": (f"Patient {case['case_id']} with conditions {', '.join(conditions)}. "
                              f"Plan: (1) care gaps, (2) contraindications, (3) guideline alignment "
                              f"of ordered items. Then form a clinical recommendation.")}]
        findings = []

        def run(res):
            trace.append({"type": "observation", "tool": res["tool"], "finding": res["finding"],
                          "content": res["detail"], "reference": res.get("reference")})
            findings.append(res)
            return res

        gaps = run(check_care_gaps(conditions, case.get("history", []), case.get("current_meds", [])))
        ci = run(check_contraindication(ordered, conditions))
        align = run(check_guideline_alignment(ordered, conditions))

        cites = [{"rule_id": f["rule_id"], "description": f["detail"], "reference": f["reference"]}
                 for f in findings if f["finding"] in ("violation", "anomaly") and f["rule_id"]]

        # decide + light clinical governance (high severity -> clinician sign-off)
        if ci["finding"] == "violation":
            rec, sev, conf = "ROUTE_TO_CLINICIAN", "high", 0.92
            rationale = ci["detail"] + " High-severity safety issue requires clinician review."
        elif ci["finding"] == "anomaly" or align["finding"] == "anomaly":
            rec, sev, conf = "CLINICAL_ALERT", "moderate", 0.8
            rationale = " ".join(f["detail"] for f in (ci, align) if f["finding"] == "anomaly")
        elif gaps["finding"] == "anomaly":
            rec, sev, conf = "ADDRESS_CARE_GAP", "moderate", 0.85
            rationale = gaps["detail"]
        else:
            rec, sev, conf = "ALIGNED", "none", 0.9
            rationale = "Care is consistent with guidelines; no gaps or safety issues detected."

        trace.append({"type": "reasoning",
                      "content": f"Synthesizing three checks -> severity '{sev}'."})
        trace.append({"type": "decision", "content": f"Clinical recommendation: {rec}."})
        recommendation = {"recommendation": rec, "severity": sev, "confidence": conf,
                          "rationale": rationale, "guidelines_cited": cites}
        return recommendation, trace


def review_patient(case):
    return CDSAgent().review(case)


# ------------------------------------- claim-level clinical advisory (panel) --
def advisory_for_claim(claim):
    """Lightweight clinical read of a claim (advisory only; does not affect payment).

    Uses only what a claim carries: diagnoses + ordered procedures. It does not
    attempt full care-gap detection (no clinical history on a claim), so it
    surfaces guideline alignment + chronic-condition reminders instead.
    """
    kb = load_clinical_kb()
    conditions = sorted({dx for l in claim["lines"] for dx in l["icd10"]})
    ordered = [{"type": "procedure", "cpt": l["cpt"]} for l in claim["lines"]]

    align = check_guideline_alignment(ordered, conditions)
    findings = [{"tool": align["tool"], "finding": align["finding"],
                 "content": align["detail"], "reference": align.get("reference")}]
    reminders = [txt for pref, txt in kb["chronic_reminders"].items()
                 if _any_condition(conditions, [pref])]
    for r in reminders:
        findings.append({"tool": "chronic_reminder", "finding": "info", "content": r,
                         "reference": "Guideline monitoring reminder (illustrative)"})

    if align["finding"] == "anomaly":
        note = "Clinical flag: ordered service may not align with the diagnosis (advisory)."
    elif reminders:
        note = "No alignment issue; chronic-condition monitoring reminders apply (advisory)."
    else:
        note = "No clinical alignment issues detected for this claim (advisory)."
    return {"note": note, "findings": findings}
