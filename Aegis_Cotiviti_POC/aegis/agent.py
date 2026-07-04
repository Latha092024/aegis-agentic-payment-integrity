"""
agent.py
--------
Two interchangeable "brains" that both produce the SAME verdict schema:

  * LLMAgent       - a genuine agentic loop. The LLM is given the tools and
                     autonomously decides which to call, reasons over the
                     results, and returns a structured verdict. Requires an
                     ANTHROPIC_API_KEY. This is the "real" agentic engineering.

  * RuleBasedAgent - a deterministic orchestrator that runs the SAME tools in a
                     sensible order and composes a verdict + reasoning trace.
                     No network needed. Used for the offline "Demo Mode" so the
                     live presentation is 100% reliable, and as a transparent
                     baseline to compare the LLM against.

Verdict schema (both agents):
    {
      "verdict": "APPROVE" | "DENY" | "ROUTE_TO_HUMAN",
      "confidence": float,               # 0..1
      "rationale": str,
      "rules_applied": [ {rule_id, description, reference} ],
      "dollars_impact": float,           # $ that would be denied / recovered
    }

Trace: list of step dicts rendered live in the UI:
    {"type": "plan"|"tool_call"|"observation"|"reasoning"|"decision", "content": str, ...}
"""

import json
import os

from .tools import TOOL_REGISTRY as _RULE_TOOLS, load_kb
from .ml import predict_claim_risk, provider_peer_outlier, detect_temporal_anomaly

# Full tool set the agents can call: deterministic rule tools + ML-backed tools.
TOOLS = {**_RULE_TOOLS,
         "predict_claim_risk": predict_claim_risk,
         "provider_peer_outlier": provider_peer_outlier,
         "detect_temporal_anomaly": detect_temporal_anomaly}

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _claim_summary(claim):
    lines = "; ".join(
        f"{l['cpt']}"
        + (f"-{'/'.join(l['modifiers'])}" if l["modifiers"] else "")
        + f" dx[{','.join(l['icd10'])}] ${l['billed_amount']:.2f}"
        for l in claim["lines"]
    )
    return (f"Claim {claim['claim_id']} | member {claim['member_id']} | provider "
            f"{claim['provider_id']} | DOS {claim['date_of_service']} | lines: {lines}")


# =========================================================================== #
#  Deterministic rule-based agent (offline demo)
# =========================================================================== #
class RuleBasedAgent:
    name = "rule_based"

    def review(self, claim, history):
        trace = []
        cpts = [l["cpt"] for l in claim["lines"]]
        mods = {l["cpt"]: l["modifiers"] for l in claim["lines"]}

        trace.append({
            "type": "plan",
            "content": (f"Received {claim['claim_id']} with {len(claim['lines'])} line(s). "
                        f"Plan: (1) NCCI/unbundling, (2) medical necessity, (3) billed-amount anomaly, "
                        f"(4) duplicate history, (5) up-coding signal, (6) ML risk score, "
                        f"(7) provider peer-group outlier, (8) provider temporal anomaly. "
                        f"Then synthesize a disposition."),
        })

        findings = []

        def run(tool_name, *args, note=None, **kwargs):
            res = TOOLS[tool_name](*args, **kwargs)
            trace.append({"type": "tool_call", "tool": tool_name,
                          "content": note or f"Calling {tool_name}", "input": _describe_args(args, kwargs)})
            trace.append({"type": "observation", "tool": tool_name,
                          "finding": res["finding"], "content": res["detail"],
                          "reference": res.get("reference")})
            findings.append(res)
            return res

        # 1. Unbundling (claim-level)
        run("check_unbundling", cpts, mods, note="Checking NCCI procedure-to-procedure edits")
        # 2. Medical necessity (per line)
        for l in claim["lines"]:
            run("check_medical_necessity", l["cpt"], l["icd10"],
                note=f"Medical-necessity check for {l['cpt']}")
        # 3. Amount anomaly (per line)
        for l in claim["lines"]:
            run("detect_amount_anomaly", l["cpt"], l["billed_amount"],
                note=f"Billed-amount anomaly check for {l['cpt']}")
        # 4. Duplicate (claim-level)
        run("check_duplicate", claim, history, note="Duplicate-history check")
        # 5. Upcoding (per line)
        for l in claim["lines"]:
            run("check_upcoding", l["cpt"], l["icd10"], note=f"Up-coding signal check for {l['cpt']}")
        # 6. ML risk prediction (claim-level)
        run("predict_claim_risk", claim, note="Scoring claim with the trained risk model")
        # 7. Provider peer-group clustering
        run("provider_peer_outlier", claim["provider_id"], note="Provider peer-group (clustering) check")
        # 8. Provider temporal (time-series) anomaly
        run("detect_temporal_anomaly", claim["provider_id"], note="Provider temporal-anomaly (time-series) check")

        verdict = self._decide(claim, findings, trace)
        return verdict, trace

    def _decide(self, claim, findings, trace):
        # Map a denied CPT back to its billed amount.
        amt = {l["cpt"]: l["billed_amount"] for l in claim["lines"]}
        rules_applied, hard_denials, anomalies = [], [], []
        risk = None  # ML risk score is advisory corroboration, not a routing trigger on its own

        for f in findings:
            if f["tool"] == "predict_claim_risk":
                risk = f
                if f["finding"] == "anomaly":
                    rules_applied.append({"rule_id": f["rule_id"], "description": f["detail"],
                                          "reference": f["reference"]})
                continue
            if f["finding"] == "violation":
                dollars = 0.0
                if f["tool"] == "check_unbundling":
                    dollars = amt.get(f["data"].get("denied_code"), 0.0)
                elif f["tool"] == "check_medical_necessity":
                    dollars = amt.get(f["data"].get("cpt"), 0.0)
                elif f["tool"] == "check_duplicate":
                    dollars = claim["total_billed"]
                hard_denials.append((f, dollars))
                rules_applied.append({"rule_id": f["rule_id"], "description": f["detail"],
                                      "reference": f["reference"]})
            elif f["finding"] == "anomaly":
                anomalies.append(f)
                rules_applied.append({"rule_id": f["rule_id"], "description": f["detail"],
                                      "reference": f["reference"]})

        risk_note = (f" ML risk model corroborates ({risk['data']['probability']:.0%} probability)."
                     if risk and risk["finding"] == "anomaly" else "")

        if hard_denials:
            total = round(sum(d for _, d in hard_denials), 2)
            reasons = " ".join(f["detail"] for f, _ in hard_denials) + risk_note
            trace.append({"type": "reasoning",
                          "content": (f"Found {len(hard_denials)} hard rule violation(s) totaling "
                                      f"${total:.2f} in non-payable charges. This warrants a denial "
                                      f"recommendation.")})
            trace.append({"type": "decision", "content": f"Proposed verdict: DENY (${total:.2f})."})
            return {"verdict": "DENY", "confidence": 0.93, "rationale": reasons,
                    "rules_applied": rules_applied, "dollars_impact": total}

        if anomalies:
            flagged = round(sum(claim["lines"][i]["billed_amount"]
                                for i, l in enumerate(claim["lines"])), 2)
            reasons = " ".join(a["detail"] for a in anomalies) + risk_note
            trace.append({"type": "reasoning",
                          "content": ("No hard rule violation, but one or more statistical/coding "
                                      "signals require judgment that claim data alone cannot resolve. "
                                      "Escalating rather than auto-denying is the safe action.")})
            trace.append({"type": "decision", "content": "Proposed verdict: ROUTE_TO_HUMAN."})
            return {"verdict": "ROUTE_TO_HUMAN", "confidence": 0.66, "rationale": reasons,
                    "rules_applied": rules_applied, "dollars_impact": flagged}

        trace.append({"type": "reasoning",
                      "content": "All checks passed with no violations or anomalies."})
        trace.append({"type": "decision", "content": "Proposed verdict: APPROVE."})
        return {"verdict": "APPROVE", "confidence": 0.95,
                "rationale": "All payment-integrity checks passed; claim is payable as billed.",
                "rules_applied": [], "dollars_impact": 0.0}


def _describe_args(args, kwargs):
    parts = []
    for a in args:
        if isinstance(a, dict) and "claim_id" in a:
            parts.append(a["claim_id"])
        elif isinstance(a, list) and a and isinstance(a[0], dict):
            parts.append(f"[{len(a)} prior claims]")
        else:
            parts.append(str(a))
    parts += [f"{k}={v}" for k, v in kwargs.items()]
    return ", ".join(parts)


# =========================================================================== #
#  LLM agentic loop (live)  -- requires anthropic + ANTHROPIC_API_KEY
# =========================================================================== #
_SYSTEM_PROMPT = """You are Aegis, an autonomous payment-integrity analyst for a health plan.
Given one medical claim, decide whether it should be APPROVED, DENIED, or ROUTED_TO_HUMAN.

Work agentically:
- Reason step by step about which checks the claim needs.
- Use the provided tools to gather evidence. Do not guess at rules; call the tools.
- Beyond the rule checks you also have a trained risk model (predict_claim_risk) and
  provider-level clustering and time-series tools; use them as supporting evidence.
- Only DENY when a tool returns a concrete rule violation. Cite the rule_id and reference.
- If you see a statistical anomaly or a signal that cannot be confirmed from claim data
  alone (e.g., possible up-coding), prefer ROUTE_TO_HUMAN over DENY.
- Be explicit about the dollar impact of any denial.

When finished, call the tool `submit_verdict` exactly once with your final decision."""

# JSON-schema tool definitions for the Anthropic Messages API.
def _llm_tool_specs():
    return [
        {"name": "check_unbundling",
         "description": "Detect NCCI procedure-to-procedure (unbundling) edits among the claim's CPT codes.",
         "input_schema": {"type": "object", "properties": {
             "cpt_codes": {"type": "array", "items": {"type": "string"}},
             "modifiers_by_code": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}}},
             "required": ["cpt_codes"]}},
        {"name": "check_medical_necessity",
         "description": "Check whether a CPT code is supported by the billed ICD-10 diagnoses.",
         "input_schema": {"type": "object", "properties": {
             "cpt": {"type": "string"}, "icd10_list": {"type": "array", "items": {"type": "string"}}},
             "required": ["cpt", "icd10_list"]}},
        {"name": "detect_amount_anomaly",
         "description": "Flag a billed amount that is a statistical outlier vs. the historical distribution for that CPT.",
         "input_schema": {"type": "object", "properties": {
             "cpt": {"type": "string"}, "billed_amount": {"type": "number"}},
             "required": ["cpt", "billed_amount"]}},
        {"name": "check_duplicate",
         "description": "Detect whether this claim duplicates a previously adjudicated claim. Pass the claim object.",
         "input_schema": {"type": "object", "properties": {"use_current_claim": {"type": "boolean"}},
                          "required": ["use_current_claim"]}},
        {"name": "check_upcoding",
         "description": "Heuristic signal for possible up-coding (high-level E/M with routine diagnosis).",
         "input_schema": {"type": "object", "properties": {
             "cpt": {"type": "string"}, "icd10_list": {"type": "array", "items": {"type": "string"}}},
             "required": ["cpt", "icd10_list"]}},
        {"name": "lookup_policy",
         "description": "Retrieve a short policy note for a topic (unbundling, medical_necessity, duplicate, high_dollar_review).",
         "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}},
                          "required": ["topic"]}},
        {"name": "predict_claim_risk",
         "description": "Score the current claim with a trained ML model; returns a problematic-claim probability.",
         "input_schema": {"type": "object", "properties": {"use_current_claim": {"type": "boolean"}},
                          "required": ["use_current_claim"]}},
        {"name": "provider_peer_outlier",
         "description": "Clustering: is this provider a peer-relative billing outlier vs. similar providers?",
         "input_schema": {"type": "object", "properties": {"provider_id": {"type": "string"}},
                          "required": ["provider_id"]}},
        {"name": "detect_temporal_anomaly",
         "description": "Time-series: does this provider's daily claim volume show a recent spike/change-point?",
         "input_schema": {"type": "object", "properties": {"provider_id": {"type": "string"}},
                          "required": ["provider_id"]}},
        {"name": "submit_verdict",
         "description": "Submit the final disposition for the claim.",
         "input_schema": {"type": "object", "properties": {
             "verdict": {"type": "string", "enum": ["APPROVE", "DENY", "ROUTE_TO_HUMAN"]},
             "confidence": {"type": "number"},
             "rationale": {"type": "string"},
             "dollars_impact": {"type": "number"},
             "rules_applied": {"type": "array", "items": {"type": "object", "properties": {
                 "rule_id": {"type": "string"}, "description": {"type": "string"},
                 "reference": {"type": "string"}}}}},
             "required": ["verdict", "confidence", "rationale", "dollars_impact"]}},
    ]


class LLMAgent:
    name = "llm"

    def __init__(self, model="claude-sonnet-4-6", max_steps=12):
        from anthropic import Anthropic  # imported lazily so demo mode needs no dependency
        self.client = Anthropic()
        self.model = model
        self.max_steps = max_steps

    def _dispatch(self, tool_name, tool_input, claim, history):
        if tool_name == "check_duplicate":
            return TOOLS["check_duplicate"](claim, history)
        if tool_name == "predict_claim_risk":
            return TOOLS["predict_claim_risk"](claim)
        if tool_name in ("provider_peer_outlier", "detect_temporal_anomaly"):
            return TOOLS[tool_name](tool_input.get("provider_id", claim["provider_id"]))
        if tool_name == "check_unbundling":
            return TOOLS["check_unbundling"](
                tool_input["cpt_codes"], tool_input.get("modifiers_by_code"))
        return TOOLS[tool_name](**tool_input)

    def review(self, claim, history):
        trace = [{"type": "plan", "content": f"LLM agent engaged for {_claim_summary(claim)}"}]
        messages = [{"role": "user", "content":
                     f"Review this claim and reach a disposition.\n\n{json.dumps(claim, indent=2)}"}]
        tools = _llm_tool_specs()
        verdict = None

        for _ in range(self.max_steps):
            resp = self.client.messages.create(
                model=self.model, max_tokens=1200, system=_SYSTEM_PROMPT,
                tools=tools, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})

            tool_results = []
            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    trace.append({"type": "reasoning", "content": block.text.strip()})
                elif block.type == "tool_use":
                    if block.name == "submit_verdict":
                        verdict = {
                            "verdict": block.input["verdict"],
                            "confidence": float(block.input["confidence"]),
                            "rationale": block.input["rationale"],
                            "dollars_impact": float(block.input.get("dollars_impact", 0.0)),
                            "rules_applied": block.input.get("rules_applied", []),
                        }
                        trace.append({"type": "decision",
                                      "content": f"Proposed verdict: {verdict['verdict']} "
                                                 f"(confidence {verdict['confidence']:.2f})."})
                        break
                    result = self._dispatch(block.name, block.input, claim, history)
                    trace.append({"type": "tool_call", "tool": block.name,
                                  "content": f"Calling {block.name}", "input": json.dumps(block.input)})
                    trace.append({"type": "observation", "tool": block.name,
                                  "finding": result["finding"], "content": result["detail"],
                                  "reference": result.get("reference")})
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                         "content": json.dumps(result)})
            if verdict is not None:
                break
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        if verdict is None:  # safety net: never leave a claim undecided
            verdict = {"verdict": "ROUTE_TO_HUMAN", "confidence": 0.0,
                       "rationale": "Agent did not converge on a verdict; escalating for safety.",
                       "dollars_impact": 0.0, "rules_applied": []}
            trace.append({"type": "decision", "content": "No verdict returned; defaulting to human review."})
        return verdict, trace


def get_agent(mode, model="claude-sonnet-4-6"):
    """mode: 'demo' -> RuleBasedAgent, 'live' -> LLMAgent."""
    if mode == "live":
        return LLMAgent(model=model)
    return RuleBasedAgent()
