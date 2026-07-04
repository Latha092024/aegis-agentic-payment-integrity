"""
governance.py
-------------
The AI-safety wrapper around the agent. An autonomous agent may *recommend* a
disposition, but this layer enforces organizational guardrails before anything
is treated as final. It implements three things a real "AI Governance Program"
cares about:

  1. Human-in-the-loop guardrails
        - low-confidence decisions are escalated to a human
        - high-dollar automated denials require human sign-off
  2. Privacy
        - member identifiers are redacted from the audit log
  3. Accountability / explainability
        - a complete, timestamped audit record of every step and every guardrail
          that fired, with the rules the decision relied on
"""

import datetime
import hashlib

CONFIDENCE_THRESHOLD = 0.70      # below this -> human review
HIGH_VALUE_DENIAL = 250.0        # automated denials above this $ -> human sign-off

VALID_VERDICTS = {"APPROVE", "DENY", "ROUTE_TO_HUMAN"}


def redact_member(member_id: str) -> str:
    """Irreversibly mask a member identifier for logs (keep it joinable via a hash)."""
    if not member_id:
        return "REDACTED"
    h = hashlib.sha256(member_id.encode()).hexdigest()[:8]
    return f"MEMBER::{h}"


def apply_guardrails(agent_verdict: dict, claim: dict) -> dict:
    """Take the agent's proposed verdict and enforce governance policy.

    Returns a dict with the (possibly overridden) final verdict, the list of
    guardrails that fired, and a redacted audit record.
    """
    proposed = agent_verdict.get("verdict", "ROUTE_TO_HUMAN")
    if proposed not in VALID_VERDICTS:
        proposed = "ROUTE_TO_HUMAN"

    confidence = float(agent_verdict.get("confidence", 0.0))
    dollars = float(agent_verdict.get("dollars_impact", 0.0))

    final = proposed
    guardrails = []

    # Guardrail 1: confidence floor
    if confidence < CONFIDENCE_THRESHOLD and proposed != "ROUTE_TO_HUMAN":
        final = "ROUTE_TO_HUMAN"
        guardrails.append(
            f"CONFIDENCE_FLOOR: model confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD:.2f}; "
            f"'{proposed}' downgraded to human review."
        )

    # Guardrail 2: high-dollar automated denial requires human sign-off
    if proposed == "DENY" and dollars > HIGH_VALUE_DENIAL:
        final = "ROUTE_TO_HUMAN"
        guardrails.append(
            f"HIGH_VALUE_DENIAL: proposed denial of ${dollars:.2f} exceeds "
            f"${HIGH_VALUE_DENIAL:.2f}; automated denial requires human sign-off."
        )

    audit = {
        "audit_id": hashlib.sha256(
            f"{claim['claim_id']}-{datetime.datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12],
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "claim_id": claim["claim_id"],
        "member_ref": redact_member(claim.get("member_id", "")),
        "provider_id": claim.get("provider_id"),
        "proposed_verdict": proposed,
        "final_verdict": final,
        "confidence": round(confidence, 2),
        "dollars_impact": round(dollars, 2),
        "guardrails_triggered": guardrails,
        "rules_applied": agent_verdict.get("rules_applied", []),
        "governance_policy": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "high_value_denial_threshold": HIGH_VALUE_DENIAL,
        },
    }

    return {
        "final_verdict": final,
        "overridden": final != proposed,
        "guardrails_triggered": guardrails,
        "audit_record": audit,
    }
