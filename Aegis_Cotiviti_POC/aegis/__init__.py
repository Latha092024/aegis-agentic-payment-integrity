"""Aegis - Agentic Payment-Integrity Analyst (POC)."""
import json
import os

from .agent import get_agent, RuleBasedAgent, LLMAgent
from .governance import apply_guardrails
from .cds import review_patient, load_patient_cases, advisory_for_claim

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_claims():
    with open(os.path.join(_DATA_DIR, "claims.json")) as f:
        data = json.load(f)
    return data["claims"], data["history"]


def review_claim(claim, history, mode="demo", model="claude-sonnet-4-6"):
    """Run the full pipeline: agent proposes -> governance enforces.

    Returns a single result dict combining the agent verdict, the governance
    outcome, and the reasoning trace.
    """
    agent = get_agent(mode, model=model)
    proposed, trace = agent.review(claim, history)
    gov = apply_guardrails(proposed, claim)
    return {
        "claim_id": claim["claim_id"],
        "agent_mode": agent.name,
        "proposed": proposed,
        "final_verdict": gov["final_verdict"],
        "overridden": gov["overridden"],
        "guardrails_triggered": gov["guardrails_triggered"],
        "audit_record": gov["audit_record"],
        "trace": trace,
        "clinical": advisory_for_claim(claim),   # advisory only - does not affect payment
    }


__all__ = ["load_claims", "review_claim", "apply_guardrails",
           "get_agent", "RuleBasedAgent", "LLMAgent",
           "review_patient", "load_patient_cases", "advisory_for_claim"]
