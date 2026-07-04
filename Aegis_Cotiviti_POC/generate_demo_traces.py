"""Run every synthetic claim through the deterministic pipeline, print a summary
table (to verify behaviour), and cache the full results to demo_traces.json so
the Streamlit app's Demo Mode is instant and offline."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from aegis import load_claims, review_claim

claims, history = load_claims()
results = {}

print(f"{'CLAIM':<10} {'PROPOSED':<15} {'FINAL':<15} {'$IMPACT':>9}  {'CONF':>5}  GUARDRAILS")
print("-" * 90)
for c in claims:
    r = review_claim(c, history, mode="demo")
    results[c["claim_id"]] = r
    p = r["proposed"]
    g = "; ".join(gr.split(":")[0] for gr in r["guardrails_triggered"]) or "-"
    print(f"{c['claim_id']:<10} {p['verdict']:<15} {r['final_verdict']:<15} "
          f"${p['dollars_impact']:>8.2f}  {p['confidence']:>5.2f}  {g}")

with open(os.path.join(os.path.dirname(__file__), "demo_traces.json"), "w") as f:
    json.dump(results, f, indent=2)

print("-" * 90)
totals = {}
for r in results.values():
    totals[r["final_verdict"]] = totals.get(r["final_verdict"], 0) + 1
print("Final dispositions:", totals)
flagged = sum(r["proposed"]["dollars_impact"] for r in results.values()
              if r["final_verdict"] in ("DENY", "ROUTE_TO_HUMAN"))
print(f"Total dollars flagged/denied: ${flagged:.2f}")
print("Wrote demo_traces.json")
