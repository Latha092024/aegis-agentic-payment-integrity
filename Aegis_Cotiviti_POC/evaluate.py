"""
evaluate.py - system-level evaluation of the whole Aegis pipeline.

Generates labelled synthetic claims, runs each through the FULL pipeline
(rules + ML + governance), and scores how well the system flags problematic
claims (final verdict != APPROVE) vs. ground truth.

  python evaluate.py            # runs 300 claims, saves data/eval_results.json

Metrics: accuracy, precision, recall, false-positive rate, F1, avg processing time.
"""
import json
import os
import time

from aegis import review_claim
from aegis.features import generate_eval_claims


def run_eval(n=300, mode="demo"):
    claims = generate_eval_claims(n)
    tp = tn = fp = fn = 0
    total_ms = 0.0
    for claim, label in claims:
        t = time.perf_counter()
        r = review_claim(claim, [], mode=mode)
        total_ms += (time.perf_counter() - t) * 1000
        pred = 0 if r["final_verdict"] == "APPROVE" else 1   # flagged = not auto-approved
        if pred and label:
            tp += 1
        elif not pred and not label:
            tn += 1
        elif pred and not label:
            fp += 1
        else:
            fn += 1
    N = len(claims)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": N, "accuracy": round((tp + tn) / N, 3),
        "precision": round(prec, 3), "recall": round(rec, 3),
        "fpr": round(fpr, 3), "f1": round(f1, 3),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "avg_ms": round(total_ms / N, 2), "mode": mode,
    }


if __name__ == "__main__":
    res = run_eval(300)
    out = os.path.join(os.path.dirname(__file__), "data", "eval_results.json")
    json.dump(res, open(out, "w"), indent=2)
    c = res["confusion"]
    print(f"Evaluated {res['n']} claims (mode={res['mode']})")
    print(f"  accuracy  {res['accuracy']:.3f}   precision {res['precision']:.3f}   "
          f"recall {res['recall']:.3f}")
    print(f"  FPR       {res['fpr']:.3f}   F1        {res['f1']:.3f}")
    print(f"  confusion TP={c['tp']} TN={c['tn']} FP={c['fp']} FN={c['fn']}")
    print(f"  avg processing time {res['avg_ms']:.2f} ms/claim")
    print(f"Wrote {out}")
