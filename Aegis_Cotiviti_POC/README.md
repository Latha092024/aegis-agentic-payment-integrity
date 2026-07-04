# 🛡️ Aegis — Agentic Payment-Integrity Analyst

A hackathon proof-of-concept for **Cotiviti (Topic 2: Agentic AI for Treatment, Payment & Operations)**.

## The problem
Healthcare insurers process **millions of claims a year**. Manual review is slow,
expensive, and inconsistent. Aegis demonstrates how an **Agentic AI** system can
autonomously review healthcare claims using payment policies, machine learning,
retrieval-augmented generation (RAG), and AI governance to support **Treatment,
Payment, and Operations (TPO)**.

## Why each component exists (business purpose)
| Component | Business problem it solves |
|---|---|
| Rule engine | Validates payment policies (NCCI unbundling, medical necessity, duplicates) |
| RAG | Retrieves supporting CMS/policy text and grounds cited answers |
| ML risk model | Predicts the probability that a claim is problematic |
| Clustering | Compares each provider against its billing peers |
| Time-series | Detects abnormal shifts in provider behavior over time |
| Clinical decision support | Checks whether the care itself follows guidelines |
| Governance | Keeps a human in the loop and produces an audit trail |

**TPO split:** Treatment → Clinical Decision Support · Payment → claim review + coding/policy
validation · Operations → provider clustering + anomaly detection.

## How it works
Aegis is an **autonomous agent** that reviews a medical claim, **plans** which
payment-integrity checks to run, **calls tools** to gather evidence, **reasons**
over the results, and reaches a **goal-directed verdict** — `APPROVE`, `DENY`,
or `ROUTE_TO_HUMAN` — with a confidence score, dollar impact, and a citation to
the exact rule it applied. A **governance layer** enforces human-in-the-loop
guardrails and writes an immutable audit trail.

## Why this maps to the role
| Topic 2 technique | Where it shows up in Aegis |
|---|---|
| Chain reasoning | Step-by-step reasoning trace: plan → tool → observe → reason → decide |
| Agentic generative AI | `LLMAgent` — a real LLM autonomously chooses and calls tools in a loop |
| Classification | 3-way verdict (approve / deny / route); each rule tool is a binary classifier |
| Prediction | Trained GradientBoosting model → problematic-claim probability (`predict_claim_risk`) |
| Inference | Logical inference (disposition from rules + evidence) + LLM model inference |
| Clustering | KMeans provider peer groups + distance-to-centroid outlier (`provider_peer_outlier`) |
| Time-series anomaly detection | Rolling z-score on per-provider daily claim volume (`detect_temporal_anomaly`) |
| Retrieval-Augmented Generation (RAG) | `rag.py`: retrieval over policy/guideline text → grounded, cited answers (Policy Q&A tab); powers `lookup_policy` |
| Treatment / Payment / Operations (TPO) | CDS clinical review (Treatment) · claim adjudication (Payment) · population analytics (Operations) |
| Clinical decision support | `cds.py`: care gaps, contraindications, guideline alignment — advisory, escalates to a clinician |
| AI governance (ethics & safety) | `governance.py`: confidence floor, high-dollar sign-off, PII redaction, audit log |

## Architecture
```
Claim ─► Agent (plans + calls tools) ─► proposes verdict ─► Governance (guardrails) ─► Final verdict + Audit
                     │
        ┌────────────┴───────────────────────────────────┐
        ▼                                                 ▼
  LLMAgent (live, real tool-calling)         RuleBasedAgent (offline, deterministic)
        └──────────────── same tools ────────────────────┘
   Rule tools:  check_unbundling · check_medical_necessity · detect_amount_anomaly
                check_duplicate · check_upcoding · lookup_policy
   ML tools:    predict_claim_risk (prediction) · provider_peer_outlier (clustering)
                detect_temporal_anomaly (time-series)
   CDS tools:   check_care_gaps · check_contraindication · check_guideline_alignment
                (Treatment axis — advisory, escalates high-severity findings to a clinician)
```
- **Two interchangeable brains, one verdict schema.** `LLMAgent` is the real agentic
  engineering (the LLM autonomously chooses tools). `RuleBasedAgent` runs the same
  tools deterministically so **Demo Mode is 100% offline and reliable** for the video —
  and doubles as a transparent baseline to compare the LLM against.

## Run it
```bash
pip install -r requirements.txt
python build_models.py      # one-time: trains the risk model, fits clustering, builds time series
streamlit run app.py
```
- **Demo Mode** (default): no API key, instant, deterministic. Use this for the video.
- **Live Mode**: `export ANTHROPIC_API_KEY=sk-...` then pick "Live (LLM)" in the sidebar.
- **Population Analytics tab**: the risk model card (AUC + feature importances), the
  provider peer-group clusters, and the per-provider time-series anomaly view.
- **Clinical Decision Support tab**: pick a patient case → the CDS agent checks care gaps,
  contraindications, and guideline alignment (the Treatment axis of TPO). The claim
  review also shows a clinical advisory panel alongside the payment verdict.
- **Policy Q&A (RAG) tab**: ask a policy/guideline question in plain English → retrieves the
  most relevant passages and grounds a cited answer in them. The corpus includes public
  CMS/NCCI/Medicare policy summaries (`data/policy_corpus.json`). Default retriever is TF-IDF
  (offline); set `AEGIS_RAG_BACKEND=dense` (after `pip install sentence-transformers`) for
  semantic embeddings. In Live mode the answer is LLM-generated from the retrieved passages.

Optional — regenerate the cached demo results and see the summary table:
```bash
python generate_demo_traces.py
```

## The 9 demo claims
Cover clean approvals, hard denials (unbundling, medical necessity, duplicate), and
human-escalation cases (billed-amount anomaly, possible up-coding, and a **high-dollar
denial that the governance layer overrides to human review even though the agent was
confident** — the AI-safety highlight).

## Real data
The billed-amount anomaly baselines use **real, public-domain CMS data** — the
*Medicare Physician & Other Practitioners by Provider and Service* dataset
(submitted charges per HCPCS code). A small real sample ships in
`data/real/cms_charges_sample.csv` so real baselines work out of the box; run
`python fetch_real_data.py` to pull the full real dataset (all demo codes) via the
CMS public API. Any code not covered by real data falls back to a synthetic baseline.

- Source: https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service (public domain, US government works)
- The sidebar shows live real-vs-synthetic coverage.

Architecture note (what's real vs synthetic): **real reference data** (CMS charges;
extendable to real NCCI edits and provider profiles) drives the checks, while the
**claims** flowing through are synthetic — for patient privacy and so each demo
scenario (clean approve, unbundling, governance override, contraindication) is
reproducible. This "real rules, controlled claims" split mirrors how a production
payment-integrity engine is actually shaped.

## Evaluation
`python evaluate.py` runs the **full pipeline** (rules + ML + governance) over a
larger labelled synthetic test set and reports system-level metrics (shown in the
Batch Dashboard tab). A representative run of 300 claims:

| Metric | Value |
|---|---|
| Accuracy | ~0.88 |
| Precision | ~0.83 |
| Recall | ~0.93 |
| False-positive rate | ~0.17 |
| Avg processing time | ~27 ms / claim |

Metrics are intentionally imperfect (the labelled data contains noise), which is
realistic for a claims-risk system and shows this is an engineering evaluation, not a
cherry-picked demo. "Flagged" = any verdict other than approve.

## Future work
- **FHIR compatibility** — ingest standard FHIR `Claim` resources instead of custom JSON.
- **Full CMS/NCCI ingestion** — load the complete real NCCI PTP, MUE, and LCD/NCD tables.
- **Versioned policy management** — track policy changes so past decisions stay reproducible.
- **Multi-agent design** — split into specialized planning, payment-integrity, clinical, and
  governance agents.
- **Continuous / federated learning + human-feedback loops** — improve the risk model over time
  from reviewer decisions without moving PHI.

## ⚠️ Disclaimer
Claims are synthetic (no PHI). The payment rules (NCCI/necessity) and clinical
guidelines are **illustrative stand-ins** and must not be used for real
adjudication or medical decisions. The billed-amount baselines are real CMS data
used for demonstration only.
