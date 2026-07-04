"""
app.py - Aegis Streamlit front end
Run:  streamlit run app.py
Demo Mode (default) needs no API key. Live Mode uses the real LLM agent and
requires ANTHROPIC_API_KEY in the environment.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

from aegis import load_claims, review_claim, review_patient, load_patient_cases
from aegis.ml import predict_claim_risk, provider_peer_outlier, detect_temporal_anomaly
from aegis.real_data import coverage_summary
from aegis.rag import answer as rag_answer, backend_name as rag_backend
from evaluate import run_eval

st.set_page_config(page_title="Aegis - Agentic Payment Integrity", page_icon="🛡️", layout="wide")

# ------------------------------------------------------------------ styling --
st.markdown("""
<style>
  /* All custom boxes use light backgrounds, so force dark text so they stay
     readable in BOTH light and dark Streamlit themes. */
  .verdict-card {padding: 1.05rem 1.35rem; border-radius: 12px; border-left: 8px solid;
                 margin: .6rem 0 1.1rem 0; color:#1f2937 !important;
                 box-shadow: 0 1px 4px rgba(0,0,0,.10);}
  .v-APPROVE {background:#E7F5EC; border-color:#0F9D58;}
  .v-DENY {background:#FCE8E6; border-color:#D32F2F;}
  .v-ROUTE_TO_HUMAN {background:#FEF6DF; border-color:#F5A623;}
  .verdict-title {font-size:1.35rem; font-weight:700; margin:0; color:#1f2937 !important;}
  .step {padding:.6rem .85rem; margin:.4rem 0; border-radius:9px; font-size:.92rem;
         line-height:1.5; border-left:4px solid #cfcfcf; background:#f6f7f9;
         color:#20262e !important; box-shadow:0 1px 2px rgba(0,0,0,.06);}
  .step b, .step strong {color:#20262e !important;}
  .s-plan {border-color:#5c6bc0; background:#eef0fb;}
  .s-tool_call {border-color:#8d6e63; background:#f5f4f2;}
  .s-reasoning {border-color:#7e57c2; background:#f4effb;}
  .s-decision {border-color:#37474f; background:#eceff1;}
  .s-violation {border-color:#D32F2F; background:#fdecea;}
  .s-anomaly {border-color:#E1901A; background:#fdf5e2;}
  .s-clean {border-color:#0F9D58; background:#eaf6ee;}
  .s-info {border-color:#90a4ae; background:#f2f5f7;}
  .tag {font-weight:700; text-transform:uppercase; font-size:.68rem; letter-spacing:.06em;
        color:#3a4250 !important; margin-right:.45rem;}
  .ref {color:#586173 !important; font-style:italic; font-size:.8rem; display:block; margin-top:.2rem;}
</style>
""", unsafe_allow_html=True)

VERDICT_LABEL = {"APPROVE": "✅ APPROVE — pay as billed",
                 "DENY": "⛔ DENY — non-payable charges identified",
                 "ROUTE_TO_HUMAN": "⚖️ ROUTE TO HUMAN — escalated for review"}
STEP_ICON = {"plan": "🧭 Plan", "tool_call": "🔧 Tool", "reasoning": "💭 Reasoning",
             "observation": "🔎 Observation", "decision": "⚖️ Decision"}

claims, history = load_claims()
claim_by_id = {c["claim_id"]: c for c in claims}

# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛡️ Aegis")
    st.caption("Agentic Payment-Integrity Analyst · Cotiviti POC")
    mode = st.radio("Agent mode", ["Demo (offline)", "Live (LLM)"], index=0,
                    help="Demo runs the deterministic reasoner — no API key needed. "
                         "Live uses a real LLM tool-calling agent.")
    mode_key = "live" if mode.startswith("Live") else "demo"
    model = "claude-sonnet-4-6"
    if mode_key == "live":
        model = st.text_input("Model", "claude-sonnet-4-6")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("Set ANTHROPIC_API_KEY to use Live mode.")
    anim = st.slider("Trace animation (sec/step)", 0.0, 0.6, 0.25, 0.05)
    st.divider()
    st.markdown("**Governance thresholds**")
    st.markdown("- Confidence floor: **0.70**\n- High-dollar denial: **$250**")
    st.caption("Low-confidence decisions and high-dollar automated denials are "
               "auto-escalated to a human reviewer.")
    st.divider()
    _cov = coverage_summary()
    st.markdown("**Data sources**")
    st.caption(f"Billed-amount baselines use **real CMS Medicare charges** for "
               f"{_cov['codes_real']} code(s) ({', '.join(_cov['real_codes'])}), "
               f"synthetic for the rest. Claims are synthetic (privacy + reproducible "
               f"demo scenarios). Run `fetch_real_data.py` to expand real coverage.")

st.title("Agentic Payment-Integrity Analyst")
st.markdown("An autonomous agent that **plans checks → calls tools → reasons → decides**, "
            "wrapped in a **governance layer** that keeps a human in the loop.")

with st.expander("ℹ️ The problem, the approach, and why each component exists"):
    st.markdown(
        "**The problem.** Healthcare insurers process millions of claims a year. Manual "
        "review is slow, costly, and inconsistent. Aegis shows how an **Agentic AI** system "
        "can autonomously review claims using payment policies, machine learning, retrieval-"
        "augmented generation (RAG), and AI governance — supporting **Treatment, Payment, and "
        "Operations (TPO)**.\n\n"
        "**Why each component exists (business purpose):**\n\n"
        "| Component | What it does for the business |\n"
        "|---|---|\n"
        "| Rule engine | Validates payment policies (NCCI unbundling, medical necessity, duplicates) |\n"
        "| RAG | Retrieves supporting CMS/policy text and grounds cited answers |\n"
        "| ML risk model | Predicts the probability a claim is problematic |\n"
        "| Clustering | Compares each provider against its billing peers |\n"
        "| Time-series | Detects abnormal changes in provider behavior over time |\n"
        "| Clinical decision support | Checks the care itself against guidelines |\n"
        "| Governance | Keeps a human in the loop and produces an audit trail |\n\n"
        "**Treatment / Payment / Operations:** Treatment → Clinical Decision Support tab · "
        "Payment → claim review + coding/policy validation · Operations → Population Analytics "
        "(provider clustering + anomaly detection).")

tab1, tab_cds, tab2, tab3, tab_rag, tab4 = st.tabs(
    ["🔍 Single-Claim Review", "🩺 Clinical Decision Support", "📊 Batch Dashboard",
     "🧠 Population Analytics", "🔎 Policy Q&A (RAG)", "📚 Rules & Data"])

# ============================================================ TAB 1: single ==
with tab1:
    ids = [f"{c['claim_id']} — {c['scenario_label']}" for c in claims]
    pick = st.selectbox("Select a claim", ids, index=1)
    claim = claim_by_id[pick.split(" — ")[0]]

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Claim")
        st.markdown(f"**{claim['claim_id']}** · provider `{claim['provider_id']}` · "
                    f"DOS `{claim['date_of_service']}` · POS `{claim['place_of_service']}`")
        st.dataframe(pd.DataFrame([{
            "CPT": l["cpt"], "Mods": "/".join(l["modifiers"]) or "—",
            "ICD-10": ", ".join(l["icd10"]), "Units": l["units"],
            "Billed": f"${l['billed_amount']:.2f}"} for l in claim["lines"]]),
            hide_index=True, use_container_width=True)
        st.caption(f"Total billed: ${claim['total_billed']:.2f}")
        go = st.button("▶ Run Aegis Review", type="primary", use_container_width=True)

    with right:
        st.subheader("Agent reasoning")
        trace_box = st.container()

    if go:
        if mode_key == "live" and not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("Live mode needs ANTHROPIC_API_KEY. Switch to Demo mode to run offline.")
            st.stop()
        with st.spinner("Agent working..."):
            result = review_claim(claim, history, mode=mode_key, model=model)

        # animated trace
        with trace_box:
            ph = st.empty()
            rendered = ""
            for step in result["trace"]:
                stype = step["type"]
                cls = f"s-{stype}"
                label = STEP_ICON.get(stype, stype)
                extra = ""
                if stype == "observation":
                    cls = f"s-{step.get('finding', 'info')}"
                    label = f"🔎 {step.get('finding', 'info').title()}"
                    if step.get("reference"):
                        extra = f"<div class='ref'>↳ {step['reference']}</div>"
                if stype == "tool_call" and step.get("input"):
                    extra = f"<div class='ref'>args: {step['input']}</div>"
                rendered += (f"<div class='step {cls}'><span class='tag'>{label}</span>"
                             f"{step['content']}{extra}</div>")
                ph.markdown(rendered, unsafe_allow_html=True)
                if anim:
                    time.sleep(anim)

        # verdict card
        fv = result["final_verdict"]
        prop = result["proposed"]
        st.markdown(f"<div class='verdict-card v-{fv}'>"
                    f"<p class='verdict-title'>{VERDICT_LABEL[fv]}</p></div>",
                    unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{prop['confidence']:.0%}")
        m2.metric("Dollar impact", f"${prop['dollars_impact']:.2f}")
        m3.metric("Agent proposed", prop["verdict"],
                  delta="overridden by governance" if result["overridden"] else "upheld",
                  delta_color="inverse" if result["overridden"] else "normal")

        if result["guardrails_triggered"]:
            st.warning("**Governance guardrail fired:** " +
                       " ".join(result["guardrails_triggered"]))

        # ---- Provider & risk intelligence (prediction + clustering + time-series) ----
        st.markdown("**Provider & risk intelligence**")
        risk = predict_claim_risk(claim)["data"]
        peer = provider_peer_outlier(claim["provider_id"])["data"]
        temp = detect_temporal_anomaly(claim["provider_id"])["data"]
        i1, i2, i3 = st.columns(3)
        i1.metric("ML risk score", f"{risk['probability']:.0%}",
                  delta=risk["level"], delta_color="off")
        i2.metric("Peer group", peer.get("cluster", "—"),
                  delta="outlier" if peer.get("is_outlier") else "typical",
                  delta_color="inverse" if peer.get("is_outlier") else "normal")
        i3.metric("Temporal signal", f"z={temp.get('peak_z', 0):+.1f}",
                  delta="recent spike" if temp.get("is_anomaly") else "stable",
                  delta_color="inverse" if temp.get("is_anomaly") else "normal")
        st.caption("Risk = trained model · Peer group = KMeans clustering · "
                   "Temporal = time-series anomaly. The risk score is advisory; verdicts "
                   "come from concrete rule, anomaly, and provider signals.")

        if prop["rules_applied"]:
            st.markdown("**Rules applied**")
            for r in prop["rules_applied"]:
                rid = r.get("rule_id") or "—"
                st.markdown(f"- `{rid}` {r.get('description','')}  \n"
                            f"<span class='ref'>{r.get('reference','')}</span>",
                            unsafe_allow_html=True)

        with st.expander("🔒 Governance audit record (immutable log)"):
            st.json(result["audit_record"])

        # ---- Clinical decision support (advisory; separate axis from payment) ----
        clin = result["clinical"]
        st.markdown("**🩺 Clinical decision support (advisory)**")
        st.caption(clin["note"])
        for f in clin["findings"]:
            cls = f"s-{f.get('finding', 'info')}"
            ref = f"<div class='ref'>↳ {f['reference']}</div>" if f.get("reference") else ""
            st.markdown(f"<div class='step {cls}'>{f['content']}{ref}</div>",
                        unsafe_allow_html=True)

# ==================================================== TAB: clinical (CDS) =====
with tab_cds:
    st.caption("The Treatment axis of TPO: an agent that checks a patient's ordered care "
               "against clinical guidelines — care gaps, contraindications, and guideline "
               "alignment — with high-severity findings escalated to a clinician. Same agent "
               "pattern as the payment side; clinical findings never change payment.")
    cases = load_patient_cases()
    clabels = [f"{c['case_id']} — {c['label']}" for c in cases]
    cpick = st.selectbox("Select a patient case", clabels, index=1)  # default PT-B (the safety alert)
    case = cases[clabels.index(cpick)]

    cl, cr = st.columns([1, 1])
    with cl:
        st.subheader("Patient")
        st.markdown(f"**{case['case_id']}** · conditions `{', '.join(case['conditions'])}`")
        if case.get("current_meds"):
            st.markdown("Current meds: " + ", ".join(case["current_meds"]))
        st.markdown("Ordered: " + ", ".join(
            o.get("label") or o.get("name") or o.get("cpt", "?") for o in case["ordered"]))
        if case.get("history"):
            st.markdown("History: " + ", ".join(
                f"{h['cpt']} ({h['days_ago']}d ago)" for h in case["history"]))
        go_cds = st.button("▶ Run clinical review", type="primary", use_container_width=True)
    with cr:
        st.subheader("Clinical reasoning")
        cds_box = st.container()

    if go_cds:
        rec, ctrace = review_patient(case)
        with cds_box:
            for step in ctrace:
                stype = step["type"]; cls = f"s-{stype}"; label = STEP_ICON.get(stype, stype)
                if stype == "observation":
                    cls = f"s-{step.get('finding', 'info')}"
                    label = f"🔎 {step.get('finding', 'info').title()}"
                ref = f"<div class='ref'>↳ {step['reference']}</div>" if step.get("reference") else ""
                st.markdown(f"<div class='step {cls}'><span class='tag'>{label}</span>"
                            f"{step['content']}{ref}</div>", unsafe_allow_html=True)

        recmap = {"ALIGNED": ("v-APPROVE", "✅ Guideline-aligned"),
                  "ADDRESS_CARE_GAP": ("v-ROUTE_TO_HUMAN", "📋 Address care gap"),
                  "CLINICAL_ALERT": ("v-ROUTE_TO_HUMAN", "⚠️ Clinical alert"),
                  "ROUTE_TO_CLINICIAN": ("v-DENY", "🩺 Route to clinician — safety review")}
        rcls, rlabel = recmap.get(rec["recommendation"], ("v-ROUTE_TO_HUMAN", rec["recommendation"]))
        st.markdown(f"<div class='verdict-card {rcls}'><p class='verdict-title'>{rlabel}</p></div>",
                    unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        s1.metric("Severity", rec["severity"].title())
        s2.metric("Confidence", f"{rec['confidence']:.0%}")
        st.markdown(rec["rationale"])
        if rec["guidelines_cited"]:
            st.markdown("**Guidelines cited**")
            for g in rec["guidelines_cited"]:
                st.markdown(f"- `{g.get('rule_id', '—')}` {g.get('description', '')}  \n"
                            f"<span class='ref'>{g.get('reference', '')}</span>",
                            unsafe_allow_html=True)

# ============================================================ TAB 2: batch ===
with tab2:
    st.subheader("Batch adjudication")
    st.caption("Run the agent across the whole synthetic claim batch.")
    if st.button("▶ Run full batch", type="primary"):
        rows, disp = [], {"APPROVE": 0, "DENY": 0, "ROUTE_TO_HUMAN": 0}
        prog = st.progress(0.0)
        for i, c in enumerate(claims):
            r = review_claim(c, history, mode="demo")  # batch always uses fast reasoner
            disp[r["final_verdict"]] += 1
            rows.append({"Claim": c["claim_id"], "Scenario": c["scenario_label"],
                         "Proposed": r["proposed"]["verdict"], "Final": r["final_verdict"],
                         "Confidence": r["proposed"]["confidence"],
                         "$ Impact": r["proposed"]["dollars_impact"],
                         "Guardrail": "; ".join(g.split(":")[0]
                                                for g in r["guardrails_triggered"]) or "—"})
            prog.progress((i + 1) / len(claims))
        df = pd.DataFrame(rows)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Approved", disp["APPROVE"])
        c2.metric("⛔ Denied", disp["DENY"])
        c3.metric("⚖️ To human", disp["ROUTE_TO_HUMAN"])
        c4.metric("$ Flagged/denied",
                  f"${df[df['Final'] != 'APPROVE']['$ Impact'].sum():.0f}")
        st.bar_chart(pd.Series(disp))
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={"$ Impact": st.column_config.NumberColumn(format="$%.2f"),
                                    "Confidence": st.column_config.NumberColumn(format="%.2f")})

    st.divider()
    st.subheader("System evaluation")
    st.caption("How well the full pipeline (rules + ML + governance) flags problematic claims "
               "on a larger labelled synthetic test set — flagged = any verdict other than approve.")

    def _show_eval(res):
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Accuracy", f"{res['accuracy']:.1%}")
        e2.metric("Precision", f"{res['precision']:.1%}")
        e3.metric("Recall", f"{res['recall']:.1%}")
        e4.metric("False-positive rate", f"{res['fpr']:.1%}")
        e5.metric("Avg time / claim", f"{res['avg_ms']:.0f} ms")
        cm = res["confusion"]
        cdf = pd.DataFrame(
            [[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]],
            index=["Actually problematic", "Actually clean"],
            columns=["Flagged", "Approved"])
        st.markdown(f"**Confusion matrix** ({res['n']} test claims)")
        st.dataframe(cdf, use_container_width=True)

    try:
        cached = json.load(open(os.path.join(os.path.dirname(__file__), "data", "eval_results.json")))
        _show_eval(cached)
        st.caption(f"Cached run of {cached['n']} claims. Metrics are imperfect by design "
                   "(labelled data includes noise), which is realistic for a claims-risk system.")
    except FileNotFoundError:
        st.info("Run `python evaluate.py` to generate evaluation metrics.")
    if st.button("▶ Re-run quick evaluation (80 claims)"):
        with st.spinner("Evaluating…"):
            _show_eval(run_eval(80))

# ==================================================== TAB 3: population =======
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

with tab3:
    st.caption("The population-level techniques from the topic: a trained risk model "
               "(prediction), KMeans peer groups (clustering), and per-provider "
               "time-series anomaly detection.")

    # --- Risk model card + feature importances ---
    st.subheader("Prediction — claim-risk model")
    try:
        meta = json.load(open(os.path.join(MODELS_DIR, "risk_meta.json")))
        a, b, c = st.columns(3)
        a.metric("Holdout ROC-AUC", meta["auc"])
        b.metric("Accuracy", meta["accuracy"])
        c.metric("Training claims", f"{meta['n_train'] + meta['n_test']:,}")
        imp = pd.DataFrame({"feature": meta["feature_names"], "importance": meta["importances"]})
        imp = imp.sort_values("importance", ascending=False).set_index("feature")
        st.bar_chart(imp, horizontal=True)
    except FileNotFoundError:
        st.warning("Run `python build_models.py` first to train the model.")

    st.divider()

    # --- Clustering scatter ---
    st.subheader("Clustering — provider peer groups")
    try:
        pdata = json.load(open(os.path.join(DATA_DIR, "providers.json")))["providers"]
        pdf = pd.DataFrame(pdata)
        pdf["marker"] = np.where(pdf["is_outlier"], 90, 28)
        pdf["group"] = np.where(pdf["is_outlier"], "⚠ peer outlier", pdf["cluster_name"])
        st.scatter_chart(pdf, x="viz_x", y="viz_y", color="group", size="marker", height=360)
        st.caption("Each point is a provider (billing profile projected to 2-D). "
                   "Large points are peer-relative outliers.")
        outs = pdf[pdf["is_outlier"]][["provider_id", "cluster_name", "avg_billed",
                                       "claim_volume", "centroid_distance"]]
        st.markdown("**Flagged peer outliers**")
        st.dataframe(outs, hide_index=True, use_container_width=True,
                     column_config={"avg_billed": st.column_config.NumberColumn(format="$%.0f"),
                                    "claim_volume": st.column_config.NumberColumn(format="%.0f")})
    except FileNotFoundError:
        st.warning("Run `python build_models.py` first to build provider clusters.")

    st.divider()

    # --- Time-series anomaly ---
    st.subheader("Time-series — provider daily volume")
    try:
        ts = json.load(open(os.path.join(DATA_DIR, "provider_timeseries.json")))
        options = ["P-213"] + [p for p in ts if p != "P-213"]
        sel = st.selectbox("Provider", options, index=0)
        series = ts[sel]
        sdf = pd.DataFrame({"date": pd.to_datetime(series["dates"]), "claims": series["counts"]})
        st.line_chart(sdf.set_index("date"))
        res = detect_temporal_anomaly(sel)
        (st.error if res["finding"] == "anomaly" else st.success)(res["detail"])
    except FileNotFoundError:
        st.warning("Run `python build_models.py` first to generate time series.")

# ==================================================== TAB: Policy Q&A (RAG) ===
with tab_rag:
    st.caption("Retrieval-Augmented Generation over policy & guideline text. Ask in plain "
               "English → Aegis retrieves the most relevant passages and grounds its answer "
               f"in them with citations. Retriever: {rag_backend()}. In production this "
               "corpus is full LCD/NCD coverage policies and clinical guidelines "
               "(the bridge to Topic 3: policy text → rules).")
    examples = ["Can a basic and comprehensive metabolic panel be billed together?",
                "Is metformin safe in advanced kidney disease?",
                "Is an MRI appropriate for acute low back pain?",
                "When is an automated denial escalated to a human?"]
    ex = st.selectbox("Example questions", ["(type your own)"] + examples)
    q = st.text_input("Your policy question",
                      value="" if ex.startswith("(") else ex,
                      placeholder="e.g. Can I bill 80048 with 80053?")
    dom = st.radio("Scope", ["All", "Payment", "Clinical"], horizontal=True)
    if st.button("🔎 Retrieve & answer", type="primary") and q.strip():
        domain = None if dom == "All" else dom.lower()
        res = rag_answer(q, k=3, live=(mode_key == "live"), domain=domain)
        st.markdown("**Answer** " + ("*(LLM, grounded in retrieved passages)*"
                                     if res["mode"] == "llm" else "*(extractive, grounded)*"))
        st.info(res["answer"])
        st.markdown("**Retrieved passages** (what the answer is grounded in)")
        for p in res["passages"]:
            st.markdown(f"<div class='step s-info'><span class='tag'>score "
                        f"{p['score']:.2f} · {p['domain']}</span>{p['text']}"
                        f"<div class='ref'>↳ {p['id']} · {p['source']}</div></div>",
                        unsafe_allow_html=True)

# ============================================================ TAB 4: rules ====
with tab4:
    st.subheader("Policy knowledge base")
    st.caption("Illustrative synthetic rules the agent's tools evaluate against. "
               "In production these would be maintained NCCI edits, coverage policies, "
               "and plan payment rules — the bridge to Topic 3 (policy → code).")
    with open(os.path.join(DATA_DIR, "policy_kb.json")) as f:
        kb = json.load(f)
    st.json(kb, expanded=False)
