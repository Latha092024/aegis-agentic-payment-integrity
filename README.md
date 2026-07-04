# Aegis — Agentic Payment-Integrity Analyst

An agentic AI proof-of-concept that reviews medical insurance claims and decides whether they should be paid — across treatment, payment, and operations — with a governance layer that keeps a human in the loop.

## What's in this repo
- **Aegis_Cotiviti_POC/** — the proof-of-concept code (Python / Streamlit)
- **Aegis_Presentation.pptx** — slide deck
- **Agentic AI ... .docx / .pdf** — written report
- **Aegis.mp4** — recorded demo video

## How the POC works
A single AI agent reviews a claim: it plans which checks to run, calls its tools (unbundling, medical necessity, billed-amount anomaly, duplicates, a trained ML risk model, provider clustering, and time-series anomaly detection), reasons over the results, and returns a verdict — approve, deny, or route to a human — inside a governance layer that keeps a human in control.

## Run the POC
cd Aegis_Cotiviti_POC  
pip install -r requirements.txt
python build_models.py
python -m streamlit run app.py

Open http://localhost:8501 and use Demo mode.

## Note
Claims are synthetic (no real patient data). Billed-amount baselines use real public CMS Medicare charge data. Rules are illustrative for a proof-of-concept.
