"""
rag.py - Retrieval-Augmented Generation over policy & guideline text
--------------------------------------------------------------------
Structured rules (tools.py) are great for exact, auditable edits. But real
coverage policies and clinical guidelines are unstructured PROSE. RAG is the
right tool there: retrieve the passages relevant to a question, then ground an
answer (or an agent's citation) in them.

Retriever backends:
  - "tfidf" (default): sparse TF-IDF + cosine. Offline, no downloads, no key.
  - "dense" (optional): sentence-transformers embeddings for true semantic
      search. Enable with AEGIS_RAG_BACKEND=dense (falls back to tfidf if the
      package isn't installed).

Corpus = the prose text already in policy_kb.json and clinical_kb.json. In
production you'd point this at full LCD/NCD coverage policies and guideline
documents (the bridge to Topic 3: policy text -> rules/citations).
"""
import json
import os
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@lru_cache(maxsize=1)
def _kbs():
    with open(os.path.join(_DATA, "policy_kb.json")) as f:
        pay = json.load(f)
    with open(os.path.join(_DATA, "clinical_kb.json")) as f:
        clin = json.load(f)
    return pay, clin


@lru_cache(maxsize=1)
def build_corpus():
    """Turn the structured KBs into a small document corpus for retrieval."""
    pay, clin = _kbs()
    docs = []
    for k, d in pay.get("policy_docs", {}).items():
        docs.append({"id": f"policy:{k}", "text": d["text"], "source": d["reference"],
                     "domain": "payment"})
    for e in pay.get("ncci_edits", []):
        docs.append({"id": e["rule_id"], "text": e["rationale"], "source": e["reference"],
                     "domain": "payment"})
    for m in pay.get("medical_necessity", []):
        docs.append({"id": m["rule_id"], "text": m["rationale"], "source": m["reference"],
                     "domain": "payment"})
    for c in clin.get("contraindications", []):
        docs.append({"id": c["rule_id"], "text": c["detail"], "source": c["reference"],
                     "domain": "clinical"})
    for a in clin.get("alignment", []):
        docs.append({"id": a["rule_id"], "text": a["detail"], "source": a["reference"],
                     "domain": "clinical"})
    for g in clin.get("care_guidelines", []):
        docs.append({"id": g["rule_id"], "text": g["recommended"]["label"],
                     "source": g["reference"], "domain": "clinical"})
    # expanded corpus of public CMS/NCCI/Medicare policy summaries (if present)
    corpus_path = os.path.join(_DATA, "policy_corpus.json")
    if os.path.exists(corpus_path):
        for d in json.load(open(corpus_path)).get("documents", []):
            docs.append({"id": d["id"], "text": d["text"], "source": d["source"],
                         "domain": d.get("domain", "payment")})
    return docs


# --------------------------------------------------------------- TF-IDF index
@lru_cache(maxsize=1)
def _tfidf_index():
    corpus = build_corpus()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform([d["text"] for d in corpus])
    return vec, mat, corpus


# ---------------------------------------------------- optional dense index ---
@lru_cache(maxsize=1)
def _dense_index():
    from sentence_transformers import SentenceTransformer  # optional dependency
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = build_corpus()
    emb = model.encode([d["text"] for d in corpus], normalize_embeddings=True)
    return model, np.asarray(emb), corpus


def backend_name():
    if os.environ.get("AEGIS_RAG_BACKEND") == "dense":
        try:
            _dense_index()
            return "dense (sentence-transformers, semantic)"
        except Exception:
            return "sparse (TF-IDF) — dense backend unavailable"
    return "sparse (TF-IDF, cosine)"


def retrieve(query, k=3, domain=None):
    """Return the top-k passages for a query as
    [{id, text, source, domain, score}], highest score first."""
    use_dense = os.environ.get("AEGIS_RAG_BACKEND") == "dense"
    if use_dense:
        try:
            model, emb, corpus = _dense_index()
            qv = model.encode([query], normalize_embeddings=True)[0]
            sims = emb @ qv
        except Exception:
            use_dense = False
    if not use_dense:
        vec, mat, corpus = _tfidf_index()
        sims = linear_kernel(vec.transform([query]), mat).flatten()

    order = np.argsort(sims)[::-1]
    out = []
    for i in order:
        if domain and corpus[i]["domain"] != domain:
            continue
        out.append({**corpus[i], "score": round(float(sims[i]), 3)})
        if len(out) >= k:
            break
    return out


def answer(query, k=3, live=False, model="claude-sonnet-4-6", domain=None):
    """Retrieve, then generate a grounded answer.
    live + ANTHROPIC_API_KEY -> LLM answer grounded in retrieved passages.
    otherwise -> extractive grounded answer (offline, deterministic)."""
    passages = retrieve(query, k, domain=domain)
    if not passages or passages[0]["score"] < 0.02:
        return {"answer": "No sufficiently relevant policy passage was found for this question.",
                "passages": passages, "mode": "extractive"}

    if live and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from anthropic import Anthropic
            ctx = "\n".join(f"[{p['id']}] {p['text']} (source: {p['source']})" for p in passages)
            msg = Anthropic().messages.create(
                model=model, max_tokens=400,
                system=("Answer the user's policy question using ONLY the provided context. "
                        "Cite the [id] of any passage you use. If the context is insufficient, say so."),
                messages=[{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}])
            text = "".join(b.text for b in msg.content if b.type == "text")
            return {"answer": text.strip(), "passages": passages, "mode": "llm"}
        except Exception:
            pass

    top = passages[0]
    return {"answer": f"Based on the retrieved policy, the most relevant guidance is: "
                      f"\"{top['text']}\" (source: {top['source']}).",
            "passages": passages, "mode": "extractive"}
