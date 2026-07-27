# 🔍 GB-Agent — Granular-Ball FFN + LLM Hybrid for Fake-News Verification

**Live demo:** https://gb-agent-rajiv-kumar.streamlit.app/

An agentic fake-news verification system that combines a **Granular-Ball Feedforward Network (GB-FFN)** — a lightweight semantic-prototype router — with a **frontier LLM arbiter** and **live web evidence**. Built on the GB-FFN model from our published work and extended into a deployable cascading-inference agent.

---

## What it does

Type a claim → get an evidence-grounded verdict (TRUE / FALSE / UNVERIFIABLE) with a rationale and sources.

```
claim
 → ERNIE embedding
 → GB-FFN router  (nearest semantic prototype across 4 benchmark ball-sets)
      ├─ confident (near a pure prototype)  → GB-FFN verdict            [fast, local]
      └─ uncertain (far OR ambiguous ball)  → escalate:
             → DuckDuckGo live web search
             → Gemini reads evidence → grounded verdict                [deep]
 → verdict + confidence + rationale + evidence
```

The design is **cascading inference**: the lightweight GB-FFN handles claims that match known prototypes; uncertain claims escalate to web-grounded LLM reasoning. GB-FFN acts as both a fast classifier and an out-of-distribution router.

---

## Architecture

| Stage | Component | Role |
|-------|-----------|------|
| Embedding | ERNIE 2.0 base (`[CLS]`, frozen) | Dense semantic representation |
| Routing | **GB-FFN** (Granular-Ball clustering) | Nearest-prototype match + escalate decision |
| Evidence | DuckDuckGo (`ddgs`) | Free live web retrieval, no API key |
| Arbiter | Gemini 3.5 Flash | Evidence-grounded verdict (forced JSON) |
| Orchestration | LangGraph | Node graph + conditional routing |
| UI / Deploy | Streamlit Community Cloud | Public web app |

Routing runs against **all four benchmark datasets** (ISOT, WELFake, GossipCop, PolitiFact) simultaneously: a claim is matched to its nearest prototype across every set, with per-dataset distance thresholds (τ) normalized for fair comparison.

---

## Key findings

- **GB-FFN prototype routing detects out-of-genre novelty well.** Distance-to-prototype separates claim-style text (LIAR2) from article-style training data with **AUC 0.97–0.99** across all four datasets.
- **Veracity structure is domain-specific.** Naively merging all four datasets into one prototype space collapses purity from ~0.80 to ~0.59 (near base rate), so the system keeps four *separate* clean ball-sets and routes across them rather than blending. (Documented as a negative result in `scripts/fit_gb_merged.py`.)
- **Cascading design keeps the fast path safe.** Because prototype purity reflects the training neighborhood — not the truth of a new claim — GB-FFN escalates ambiguous cases to evidence-grounded LLM reasoning rather than guessing.

---

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"     # free key: aistudio.google.com
streamlit run app.py
```

ERNIE weights are pulled automatically from Hugging Face (`grjwr/ernie-base-en-gb`). GB ball-sets ship in `artifacts/`.

---

## Repository layout

```
agent/          embedding, GB routing nodes, Gemini arbiter, web search, LangGraph graph
gb/             Granular-Ball fitting and multi-dataset assignment
artifacts/      fitted GB ball-sets (ISOT, WELFake, GossipCop, PolitiFact)
scripts/        embedding generation, GB fitting, tau calibration, ablations
results/        per-dataset FFN and tau-calibration results
app.py          Streamlit web app
```

---

## Reference

This system is built on the GB-FFN model introduced in:

> R. K. Gurjwar, A. Kumar, U. P. Rao. **"Granular ball feedforward network for fake news detection."** *Applied Soft Computing*, 202 (2026) 115812. https://doi.org/10.1016/j.asoc.2026.115812

Related prior work — a lightweight real-time detector:

> R. K. Gurjwar, A. Kumar, U. P. Rao. **"EPRVFL: A fast and scalable model for real-time fake news detection."** *Pattern Recognition Letters*, 196 (2025) 267–273. https://doi.org/10.1016/j.patrec.2025.06.006

---

## Tech stack

ERNIE 2.0 · Granular-Ball clustering · LangGraph · Gemini API · DuckDuckGo search · Streamlit · PyTorch · scikit-learn

> *Note: the GB-FFN model is CPU-light. Heavier 30B-scale arbiter evaluation was run separately on GPU (H100) for benchmarking and is not part of the live demo.*
