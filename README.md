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

*The following are results from this agent system's own experiments (reproducible via the scripts in this repo), complementary to the classification benchmarks reported in the paper.*

- **Strong out-of-distribution routing.** Distance-to-prototype cleanly separates unfamiliar claim-style text (LIAR2) from the article-style training data, with **AUC 0.97–0.99** across all four datasets — a reliable signal for deciding when to escalate.
- **Effective cascading hybrid.** GB-FFN resolves claims that match known prototypes locally, while ambiguous or novel claims escalate to live web evidence and LLM reasoning — combining the speed of a lightweight model with the accuracy of grounded verification.
- **Deployed and working end-to-end.** A public, interactive web app runs the full pipeline (ERNIE → GB-FFN → web → LLM) on free infrastructure, with the ERNIE encoder and four GB ball-sets loaded at runtime.

---

## Limitations

- **A single merged prototype space does not generalize across domains.** When all four datasets are pooled into one Granular-Ball space, cluster purity collapses from ~0.80 (per-dataset) to ~0.59 — near the base rate — indicating that veracity structure is **domain-specific** rather than shared across news genres. The system works around this by maintaining four separate, individually-clean ball-sets and routing across them, rather than relying on one generalized space. (Reproducible via `scripts/fit_gb_merged.py`.)
- **Prototype purity is not a truth signal.** A claim landing near a high-purity ball reflects similarity to a clean *training* neighborhood, not the factual truth of the claim. This is precisely why the fast path is conservative and escalates uncertain cases to evidence-grounded reasoning rather than trusting the prototype label alone.
- **Short claims often escalate.** User-typed claims are stylistically unlike the article-length training text, so many are routed to the LLM path. This is the safe direction, but it means GB-FFN acts more as a router than a standalone classifier for short-form input.

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
