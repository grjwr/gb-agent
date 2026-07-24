# GB-Agent — Prototype-Grounded Fake News Claim Verification

Agentic fake news verification built on the **GB-FFN** architecture
(Gurjwar, Kumar & Rao, *Applied Soft Computing* 2026). Granular Ball clustering
compresses ERNIE embeddings into semantic prototypes; a router escalates
low-confidence claims to an LLM arbiter with web + RAG evidence.

## Architecture
claim → ERNIE [CLS] → GB assignment → route
  ├─ near a pure prototype → fast-path verdict
  └─ far OR ambiguous prototype → LLM arbiter (web search + RAG)

## Published GB-FFN benchmarks (cited, not re-derived)
F1: ISOT 97.47 · GossipCop 97.96 · WELFake 96.12 · PolitiFact 91.71

## Week 1 — GB layer (built & characterized)
- ERNIE embeddings for ISOT, GossipCop, WELFake, PolitiFact (frozen, [CLS], L=128).
- Granular Ball fitting: recursive 2-means, purity θ=0.95, adaptive min-sample-size.
- **Role of GB in this agent: compression + routing geometry, not classification.**
  The fast-path verdict uses raw-embedding classification (F1 93–97); GB provides
  prototype distance and purity for routing.
- Routing rule (option 1, grounded in paper quantities): escalate if
  `distance > τ` OR `nearest_ball_purity < 0.80`.
- τ calibrated on ISOT (p95 = 29.66). Distance separates **out-of-genre** novelty
  (LIAR2 AUC 0.97) but not **out-of-topic** novelty in same-genre prose
  (BuzzFeed/PolitiFact/GossipCop AUC 0.55–0.66). Router reframed as an
  abstention/confidence signal accordingly.
- `GBAssigner` tested: escalation rate LIAR2 95% vs ISOT 33%.

## Layout
    gb/            fitting + assignment
    scripts/       fit, sweep, tau calibration, FFN characterization
    artifacts/     fitted GB pickles (gitignored)
    tests/

## Status
Week 1 complete. Week 2: LangGraph assembly + live ERNIE + Qwen3-8B arbiter.
