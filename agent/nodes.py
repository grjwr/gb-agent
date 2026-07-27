"""Hybrid GBFFN + Gemini nodes. GB verdicts the easy cases; Gemini handles escalations."""
import os, numpy as np
from agent.embed import ErnieEmbedder
from gb.assign import GBAssigner
from agent.websearch import web_search
from agent.arbiter_gemini import GeminiArbiter

ART = "/home/akumar/gb-agent/artifacts"
DATASET = os.environ.get("GB_DATASET", "isot")
TAU = float(os.environ.get("GB_TAU", "31.80"))

_emb = _asg = _gem = None
def _lazy():
    global _emb, _asg, _gem
    if _emb is None:
        _emb = ErnieEmbedder()
        _asg = GBAssigner(f"{ART}/gb_{DATASET}.pkl", tau=TAU, purity_floor=0.80)
        _gem = GeminiArbiter()
    return _emb, _asg, _gem

def embed_node(state):
    emb, _, _ = _lazy()
    return {"embedding": emb.embed(state["claim"])[0].tolist()}

def gb_node(state):
    _, asg, _ = _lazy()
    d = asg.assign(np.array(state["embedding"]))
    # GB's own verdict from the nearest ball
    label = "TRUE" if d["ball_label"] == 1 else "FALSE"
    return {"gb": d,
            "route": "escalate" if d["escalate"] else "fast_path",
            "gb_verdict": {"verdict": label, "confidence": round(d["ball_purity"], 3),
                           "rationale": f"GBFFN: claim near ball {d['ball_id']} "
                                        f"(purity {d['ball_purity']}, dist {d['distance']})",
                           "source": "GBFFN-fast-path"}}

def route_selector(state):
    return state["route"]

def fast_path_node(state):
    # GB is confident -> use its verdict directly
    return {"verdict": state["gb_verdict"]}

def escalate_node(state):
    # GB unsure -> gather web evidence, let Gemini decide
    _, _, gem = _lazy()
    ev = web_search(state["claim"], k=5)
    v = gem.verdict(state["claim"], ev)
    v["source"] = "Gemini+web (GBFFN escalated)"
    return {"evidence": ev, "verdict": v}
