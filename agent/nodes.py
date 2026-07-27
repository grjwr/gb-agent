"""Hybrid nodes with multi-dataset GB routing (all four benchmark ball-sets)."""
import os, numpy as np
from agent.embed import ErnieEmbedder
from gb.assign_multi import MultiGBAssigner
from agent.websearch import web_search
from agent.arbiter_gemini import GeminiArbiter

ART = "/home/akumar/gb-agent/artifacts"
PATHS = {"isot": f"{ART}/gb_isot.pkl", "welfake": f"{ART}/gb_welfake.pkl",
         "gossipcop": f"{ART}/gb_gossipcop.pkl", "politifact": f"{ART}/gb_politifact.pkl"}
TAUS = {"isot": 31.80, "welfake": 30.29, "gossipcop": 31.27, "politifact": 36.26}

_emb = _asg = _gem = None
def _lazy():
    global _emb, _asg, _gem
    if _emb is None:
        _emb = ErnieEmbedder()
        _asg = MultiGBAssigner(PATHS, TAUS, purity_floor=0.80)
        _gem = GeminiArbiter()
    return _emb, _asg, _gem

def embed_node(state):
    emb, _, _ = _lazy()
    return {"embedding": emb.embed(state["claim"])[0].tolist()}

def gb_node(state):
    _, asg, _ = _lazy()
    d = asg.assign(np.array(state["embedding"]))
    label = "TRUE" if d["ball_label"] == 1 else "FALSE"
    return {"gb": d,
            "route": "escalate" if d["escalate"] else "fast_path",
            "gb_verdict": {"verdict": label, "confidence": round(d["ball_purity"], 3),
                           "rationale": f"GBFFN: nearest prototype in '{d['dataset']}' "
                                        f"(ball {d['ball_id']}, purity {d['ball_purity']}, "
                                        f"dist {d['distance']:.1f})",
                           "source": f"GBFFN-fast-path ({d['dataset']})"}}

def route_selector(state):
    return state["route"]

def fast_path_node(state):
    return {"verdict": state["gb_verdict"]}

def escalate_node(state):
    _, _, gem = _lazy()
    ev = web_search(state["claim"], k=5)
    v = gem.verdict(state["claim"], ev)
    v["source"] = "Gemini+web (GBFFN escalated)"
    return {"evidence": ev, "verdict": v}
