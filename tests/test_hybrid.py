import sys; sys.path.insert(0, "/home/akumar/gb-agent")
from agent.graph import build_graph
app = build_graph()
claims = [
    "The president signed a new infrastructure bill into law on Tuesday.",
    "Scientists confirm the moon is made entirely of compressed cheese.",
    "The Earth's average temperature has risen over the past century.",
]
for c in claims:
    out = app.invoke({"claim": c})
    v = out["verdict"]; gb = out["gb"]
    print(f"\nCLAIM: {c}")
    print(f"  route={out['route']}  GB(ball={gb['ball_id']} purity={gb['ball_purity']} dist={gb['distance']})")
    print(f"  VERDICT: {v['verdict']}  conf={v.get('confidence')}  [{v.get('source')}]")
    print(f"  rationale: {v.get('rationale','')[:150]}")
