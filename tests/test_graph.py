import sys; sys.path.insert(0, "/home/akumar/gb-agent")
from agent.graph import build_graph

app = build_graph()
claims = [
    "The president signed a new infrastructure bill into law on Tuesday.",
    "Scientists confirm the moon is made entirely of compressed cheese.",
    "Local council approves budget for road repairs next fiscal year.",
]
for c in claims:
    out = app.invoke({"claim": c})
    v = out["verdict"]; gb = out["gb"]
    print(f"\nCLAIM: {c}")
    print(f"  route={out['route']}  ball={gb['ball_id']} dist={gb['distance']} "
          f"purity={gb['ball_purity']} escalate={gb['escalate']}")
    print(f"  -> {v['rationale']}")
