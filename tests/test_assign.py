import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gb.assign import GBAssigner

EMB = "/home/akumar/embeddings/ERNIE_Precomputed"
ART = "/home/akumar/gb-agent/artifacts"

a = GBAssigner(f"{ART}/gb_isot.pkl", tau=29.66, purity_floor=0.80)
print(f"loaded {a.M} balls, tau={a.tau}, purity_floor={a.purity_floor}")

# single call sanity
d = pickle.load(open(f"{ART}/gb_isot.pkl", "rb"))
Xisot = np.load(f"{EMB}/isot_embeddings.npy")
r = a.assign(Xisot[d["test_idx"][0]])
print("sample decision:", r)
assert set(r) >= {"ball_id","distance","escalate","reason"}
assert 0 <= r["ball_id"] < a.M

# escalation rates: ISOT test (should be low) vs LIAR2 (should be high)
isot_dec = a.assign_batch(Xisot[d["test_idx"][:500]])
liar = np.load(f"{EMB}/liar2_binary_embeddings.npy")[:500]
liar_dec = a.assign_batch(liar)
er_isot = np.mean([x["escalate"] for x in isot_dec])
er_liar = np.mean([x["escalate"] for x in liar_dec])
print(f"escalation rate  ISOT(in-dist)={er_isot:.3f}   LIAR2(ood)={er_liar:.3f}")
assert er_liar > er_isot, "OOD should escalate more than in-dist"
print("\nALL CHECKS PASSED")
