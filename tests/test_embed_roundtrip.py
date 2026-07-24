import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.embed import ErnieEmbedder
from gb.assign import GBAssigner
from sklearn.model_selection import StratifiedShuffleSplit

CSV = "/home/akumar/Dataset/ISOT.csv"
NPY = "/home/akumar/embeddings/ERNIE_Precomputed/isot_embeddings.npy"

# --- replicate load_csv + stratified_indices from 01_generate_embeddings.py ---
df = pd.read_csv(CSV); df.columns = [c.lower().strip() for c in df.columns]
tcol = next(c for c in df.columns if c in ("title","text","content","body","news","statement"))
lcol = next(c for c in df.columns if c in ("verdict","label","labels","class","target","fake"))
texts = df[tcol].fillna("").str.lower().str.strip().tolist()
y = df[lcol].astype(int).values

stored = np.load(NPY)
print(f"CSV rows={len(texts)}  stored_emb={stored.shape}")

# --- embed a few rows live, compare to stored .npy at same row index ---
emb = ErnieEmbedder()
probe = [0, 1, 100, 5000, len(texts)-1]
live = emb.embed([texts[i] for i in probe])
for k, i in enumerate(probe):
    diff = np.abs(live[k] - stored[i]).max()
    cos = np.dot(live[k], stored[i]) / (np.linalg.norm(live[k])*np.linalg.norm(stored[i]))
    print(f"row {i:6d}: max|Δ|={diff:.4f}  cos={cos:.5f}")

# --- routing sanity: live embeddings should land in-distribution ---
a = GBAssigner("/home/akumar/gb-agent/artifacts/gb_isot.pkl", tau=31.80)
for k, i in enumerate(probe):
    r = a.assign(live[k])
    print(f"row {i:6d}: dist={r['distance']:.2f} ball={r['ball_id']} "
          f"purity={r['ball_purity']:.3f} escalate={r['escalate']}")
