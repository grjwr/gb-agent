"""Option C: fit granular balls on ALL four datasets combined -> one prototype space."""
import sys, os, pickle, json, time
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.insert(0, "/home/akumar/gb-agent")
from gb.fit import fit_granular_balls, GBConfig

EMB = "/home/akumar/embeddings/GB_AGENT_v2"
OUT = "/home/akumar/gb-agent/artifacts"
DATASETS = ["isot", "welfake", "gossipcop", "politifact"]

Xs, ys = [], []
for ds in DATASETS:
    X = np.load(f"{EMB}/{ds}_embeddings.npy").astype(np.float64)
    y = np.load(f"{EMB}/{ds}_labels.npy").astype(int)
    Xs.append(X); ys.append(y)
    print(f"{ds}: {X.shape} balance={np.bincount(y)}")

X = np.vstack(Xs); y = np.concatenate(ys)
print(f"MERGED: {X.shape} balance={np.bincount(y)}")

# single global split + standardization (train-only mu/sigma)
tr, te = train_test_split(np.arange(len(X)), test_size=0.30, stratify=y, random_state=42)
mu, sigma = X[tr].mean(0), X[tr].std(0); sigma[sigma == 0] = 1.0
Xz = (X - mu) / sigma

# min_sample_size scaled to the larger merged N (adaptive rule)
cfg = GBConfig(theta=0.95, min_sample_size=None, seed=42)  # None -> max(40, N/100)
t0 = time.time()
balls, stats = fit_granular_balls(Xz[tr], y[tr], cfg)
for b in balls: b.member_idx = tr[b.member_idx]
elapsed = time.time() - t0

purities = np.array([b.purity for b in balls])
stats.update({"dataset": "merged_all4", "n_train": len(tr), "n_test": len(te),
              "fit_seconds": round(elapsed, 1),
              "purity_mean": float(purities.mean()),
              "pct_at_theta": float((purities >= 0.95).mean()),
              "datasets": DATASETS})

pickle.dump({"balls": balls, "mu": mu, "sigma": sigma, "train_idx": tr, "test_idx": te,
             "stats": stats, "centroids": np.stack([b.centroid for b in balls]),
             "ball_labels": np.array([b.label for b in balls]),
             "ball_purities": purities}, open(f"{OUT}/gb_merged.pkl", "wb"))
print(json.dumps(stats, indent=2))
print(f"-> {OUT}/gb_merged.pkl")
