"""Fit granular balls on a dataset and pickle the artifact."""
import argparse, pickle, json, time, sys, os
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gb.fit import fit_granular_balls, GBConfig

EMB = "/home/akumar/embeddings/GB_AGENT_v2"
OUT = "/home/akumar/gb-agent/artifacts"

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--theta", type=float, default=0.95)
p.add_argument("--min-size", type=int, default=None)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

X = np.load(f"{EMB}/{a.dataset}_embeddings.npy").astype(np.float64)
y = np.load(f"{EMB}/{a.dataset}_labels.npy").astype(int)
print(f"{a.dataset}: X={X.shape} y={np.bincount(y)}")

# Frozen stratified 70:30 split
idx = np.arange(len(X))
tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=a.seed)

# z-score using TRAIN ONLY (paper Sec 3.3 — avoids leakage)
mu, sigma = X[tr].mean(axis=0), X[tr].std(axis=0)
sigma[sigma == 0] = 1.0
Xz = (X - mu) / sigma

cfg = GBConfig(theta=a.theta, min_sample_size=a.min_size, seed=a.seed)
t0 = time.time()
balls, stats = fit_granular_balls(Xz[tr], y[tr], cfg)
elapsed = time.time() - t0

# member_idx is local to tr -> remap to global row ids
for b in balls:
    b.member_idx = tr[b.member_idx]

purities = np.array([b.purity for b in balls])
sizes = np.array([b.size for b in balls])
stats.update({
    "dataset": a.dataset, "theta": a.theta, "seed": a.seed,
    "n_train": len(tr), "n_test": len(te), "fit_seconds": round(elapsed, 1),
    "purity_mean": float(purities.mean()), "purity_min": float(purities.min()),
    "pct_balls_at_theta": float((purities >= a.theta).mean()),
    "size_median": int(np.median(sizes)), "size_min": int(sizes.min()),
    "label_balance": np.bincount([b.label for b in balls], minlength=2).tolist(),
})

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/gb_{a.dataset}.pkl", "wb") as f:
    pickle.dump({"balls": balls, "mu": mu, "sigma": sigma,
                 "train_idx": tr, "test_idx": te, "stats": stats,
                 "centroids": np.stack([b.centroid for b in balls]),
                 "ball_labels": np.array([b.label for b in balls]),
                 "ball_purities": purities}, f)

print(json.dumps(stats, indent=2))
print(f"-> {OUT}/gb_{a.dataset}.pkl")
