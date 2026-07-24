"""Sweep min_sample_size and report purity saturation. No artifacts written."""
import sys, os, json, time
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gb.fit import fit_granular_balls, GBConfig

EMB = "/home/akumar/embeddings/ERNIE_Precomputed"
ds = sys.argv[1]
grid = [int(v) for v in sys.argv[2].split(",")]

X = np.load(f"{EMB}/{ds}_embeddings.npy").astype(np.float64)
y = np.load(f"{EMB}/{ds}_labels.npy").astype(int)
tr, te = train_test_split(np.arange(len(X)), test_size=0.30, stratify=y, random_state=42)
mu, sigma = X[tr].mean(0), X[tr].std(0); sigma[sigma == 0] = 1.0
Xz = (X - mu) / sigma
base = max(np.bincount(y[tr])) / len(tr)
print(f"{ds}: n_train={len(tr)} base_rate={base:.4f}\n")

rows = []
for ms in grid:
    t0 = time.time()
    balls, st = fit_granular_balls(Xz[tr], y[tr], GBConfig(min_sample_size=ms, seed=42))
    p = np.array([b.purity for b in balls]); s = np.array([b.size for b in balls])
    w = float((p * s).sum() / s.sum())
    r = dict(min_size=ms, n_balls=len(balls), purity_mean=round(float(p.mean()), 4),
             purity_weighted=round(w, 4), lift=round(w - base, 4),
             pct_at_theta=round(float((p >= 0.95).mean()), 4),
             stopped_purity=st["stop_reasons"]["purity"],
             stopped_minsize=st["stop_reasons"]["min_size"],
             size_min=int(s.min()), secs=round(time.time() - t0, 1))
    rows.append(r); print(json.dumps(r), flush=True)

print("\nmin_size  n_balls  pur_wt   lift   pct@.95  min_sz")
for r in rows:
    print(f"{r['min_size']:8d} {r['n_balls']:8d} {r['purity_weighted']:7.4f} "
          f"{r['lift']:+6.4f} {r['pct_at_theta']:8.3f} {r['size_min']:7d}")
