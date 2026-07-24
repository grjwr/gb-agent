"""Day 6: does nearest-centroid distance separate in-distribution from OOD claims?"""
import sys, os, pickle, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gb.fit  # noqa
from sklearn.metrics import roc_auc_score

EMB = "/home/akumar/embeddings/ERNIE_Precomputed"
ART = "/home/akumar/gb-agent/artifacts"
ind = sys.argv[1] if len(sys.argv) > 1 else "isot"
oods = sys.argv[2].split(",") if len(sys.argv) > 2 else ["buzzfeed","politifact","liar2_binary"]

d = pickle.load(open(f"{ART}/gb_{ind}.pkl","rb"))
C, mu, sigma = d["centroids"], d["mu"], d["sigma"]

def nearest(ds):
    X = np.load(f"{EMB}/{ds}_embeddings.npy").astype(np.float64)
    Xz = (X - mu) / sigma
    out = np.empty(len(Xz))
    for i in range(0, len(Xz), 2000):
        b = Xz[i:i+2000]
        out[i:i+2000] = np.sqrt(((b[:,None,:]-C[None,:,:])**2).sum(-1)).min(1)
    return out

d_in = nearest(ind)[d["test_idx"]]
tau = float(np.percentile(d_in, 95))
print(f"in-dist={ind}  n_balls={len(C)}  n_test={len(d_in)}")
print(f"in-dist distance: mean={d_in.mean():.2f} p50={np.percentile(d_in,50):.2f} "
      f"p95={tau:.2f}\ntau (p95) = {tau:.2f}\n")

rows=[]
for ds in oods:
    if ds==ind: continue
    do = nearest(ds)
    auc = roc_auc_score(np.r_[np.zeros(len(d_in)),np.ones(len(do))], np.r_[d_in,do])
    flagged = float((do>tau).mean())
    print(f"{ds:14s} n={len(do):6d} mean={do.mean():7.2f} p50={np.percentile(do,50):7.2f} "
          f"AUC={auc:.3f} flagged@tau={flagged:.3f}")
    rows.append(dict(ood=ds,auc=round(float(auc),4),ood_flagged=round(flagged,4)))

json.dump({"in_dist":ind,"tau":tau,"results":rows},
          open(f"/home/akumar/gb-agent/results/tau_{ind}.json","w"),indent=2)
