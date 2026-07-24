"""Day 5b: GB-FFN with size-weighted loss and/or soft labels. Paper params unchanged."""
import sys, os, pickle, json, argparse
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gb.fit  # noqa
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

EMB = "/home/akumar/embeddings/GB_AGENT_v2"
ART = "/home/akumar/gb-agent/artifacts"

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--weighted", action="store_true", help="weight loss by ball size")
p.add_argument("--soft", action="store_true", help="use ball purity as soft target")
p.add_argument("--min-purity", type=float, default=0.0, help="drop balls below this purity")
a = p.parse_args()

d = pickle.load(open(f"{ART}/gb_{a.dataset}.pkl", "rb"))
X = np.load(f"{EMB}/{a.dataset}_embeddings.npy").astype(np.float64)
y = np.load(f"{EMB}/{a.dataset}_labels.npy").astype(int)
Xz = (X - d["mu"]) / d["sigma"]
Xte, yte = Xz[d["test_idx"]], y[d["test_idx"]]

balls = d["balls"]
keep = np.array([b.purity >= a.min_purity for b in balls])
C   = d["centroids"][keep]
lab = d["ball_labels"][keep].astype(np.float64)
pur = d["ball_purities"][keep]
siz = np.array([b.size for b in balls])[keep].astype(np.float64)

# soft target: fraction of the ball that is class 1
target = np.where(lab == 1, pur, 1.0 - pur) if a.soft else lab
w = siz / siz.mean() if a.weighted else np.ones_like(siz)

print(f"{a.dataset}: balls={len(C)}/{len(balls)} kept  weighted={a.weighted} "
      f"soft={a.soft} min_purity={a.min_purity}  test={Xte.shape}")

def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = nn.Sequential(nn.Linear(C.shape[1], 128), nn.ReLU(),
                        nn.Dropout(0.2), nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.tensor(C, dtype=torch.float32)
    yt = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
    wt = torch.tensor(w, dtype=torch.float32).unsqueeze(1)
    n = len(Xt)
    for ep in range(a.epochs):
        perm = torch.randperm(n); net.train()
        for i in range(0, n, 32):
            b = perm[i:i+32]
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(Xt[b]), yt[b], weight=wt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(net(torch.tensor(Xte, dtype=torch.float32))) > 0.5).int().numpy().ravel()
    return dict(acc=accuracy_score(yte,pred)*100, prec=precision_score(yte,pred,zero_division=0)*100,
                rec=recall_score(yte,pred,zero_division=0)*100, f1=f1_score(yte,pred,zero_division=0)*100)

rs = [run(s) for s in range(a.seeds)]
res = {k: (round(float(np.mean([r[k] for r in rs])),2), round(float(np.std([r[k] for r in rs])),2)) for k in rs[0]}
print("  ".join(f"{k}={v[0]:.2f}±{v[1]:.2f}" for k,v in res.items()))
