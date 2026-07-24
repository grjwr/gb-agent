"""Day 5: shallow FFN on GB centroids vs NG-FFN baseline (paper Sec 3.5 / 4.6)."""
import sys, os, pickle, json, argparse
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gb.fit  # noqa: needed for unpickling GranularBall
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

EMB = "/home/akumar/embeddings/ERNIE_Precomputed"
ART = "/home/akumar/gb-agent/artifacts"

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--seeds", type=int, default=5)
a = p.parse_args()

d = pickle.load(open(f"{ART}/gb_{a.dataset}.pkl", "rb"))
X = np.load(f"{EMB}/{a.dataset}_embeddings.npy").astype(np.float64)
y = np.load(f"{EMB}/{a.dataset}_labels.npy").astype(int)
Xz = (X - d["mu"]) / d["sigma"]
tr, te = d["train_idx"], d["test_idx"]

Xte, yte = Xz[te], y[te]
Xgb, ygb = d["centroids"], d["ball_labels"]          # GB-FFN input
Xng, yng = Xz[tr], y[tr]                             # NG-FFN input (ablation)
print(f"{a.dataset}: GB train={Xgb.shape}  NG train={Xng.shape}  test={Xte.shape}")

def run(Xtr, ytr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
                        nn.Dropout(0.2), nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    n = len(Xt)
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        net.train()
        for i in range(0, n, 32):
            b = perm[i:i+32]
            opt.zero_grad(); loss = lossf(net(Xt[b]), yt[b]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(net(torch.tensor(Xte, dtype=torch.float32))) > 0.5).int().numpy().ravel()
    return dict(acc=accuracy_score(yte, pred)*100, prec=precision_score(yte, pred, zero_division=0)*100,
                rec=recall_score(yte, pred, zero_division=0)*100, f1=f1_score(yte, pred, zero_division=0)*100)

out = {}
for name, (Xtr, ytr) in {"GB-FFN": (Xgb, ygb), "NG-FFN": (Xng, yng)}.items():
    rs = [run(Xtr, ytr, s) for s in range(a.seeds)]
    out[name] = {k: (round(float(np.mean([r[k] for r in rs])), 2),
                     round(float(np.std([r[k] for r in rs])), 2)) for k in rs[0]}
    print(f"{name}: " + "  ".join(f"{k}={v[0]:.2f}±{v[1]:.2f}" for k, v in out[name].items()), flush=True)

print("\ndelta F1 (GB - NG): %+.2f" % (out["GB-FFN"]["f1"][0] - out["NG-FFN"]["f1"][0]))
json.dump(out, open(f"/home/akumar/gb-agent/results/ffn_{a.dataset}.json", "w"), indent=2)
