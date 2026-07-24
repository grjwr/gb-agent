import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, "/home/akumar/gb-agent")
from agent.embed import ErnieEmbedder
from sklearn.model_selection import StratifiedShuffleSplit

CSV="/home/akumar/Dataset/ISOT.csv"
stored=np.load("/home/akumar/embeddings/ERNIE_Precomputed/isot_embeddings.npy")

df=pd.read_csv(CSV); df.columns=[c.lower().strip() for c in df.columns]
tcol=next(c for c in df.columns if c in ("title","text","content","body","news","statement"))
lcol=next(c for c in df.columns if c in ("verdict","label","labels","class","target","fake"))
texts=df[tcol].fillna("").str.lower().str.strip().tolist()
y=df[lcol].astype(int).values
N=len(texts); print(f"N={N} stored={stored.shape}")

emb=ErnieEmbedder()

def check(name, order):
    # embed the CSV rows that 'order' says occupy stored[0], stored[1], stored[5]
    probe=[0,1,5]
    live=emb.embed([texts[order[p]] for p in probe])
    for k,p in enumerate(probe):
        c=np.dot(live[k],stored[p])/(np.linalg.norm(live[k])*np.linalg.norm(stored[p]))
        print(f"  {name}: stored[{p}] vs csv[{order[p]}]  cos={c:.4f}")

# hypothesis A: identity (raw CSV order)
check("identity", np.arange(N))

# hypothesis B: full stratified 70/15/15 concatenated tr+va+te (SEED=42)
tr,rest=next(StratifiedShuffleSplit(1,test_size=0.30,random_state=42).split(np.arange(N).reshape(-1,1),y))
va,te=next(StratifiedShuffleSplit(1,test_size=0.50,random_state=42).split(rest.reshape(-1,1),y[rest]))
order_split=np.concatenate([tr,rest[va],rest[te]])
check("tr+va+te", order_split)

# hypothesis C: just the 'all' split = identity already covered; try test-only
check("test-first", np.concatenate([rest[te],tr,rest[va]]))
