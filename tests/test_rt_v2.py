import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, "/home/akumar/gb-agent")
from agent.embed import ErnieEmbedder

CSV="/home/akumar/Dataset/ISOT.csv"
STORED="/home/akumar/embeddings/GB_AGENT_v2/isot_embeddings.npy"
df=pd.read_csv(CSV); df.columns=[c.lower().strip() for c in df.columns]
tcol=next(c for c in df.columns if c in ("title","text","content","body","news","statement"))
texts=df[tcol].fillna("").astype(str).str.lower().str.strip().tolist()
stored=np.load(STORED)
print(f"rows={len(texts)} stored={stored.shape}")

emb=ErnieEmbedder()
probe=[0,1,100,5000,len(texts)-1]
live=emb.embed([texts[i] for i in probe])
for k,i in enumerate(probe):
    d=np.abs(live[k]-stored[i]).max()
    c=np.dot(live[k],stored[i])/(np.linalg.norm(live[k])*np.linalg.norm(stored[i]))
    print(f"row {i:6d}: max|Δ|={d:.6f}  cos={c:.6f}")
