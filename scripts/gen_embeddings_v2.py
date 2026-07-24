"""
Clean ERNIE embedding generation for GB-Agent v2.
Recipe: CLS = last_hidden_state[:,0,:], lowercase+strip, max_len=128, RAW output.
Same recipe the live node (agent/embed.py) uses -> guaranteed alignment.
Writes flat (N,768) .npy + (N,) labels into GB_AGENT_v2/.
"""
import os, sys, argparse
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel

MODEL = "/home/akumar/local_models/ernie-base-en"
OUT   = "/home/akumar/embeddings/GB_AGENT_v2"
MAX_LEN = 128
BATCH = 64

CSV = {
    "isot":       "/home/akumar/Dataset/ISOT.csv",
    "welfake":    "/home/akumar/Dataset/WELFake.csv",
    "gossipcop":  "/home/akumar/Dataset/GossipCop.csv",
    "politifact": "/home/akumar/Dataset/Politifact.csv",
    "buzzfeed":   "/home/akumar/Dataset/BuzzFeed.csv",
    "liar2":      "/home/akumar/Dataset/LIAR2.csv",
}

def load(path):
    df = pd.read_csv(path); df.columns=[c.lower().strip() for c in df.columns]
    tcol=next(c for c in df.columns if c in ("title","text","content","body","news","statement"))
    lcol=next(c for c in df.columns if c in ("verdict","label","labels","class","target","fake"))
    txt=df[tcol].fillna("").astype(str).str.lower().str.strip().tolist()
    y=df[lcol].astype(int).values
    print(f"  loaded {len(txt)} rows  text='{tcol}' label='{lcol}'  balance={np.bincount(y)}")
    return txt, y

def main(ds):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModel.from_pretrained(MODEL).to(dev).eval()
    txt,y=load(CSV[ds])
    embs=[]
    with torch.no_grad():
        for i in range(0,len(txt),BATCH):
            enc=tok(txt[i:i+BATCH],truncation=True,padding="max_length",
                    max_length=MAX_LEN,return_tensors="pt").to(dev)
            cls=model(**enc).last_hidden_state[:,0,:]
            embs.append(cls.cpu().float().numpy())
            if i % (BATCH*50)==0: print(f"    {i}/{len(txt)}",flush=True)
    E=np.concatenate(embs,0)
    os.makedirs(OUT,exist_ok=True)
    np.save(f"{OUT}/{ds}_embeddings.npy",E)
    np.save(f"{OUT}/{ds}_labels.npy",y.astype(int))
    print(f"  saved {ds}: {E.shape} -> {OUT}/{ds}_embeddings.npy")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); a=p.parse_args()
    main(a.dataset)
