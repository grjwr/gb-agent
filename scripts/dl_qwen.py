import os, sys
os.environ["HF_HOME"]="/home/akumar/hf_cache"
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)  # disable, it hangs if half-installed
from huggingface_hub import snapshot_download
print("starting download...", flush=True)
p = snapshot_download("Qwen/Qwen3-30B-A3B-Instruct-2507",
                      max_workers=4,
                      tqdm_class=None)
print("DONE:", p, flush=True)
