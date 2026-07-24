"""
Live ERNIE embedding — byte-faithful to TRVFL/01_generate_embeddings.py.
CLS = last_hidden_state[:,0,:], lowercase+strip, max_len=512, RAW (no z-score).
GBAssigner applies mu/sigma downstream, so this must return raw CLS.
"""
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "/home/akumar/local_models/ernie-base-en"
MAX_LEN = 128  # matches gen_embeddings_v2.py


class ErnieEmbedder:
    def __init__(self, model_path=MODEL_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()

    @torch.no_grad()
    def embed(self, texts):
        """texts: str or list[str] -> np.ndarray (N, 768) raw CLS."""
        if isinstance(texts, str):
            texts = [texts]
        texts = [t.lower().strip() for t in texts]           # match load_csv
        enc = self.tok(texts, truncation=True, padding="max_length",
                       max_length=MAX_LEN, return_tensors="pt").to(self.device)
        cls = self.model(**enc).last_hidden_state[:, 0, :]   # match generate_embeddings
        return cls.cpu().float().numpy()
