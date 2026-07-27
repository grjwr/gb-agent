"""
Qwen3-30B-A3B-Instruct arbiter. Inference-only. Takes claim + evidence,
returns structured verdict JSON. Runs on GPU node, reads model from shared HF cache.
"""
import os, json, re
os.environ.setdefault("HF_HOME", "/home/akumar/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never phone home from compute node
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

SYS = (
    "You are a fake-news verification assistant. Given a CLAIM and EVIDENCE passages, "
    "decide whether the claim is TRUE or FALSE based ONLY on the evidence. "
    "Respond with a single JSON object and nothing else, of the form: "
    '{"verdict": "TRUE|FALSE|UNVERIFIABLE", "confidence": 0.0-1.0, '
    '"rationale": "one sentence", "evidence_ids": [list of passage numbers you used]}. '
    "If the evidence is insufficient, use UNVERIFIABLE."
)

class Arbiter:
    def __init__(self, model=MODEL):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype="auto", device_map="auto").eval()

    def _prompt(self, claim, evidence):
        ev = "\n".join(f"[{i}] {e['text']}" for i, e in enumerate(evidence)) or "(no evidence retrieved)"
        return [{"role": "system", "content": SYS},
                {"role": "user", "content": f"CLAIM: {claim}\n\nEVIDENCE:\n{ev}"}]

    @torch.no_grad()
    def verdict(self, claim, evidence):
        msgs = self._prompt(claim, evidence)
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.tok(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=256, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            return json.loads(m.group()) if m else {"verdict": "UNVERIFIABLE",
                     "confidence": 0.0, "rationale": "parse_failed", "raw": raw[:200]}
        except json.JSONDecodeError:
            return {"verdict": "UNVERIFIABLE", "confidence": 0.0,
                    "rationale": "json_error", "raw": raw[:200]}
