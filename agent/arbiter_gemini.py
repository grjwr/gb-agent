"""Gemini arbiter with forced JSON output. claim + evidence -> verdict dict."""
import os, json, re
import google.generativeai as genai

INSTR = (
    "You are a fake-news verification assistant. Given a CLAIM and EVIDENCE passages, "
    "decide whether the claim is TRUE, FALSE, or UNVERIFIABLE based ONLY on the evidence. "
    "Return a JSON object with keys: verdict (TRUE|FALSE|UNVERIFIABLE), "
    "confidence (0.0-1.0), rationale (one sentence), evidence_ids (list of integers). "
    "Use UNVERIFIABLE if evidence is insufficient."
)

class GeminiArbiter:
    def __init__(self, model="models/gemini-3.5-flash", api_key=None):
        genai.configure(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(
            model, system_instruction=INSTR,
            generation_config={"temperature": 0.0, "max_output_tokens": 2048,
                               "response_mime_type": "application/json"})

    def verdict(self, claim, evidence):
        ev = "\n".join(f"[{i}] {e['text']}" for i, e in enumerate(evidence)) or "(no evidence)"
        try:
            raw = self.model.generate_content(f"CLAIM: {claim}\n\nEVIDENCE:\n{ev}").text
        except Exception as e:
            return {"verdict": "UNVERIFIABLE", "confidence": 0.0,
                    "rationale": f"api_error: {type(e).__name__}", "evidence_ids": []}
        try:
            d = json.loads(raw)
            d.setdefault("evidence_ids", [])
            return d
        except (json.JSONDecodeError, TypeError):
            m = re.search(r"\{.*\}", raw or "", re.DOTALL)
            if m:
                try: return json.loads(m.group())
                except json.JSONDecodeError: pass
            return {"verdict": "UNVERIFIABLE", "confidence": 0.0,
                    "rationale": "parse_failed", "raw": (raw or "")[:200]}
