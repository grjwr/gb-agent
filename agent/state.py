"""Shared state for the GB-Agent LangGraph pipeline."""
from typing import TypedDict, Optional, List
from typing_extensions import Annotated
import operator

class AgentState(TypedDict, total=False):
    claim: str                          # raw user claim
    embedding: list                     # ERNIE [CLS] (768,)
    gb: dict                            # GBAssigner.assign() output
    route: str                          # "fast_path" | "escalate"
    evidence: Annotated[List[dict], operator.add]  # web + rag, fan-in
    verdict: dict                       # final {label, confidence, rationale, sources}
