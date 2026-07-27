"""GB-Agent: GBFFN + LLM hybrid fake-news verification. Streamlit UI."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st

st.set_page_config(page_title="GB-Agent · Fake News Verifier", page_icon="🔍", layout="centered")

@st.cache_resource
def load_agent():
    from agent.graph import build_graph
    return build_graph()

st.title("🔍 GB-Agent")
st.caption("Granular-Ball FFN + LLM hybrid for fake-news verification")

with st.expander("How it works"):
    st.markdown(
        "1. **ERNIE** embeds your claim.\n"
        "2. **Granular-Ball FFN** finds the nearest semantic prototype and routes:\n"
        "   - *Confident* (near a pure prototype) → GBFFN answers directly (fast).\n"
        "   - *Uncertain* → escalate to web search + LLM.\n"
        "3. **DuckDuckGo** retrieves live evidence.\n"
        "4. **Gemini** delivers an evidence-grounded verdict.")

claim = st.text_area("Enter a claim to verify:",
                     placeholder="e.g. The president signed an infrastructure bill into law on Tuesday.",
                     height=100)

if st.button("Verify", type="primary", disabled=not claim.strip()):
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not set. Add it in Space settings → Secrets.")
        st.stop()
    with st.spinner("Embedding → GB routing → evidence → verdict…"):
        app = load_agent()
        out = app.invoke({"claim": claim.strip()})
    v, gb = out["verdict"], out["gb"]

    verdict = v.get("verdict", "UNVERIFIABLE")
    color = {"TRUE": "green", "FALSE": "red", "UNVERIFIABLE": "orange"}.get(verdict, "gray")
    st.markdown(f"### Verdict: :{color}[{verdict}]")
    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{v.get('confidence', 0):.0%}")
    c2.metric("Path", "GBFFN fast" if out["route"] == "fast_path" else "LLM escalated")
    st.write(f"**Rationale:** {v.get('rationale','—')}")

    with st.expander("GBFFN routing detail"):
        st.json({"nearest_ball": gb["ball_id"], "ball_purity": gb["ball_purity"],
                 "distance": gb["distance"], "escalated": gb["escalate"],
                 "reason": gb.get("reason")})

    if out.get("evidence"):
        st.subheader("Evidence")
        for i, e in enumerate(out["evidence"]):
            st.markdown(f"**[{i}]** {e['text'][:300]}")
            if e.get("url"): st.caption(e["url"])

st.divider()
st.caption("GBFFN model: Gurjwar et al., Applied Soft Computing 2026 · Demo uses live web + Gemini")
