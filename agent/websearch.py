"""Free web-search evidence via DuckDuckGo. No API key. Returns [{"text","url"}]."""
from ddgs import DDGS

def web_search(claim, k=5):
    out = []
    try:
        with DDGS() as ddg:
            for r in ddg.text(claim, max_results=k):
                snippet = f"{r.get('title','')}. {r.get('body','')}".strip()
                if snippet:
                    out.append({"text": snippet, "url": r.get("href", "")})
    except Exception as e:
        return [{"text": f"(web search unavailable: {type(e).__name__})", "url": ""}]
    return out
