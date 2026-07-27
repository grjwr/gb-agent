from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import embed_node, gb_node, route_selector, fast_path_node, escalate_node

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("embed", embed_node)
    g.add_node("gb", gb_node)
    g.add_node("fast_path", fast_path_node)
    g.add_node("escalate", escalate_node)
    g.set_entry_point("embed")
    g.add_edge("embed", "gb")
    g.add_conditional_edges("gb", route_selector,
                            {"fast_path": "fast_path", "escalate": "escalate"})
    g.add_edge("fast_path", END)
    g.add_edge("escalate", END)
    return g.compile()
