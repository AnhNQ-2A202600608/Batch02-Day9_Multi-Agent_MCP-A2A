"""Tax Agent LangGraph definition.

Uses create_react_agent with a tax-specialised system prompt.
No tools — it answers purely from LLM knowledge.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from common.llm import get_llm

TAX_SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn luật thuế. Hãy phân tích khía cạnh thuế của câu hỏi.
YÊU CẦU QUAN TRỌNG: Hãy trả lời cực kỳ ngắn gọn, trong tối đa 3 câu và viết bằng tiếng Việt.
"""


def create_graph():
    """Return a compiled LangGraph create_react_agent for tax questions."""
    llm = get_llm()
    graph = create_react_agent(
        model=llm,
        tools=[],
        prompt=TAX_SYSTEM_PROMPT,
    )
    return graph