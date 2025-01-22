from typing import Annotated

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.utils.strings import stringify_value
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from prompts import SYSTEM_PROMPTS
from .search import search_agent


@tool
async def web_search_tool(
    query: Annotated[str, '''The query to perform web searches for. It's best that the query is also expressed as a search query.''']
    ):
    '''Tool for performing multiple web searches for aiding in the response to the given query.'''
    
    response = await search_agent.ainvoke(
        input={
            "question": query,
            "attempts": 0
        },
        config={"configurable": {"thread_id": None}}  # stateless
    )
    
    return stringify_value(response['searched_summaries'])

tools = [web_search_tool]

chatbot = (
    ChatPromptTemplate(
        messages=[
            SystemMessage(content=SYSTEM_PROMPTS['conversational-agent']),
            MessagesPlaceholder("messages")
        ]
    ) | ChatOpenAI(
        name="chatbot",
        model='gpt-4o-mini-2024-07-18', 
        verbose=False
    ).bind_tools(tools)
)


def should_continue(state: MessagesState):
    return "tools" if state["messages"][-1].tool_calls else END


async def answer(state: MessagesState):
    return {"messages": [await chatbot.ainvoke(state["messages"])]}


workflow = StateGraph(MessagesState)

workflow.add_node("answer", answer)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "answer")
workflow.add_conditional_edges("answer", should_continue, ["tools", END])
workflow.add_edge("tools", "answer")

chat_agent = workflow.compile(checkpointer=MemorySaver())
