from typing import Annotated, Dict
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from prompts import PROMPT_TEMPLATES, SYSTEM_PROMPTS

from nodes import *


class GraphState(TypedDict):
    """Represents the state of the graph."""
    question: str
    
    search_query: str
    search_results: list[dict]
    
    searched_queries: Annotated[list[str], add] = []
    searched_summaries: Annotated[list[str], add] = []
    
    attempts: int


class OutputState(TypedDict):
    searched_summaries: list[str]


classifier = (
    ChatPromptTemplate([
        ('system', SYSTEM_PROMPTS['search-agent']),
        ('user', PROMPT_TEMPLATES['searched-summaries-classification']),
    ]) | ChatOpenAI(model='gpt-4o-mini-2024-07-18', verbose=False)
)
    

def generate_search_query(state: GraphState) -> dict:
    search_query = generate_search_query_(
        question=state['question'], 
        searched_queries=state['searched_queries'], 
        searched_summaries=state['searched_summaries']
    )
    
    return {
        'search_query': search_query, 
        'searched_queries': [search_query], 
        'attempts': state['attempts'] + 1
    }


async def web_search_node(state: GraphState) -> dict:
    return {
        'search_results': await web_search_(query=state['search_query']), 
    }


def retrieve_node(state: GraphState) -> dict:
    return {
        'search_results': retrieve_(
            query=state['search_query'], 
            search_results=state['search_results']
        ), 
    }


def summary_node(state: GraphState) -> dict:
    return {
        'searched_summaries': [summary_(
            query=state['search_query'], 
            search_results=state['search_results']
        )],
    }


def check_if_further_search_is_needed(state: Dict) -> str:
    """Checks if further search is needed based on the current summaries."""
    if state['attempts'] > 5:
        return END
    
    response = classifier.invoke({
        'query': state['search_query'], 
        'searched_summaries': state['searched_summaries']
    }).content.strip().lower()
    
    return END if response == 'yes' else 'continue'


graph_builder = StateGraph(GraphState, output=OutputState)

graph_builder.add_node('generate_search_query', generate_search_query)
graph_builder.add_node('web_search', web_search_node)
graph_builder.add_node('retrieve', retrieve_node)
graph_builder.add_node('summary', summary_node)

graph_builder.add_edge(START, 'generate_search_query')
graph_builder.add_edge('generate_search_query', 'web_search')
graph_builder.add_edge('web_search', 'retrieve')
graph_builder.add_edge('retrieve', 'summary')

graph_builder.add_conditional_edges(
    'summary', 
    check_if_further_search_is_needed, 
    {
        END: END, 
        'continue': "generate_search_query"
    }
)

search_agent = graph_builder.compile()
