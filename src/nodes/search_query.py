from pydantic import BaseModel, Field
from typing import Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from prompts import *


class SearchQuery(BaseModel):
    '''Structure for the search query'''
    search_query: str = Field(..., description="The generated search query")

prompt = (
    ChatPromptTemplate([
        ('system', SYSTEM_PROMPTS['search-agent']),
        ('user', PROMPT_TEMPLATES['search-query-generation']),
    ]) 
)

llm = ChatOpenAI(model='gpt-4o-mini-2024-07-18', verbose=False).with_structured_output(SearchQuery)

generator = prompt | llm | (lambda sq: sq.model_dump()['search_query'])

def generate_search_query_(
    question: str, 
    searched_queries: list[str], 
    searched_summaries: list[str], 
) -> str:
    """Generates a refined search query based on previous searches."""
    
    if searched_queries == []:
        return question
    else:
        return generator.invoke({
            'query': question, 
            'searched_queries': searched_queries, 
            'searched_summaries': searched_summaries
        })
