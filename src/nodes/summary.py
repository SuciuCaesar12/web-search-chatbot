from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.utils.strings import stringify_value

from prompts import PROMPT_TEMPLATES, SYSTEM_PROMPTS


prompt = (
    ChatPromptTemplate([
        ('system', SYSTEM_PROMPTS['search-agent']),
        ('user', PROMPT_TEMPLATES['search-results-summarization']),
    ])
)

llm = ChatOpenAI(model='gpt-4o-mini-2024-07-18', verbose=False)

summarizer = prompt | llm | (lambda x: x.content)


def summary_(query: str, search_results: list[dict]) -> str:    
    return summarizer.invoke({
        'search_query': query, 
        'search_results': stringify_value(search_results)
    })
