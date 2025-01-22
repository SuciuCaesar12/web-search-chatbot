import os
import sys
import asyncio
from typing import List
from pydantic import BaseModel, Field

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.utils.strings import stringify_value
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from prompts import PROMPT_TEMPLATES, SYSTEM_PROMPTS


__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "output")

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

max_concurrent = 5


async def crawl(urls: List[str]):
    out_results = []
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, verbose=False)
    
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            tasks = []

            for j, url in enumerate(batch):
                tasks.append(
                    crawler.arun(
                        url=url, 
                        config=crawl_config, 
                        session_id=f"parallel_session_{i + j}"
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for url, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"Error crawling {url}")
                elif result.success:
                    out_results.append({
                        'url': url,
                        'content': result.markdown_v2.raw_markdown
                    })
        
        return out_results

    finally:
        await crawler.close()


crawler = RunnableLambda(func=crawl)


class SearchApiResult(BaseModel):
    """Represents a search result."""
    title: str = Field(..., description="The title of the page.")
    snippet: str = Field(..., description="The snippet of the page.")
    url: str = Field(..., description="The URL of the page.")


class SearchApiResultList(BaseModel):
    """Represents a list of search results."""
    search_results: List[SearchApiResult] = Field(..., description="The list of search results to be scraped")


prompt = (
    ChatPromptTemplate([
        ('system', SYSTEM_PROMPTS['search-agent']),
        ('user', PROMPT_TEMPLATES['search-results-classification']),
    ]) 
)

llm = ChatOpenAI(model='gpt-4o-mini-2024-07-18', verbose=False).with_structured_output(SearchApiResultList)


def model_dump(x: SearchApiResultList) -> List[str]:
    return [y['url'] for y in x.model_dump()['search_results']]


classifier = prompt | llm | model_dump

search = DuckDuckGoSearchResults(output_format="list", num_results=5) | stringify_value

chain = (
    {'search_query': RunnablePassthrough(), 'search_results': search} 
    | classifier
    | crawler
)


async def web_search_(query: str):
    return await chain.ainvoke(query)
