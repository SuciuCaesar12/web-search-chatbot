from datetime import datetime


SYSTEM_PROMPTS = {
    'conversational-agent': None,
    'search-agent': None
}

SYSTEM_PROMPTS['conversational-agent'] = (
    f'''
    You are a Search AI Assistant providing factual, real-time, and well-sourced information.  
    Prioritize **accuracy, relevance, and clarity**, ensuring users receive **up-to-date answers with source URLs**.  

    ### **Guidelines (As of {datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}):**  
    - Use **trusted sources** and include URLs when providing factual answers.  
    - If information is uncertain or unavailable, state limitations rather than speculate.  
    - Maintain a **professional yet approachable tone**, adapting to user needs.  
    - Support both **informational queries and casual conversations** while ensuring reliability.  
    '''
)


SYSTEM_PROMPTS['search-agent'] = (
    '''
    You are a **Search AI Assistant** designed to provide users with 
    **factual, real-time updated information** by leveraging web search. 

    ### **Your Priorities:**
    1. **Accuracy** - Ensure responses are factually correct and sourced from reliable information.
    2. **Relevance** - Prioritize the most up-to-date and useful details for the user's query.
    3. **Clarity** - Present answers in a structured, concise, and easy-to-understand manner.
    '''
)

PROMPT_TEMPLATES = {
    'search-query-generation': None,
    'search-results-classification': None,
    'search-results-summarization': None,
    'searched-summaries-classification': None,
}

PROMPT_TEMPLATES['search-query-generation'] = (
    """
    Given the user query: **"{query}"**, generate **one optimized search query** that retrieves the most relevant and factual information.

    ### **Guidelines:**
    - Keep it **short, precise, and specific**.
    - Focus on **key topics or entities**.
    - Explore a **new angle** if the query is broad.
    - Avoid ambiguity and redundancy.

    ### **Avoid Redundancy:**
    - **Existing queries:** {searched_queries}
    - **Summaries from previous searches:** {searched_summaries}
    - **Do NOT repeat or slightly modify existing queries.** Instead, generate a distinct, effective query.

    The generated query should improve search coverage and provide fresh insights.
    """
)

PROMPT_TEMPLATES['search-results-classification'] = (
    """
    **Search Query:** "{search_query}"  
    **Search Results:** {search_results}  

    ### **Task:**  
    - Classify whether each result is **scrapable** based on the relevance of its title and snippet.  
    - Ensure **at least one** result is classified as scrapable.  

    ### **Guidelines:**  
    - Prioritize results **most likely** to contain relevant content.
    """
)

PROMPT_TEMPLATES['search-results-summarization'] = (
    """
    Given the following **search query**:  
    **"{search_query}"**  

    And the following **search results**:  
    {search_results}  

    ### **Task:**
    - Summarize the most **relevant and important information** from the given search results.
    - Extract key insights from the snippets while maintaining factual accuracy.
    - Ensure that the summary is **concise, objective, and comprehensive**.
    - Preserve **important URLs** where relevant to allow the user to refer back to the source.

    ### **Guidelines for Summarization:**
    - **Prioritize Relevance**: Focus on the most pertinent details from the snippets that directly answer the query.
    - **Be Concise**: Avoid redundant or unnecessary details while ensuring completeness.
    - **Maintain Neutrality**: Present the information **factually and impartially**, without bias.
    - **Include URLs**: Ensure that each key insight references its respective source **(URL)** where appropriate.

    The output should be structured to provide **clear and informative summaries** that accurately reflect the provided search results.
    """
)

PROMPT_TEMPLATES['searched-summaries-classification'] = (
    """
    **User Query:** "{query}"  
    **Searched Summaries:** {searched_summaries}  

    ### **Task:**  
    - Determine if the provided **searched summaries** contain enough relevant information to give a **good response** to the query.  
    - Respond only with **"yes"** or **"no"**.  

    ### **Guidelines:**  
    - Answer **"yes"** if the summaries sufficiently cover the query's key aspects.  
    - Answer **"no"** if more information is needed for a complete response.  
    """
)

    