from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.utils.strings import stringify_value


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def build_retriever(query: str) -> RunnableLambda:
    '''Builds a retriever for the given query.'''
    
    def retrieve(result: dict) -> str:
        docs = text_splitter.create_documents(texts=[result['content']])
        
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(docs)
        
        return {
            'url': result['url'],
            'content': stringify_value(vector_store.similarity_search(query))
        }
    
    return RunnableLambda(func=retrieve)


def retrieve_(query: str, search_results: list[dict]) -> list[dict]:
    return (
        build_retriever(query).batch(search_results)
    )
