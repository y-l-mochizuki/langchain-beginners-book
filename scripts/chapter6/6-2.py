import os
from enum import Enum
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_community.retrievers import (BM25Retriever, TavilySearchAPIRetriever)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (Runnable, RunnableParallel, RunnablePassthrough)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field

from scripts.chapter6.utils import reciprocal_rank_fusion, rerank

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")


def file_filter(file_path: str) -> bool:
    return file_path.endswith('.mdx')


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

raw_documents = loader.load()
documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
db = Chroma.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈："""
{context}
"""

質問：{question}
''')

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
retriever = db.as_retriever()

# chain: Runnable = {
#     'question': RunnablePassthrough(),
#     'context': retriever,
# } | prompt | model | StrOutputParser()

# output = chain.invoke('LangChainの概要を教えて')

# HyDE
# hypothetical_prompt = ChatPromptTemplate.from_template("""\
# 次の質問に回答する一文を書いてください

# 質問：{question}
# """)

# hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

# hyde_rag_chain: Runnable = {
#     "question": RunnablePassthrough(),
#     "context": hypothetical_chain | retriever
# } | prompt | model | StrOutputParser()

# hyde_rag_chain.invoke("LangChainの概要を教えて")


# 複数の検索クエリ生成
class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description='検索クエリのリスト')


query_generation_prompt = ChatPromptTemplate.from_template("""\
質問に対してベスターデータベースから関連文書を検索するために3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、ユーザーの質問に対して複数の視点を提供することが目標です。

質問：{question}
""")

query_generation_chain = (query_generation_prompt |
                          model.with_structured_output(QueryGenerationOutput) |
                          (lambda x: x.queries))

# multi_query_rag_chain: Runnable = {
#     "question": RunnablePassthrough(),
#     "context": query_generation_chain | retriever.map(),
# } | prompt | model | StrOutputParser()

# multi_query_rag_chain.invoke("LangChainの概要を教えて")

# rag fusion
# rag_fusion_chain: Runnable = {
#     "question": RunnablePassthrough(),
#     "context": query_generation_chain | retriever.map() | reciprocal_rank_fusion,
# } | prompt | model | StrOutputParser()

# rag_fusion_chain.invoke("langchainの概要を教えて")

# rerank_rag_chain: Runnable = ({
#     "question": RunnablePassthrough(),
#     "documents": retriever
# } | RunnablePassthrough.assign(context=rerank)) | prompt | model | StrOutputParser()

# rerank_rag_chain.invoke("LangChainの概要を教えて")

# langchain_document_retriever = retriever.with_config({"run_name": "langchain_document_retriever"})

# web_retriever = TavilySearchAPIRetriever(k=3).with_config({"run_name": "web_retriever"})

# class Route(str, Enum):
#     langchain_document = "langchain_document"
#     web = "web"

# class RouteOutput(BaseModel):
#     route: Route

# route_prompt = ChatPromptTemplate.from_template(""""\
# 質問に回答するための適切なRetrieverを選択してください。
# 質問：{question}
# """)

# route_chain = route_prompt | model.with_structured_output(RouteOutput) | (lambda x: x.route)

# def routed_retriever(inp: dict[str, Any]) -> list[Document]:
#     question = inp["question"]
#     route = inp["route"]

#     if route == Route.langchain_document:
#         return langchain_document_retriever.invoke(question)
#     elif route == Route.web:
#         return web_retriever.invoke(question)

#     raise ValueError(f"Unknown route: {route}")

# route_rag_chain: Runnable = ({
#     "question": RunnablePassthrough(),
#     "route": route_chain,
# } | RunnablePassthrough.assign(context=routed_retriever) | prompt | model | StrOutputParser())

# route_rag_chain.invoke("LangChainの概要を教えて")
# route_rag_chain.invoke("東京の今日の天気は？")

chroma_retriever = retriever.with_config({"run_name": "chroma_retriever"})

bm25_retriever = BM25Retriever.from_documents(documents).with_config({
    "run_name": "bm25_retriever",
})

hybrid_retriever = (RunnableParallel({
    "chroma_documents": chroma_retriever,
    "bm25_documents": bm25_retriever,
}) | (lambda x: [x["chroma_documents"], x["bm25_documents"]]) | reciprocal_rank_fusion)

hybrid_rag_chain: Runnable = ({
    "question": RunnablePassthrough(),
    "context": hybrid_retriever,
} | prompt | model | StrOutputParser())

hybrid_rag_chain.invoke("LangChainの概要を教えて")
