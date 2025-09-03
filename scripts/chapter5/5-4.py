import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pprint

retriever = TavilySearchAPIRetriever(k=3)

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください
文脈: """
{context}
"""

質問: {question}
''')

model = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# chain = {"context": retriever, "question": RunnablePassthrough()} | RunnablePassthrough.assign(answer=prompt | model | StrOutputParser())
chain = {"context": retriever, "question": RunnablePassthrough()} | RunnablePassthrough.assign(answer=prompt | model | StrOutputParser())

output = chain.invoke("東京の天気は？")
pprint.pprint(output)