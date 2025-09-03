import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_parser = StrOutputParser()

cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーの質問にステップバイステップで回答してください。"),
    ("human", "{question}")
])

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "ステップバイステップで考えた回答から結論を抽出してください"),
    ("human", "{text}")
])

cot_chain = cot_prompt | model | output_parser
summarize_chain = summarize_prompt | model | output_parser

cot_summarize_chain = cot_chain | summarize_chain
for chunk in cot_summarize_chain.stream({'question': "10 + 2 * 3"}):
    print(chunk, end="", flush=True)
    