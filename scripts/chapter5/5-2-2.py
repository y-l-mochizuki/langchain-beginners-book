import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


prompt = ChatPromptTemplate.from_messages([
    ("system", 'あなたは有能なアシスタントです。'),
    ('human', "{input}")
])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_parser = StrOutputParser()

@chain
def upper(text: str) -> str:
    return text.upper()

chain = prompt | model | output_parser | upper
output = chain.invoke({"input": "hello!"})
print(output)