import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI



prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーが入力した料理のレシピを考えてください。"),
    ("human", "{dish}")
])


model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
output_parser = StrOutputParser()


# prompt_value = prompt.invoke({"dish": "カレーライス"})
# ai_message = model.invoke(prompt_value)
# output = output_parser.invoke(ai_message)

#上と同じ
# invoke
# chain = prompt | model | output_parser
# output = chain.invoke({"dish": 'カレー'})
# print(output)

# stream
# chain = prompt | model | output_parser
# for chunk in chain.stream({"dish": "カレー"}):
#     print(chunk, end="", flush=True)

# batch
chain = prompt | model | output_parser
outputs = chain.batch([{"dish": "カレー"}, {"dish": "うどん"}])
print(outputs)