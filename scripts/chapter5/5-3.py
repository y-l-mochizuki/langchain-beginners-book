import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

import pprint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from operator import itemgetter

model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
output_parser = StrOutputParser()

optimistic_prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは楽観主義者です。ユーザーの入力に対して楽観的な意見をください。"),
    ("human", "{topic}")
])

pessimistic_prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは悲観主義者です。ユーザーの入力に対して悲観的な意見をください。"),
    ("human", "{topic}")
])

synthesize_prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは客観的AIです。{topic}をまとめてください"),
    ("human", "楽観的意見{optimistic_option}、悲観的意見{pessimistic_option}")
])

optimistic_chain = optimistic_prompt | model | output_parser
pessimistic_chain = pessimistic_prompt | model | output_parser

# parallel_chain = RunnableParallel({
#     "optimistic_option": optimistic_chain,
#     "pessimistic_option": pessimistic_chain
# })

# これでも可
parallel_chain = {
    "optimistic_option": optimistic_chain,
    "pessimistic_option": pessimistic_chain,
    "topic": itemgetter('topic')
}

# output = parallel_chain.invoke({"input": "生成AIの進化についてどう思いますか？"})

chain = parallel_chain | synthesize_prompt | model | output_parser
output = chain.invoke({"topic": "生成AIの進化についてどう思いますか？"})
pprint.pprint(output)