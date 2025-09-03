import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `LangChain`, `OpenAI`, or `Other`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
    )
    | ChatOpenAI(model_name="gpt-4o-mini")
    | StrOutputParser()
)

# chain.invoke({"question": "how do I call OpenAI?"})



langchain_chain = PromptTemplate.from_template(
    """You are an expert in langchain. \
Always answer questions starting with "As Harrison Chase told me". \
Respond to the following question:

Question: {question}
Answer:"""
) | ChatOpenAI(model_name="gpt-4o-mini")
openai_chain = PromptTemplate.from_template(
    """You are an expert in openai. \
Always answer questions starting with "As Sam Altman told me". \
Respond to the following question:

Question: {question}
Answer:"""
) | ChatOpenAI(model_name="gpt-4o-mini")
general_chain = PromptTemplate.from_template(
    """Respond to the following question:

Question: {question}
Answer:"""
) | ChatOpenAI(model_name="gpt-4o-mini")


def route(info):
    if "openai" in info["topic"].lower():
        return openai_chain
    elif "langchain" in info["topic"].lower():
        return langchain_chain
    else:
        return general_chain

full_chain = {"topic": chain, "question": lambda x: x["question"]} | RunnableLambda(
    route
)

full_chain.invoke({"question": "how do I use langchain?"})