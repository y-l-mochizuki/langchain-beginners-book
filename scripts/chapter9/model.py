from langchain_core.runnables import ConfigurableField, RunnableSerializable
from langchain_openai import ChatOpenAI

MODEL = "gpt-4o"

chat_open_ai = ChatOpenAI(model=MODEL, temperature=0)
llm = chat_open_ai.configurable_fields(max_tokens=ConfigurableField(id="max_tokens"))
