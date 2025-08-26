# LLM
from langchain_openai import OpenAI
# model = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
model = OpenAI(model="gpt-4o-mini", temperature=0)
output = model.invoke("自己紹介してください。")
print(output)

# Chat Model
# LangChainにおける「SystemMessage」「HumanMessage」「AIMessage」はChatCompletionsAPIの「role: system」「role: user」「role: asistant」に相当する

# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from langchain_openai import ChatOpenAI

# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# messages = [
#     SystemMessage("You are a helpful assistant."), # role: system相当
#     HumanMessage("こんにちは！私はジョンといいます！"), # role: user相当
#     AIMessage(content="こんんちは、ジョンさん！どのようにお手伝いできます？"), # role: assistant相当
#     HumanMessage(content="私の名前がわかりますか？"), # role: user相当
# ]

# ai_message = model.invoke(messages)
# print(ai_message.content)



# streaming
# LangChainもストリーミングをサポートしてる

# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_openai import ChatOpenAI

# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# messages = [
#     SystemMessage("You are a helpful assistant."),
#     HumanMessage("こんにちは！"),
# ]

# for chunk in model.stream(messages):
#     print(chunk.content, end="", flush=True)
