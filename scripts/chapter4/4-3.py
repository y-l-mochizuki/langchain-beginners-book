# Prompt template

# from langchain_core.prompts import PromptTemplate

# prompt = PromptTemplate.from_template("""以下の料理のレシピを考えてください。
# 料理名: {dish}""")

# prompt_value = prompt.invoke({"dish": "カレー"})
# print(prompt_value.text)



# ChatPromptTemplate
# PromptTemplateをChatCompletionsAPIなどのチャット形式に対応したもの

# from langchain_core.prompts import ChatPromptTemplate

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "ユーザーが入力した料理のレシピを考えてください。"),
#     ("human", "{dish}") # human = role: user
# ])

# prompt_value = prompt.invoke({"dish": "カレー"})
# print(prompt_value)
# 以下が返る
# messages=[SystemMessage(content='ユーザーが入力した料理のレシピを考えてください。', additional_kwargs={}, response_metadata={}), HumanMessage(content='カレー', additional_kwargs={}, response_metadata={})]



# MessagesPlaceholder
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt = ChatPromptTemplate.from_messages([
#     ("system","You are a helpful assistant."),
#     MessagesPlaceholder("chat_history", optional=True),
#     ("human", "{input}")
# ])

# prompt_value = prompt.invoke({
#     "chat_history": [
#         HumanMessage(content="こんにちは！私はジョンといいます！"),
#         AIMessage(content="こんにちは、ジョンさん！どのようにお手伝いしますか？")
#     ],
#     "input": "私の名前がわかりますか？"
# })

# print(prompt_value)
"""
messages=[SystemMessage(content='You are a helpful assistant.', additional_kwargs={}, response_metadata={}), HumanMessage(content='こんにちは！私はジョンといいます！', additional_kwargs={}, response_metadata={}), AIMessage(content='こんにちは、ジョンさん！どのようにお手伝いしますか？', additional_kwargs={}, response_metadata={}), HumanMessage(content='私の名前がわかりますか？', additional_kwargs={}, response_metadata={})]
"""

"""
 invoke は「呼び出す/起動する」という意味で、LangChainでは：
  - プロンプトの場合：変数を埋めて完成させる
  - モデルの場合：APIを呼んで推論を実行する
  - チェーンの場合：一連の処理を実行する
"""


# langSmithのPrompts
# プロンプトの共有やバージョン管理ができる
# from langsmith import Client

# client = Client()
# prompt = client.pull_prompt("oshima/recipe")
# prompt_value  = prompt.invoke({"dish": "カレー"})
# print(prompt_value)

