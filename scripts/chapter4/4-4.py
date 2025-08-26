# Output Parser

from pydantic import BaseModel, Field

# class Recipe(BaseModel):
#     ingredients: list[str] = Field(description="ingredients of the dish")
#     steps: list[str] = Field(description="steps to make the dish")



# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI

# output_parser = PydanticOutputParser(pydantic_object=Recipe)
# format_instructions = output_parser.get_format_instructions()
# print(format_instructions)
"""
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"ingredients": {"description": "ingredients of the dish", "items": {"type": "string"}, "title": "Ingredients", "type": "array"}, "steps": {"description": "steps to make the dish", "items": {"type": "string"}, "title": "Steps", "type": "array"}}, "required": ["ingredients", "steps"]}
```
"""

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "ユーザーが入力した料理のレシピを教えてください。\n\n{format_instructions}"),
#     ("human", "{dish}")
# ])

# prompt_with_format_instructions = prompt.partial(
#     format_instructions=format_instructions
# )

# prompt_value = prompt_with_format_instructions.invoke({"dish": "カレー"})
# print("=== role: system ===")
# print(prompt_value.messages[0].content)
# print("=== role: user ===")
# print(prompt_value.messages[1].content)


# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ai_messages = model.invoke(prompt_value.messages)
# print(ai_messages.content)
"""
{
  "ingredients": [
    "鶏肉",
    "玉ねぎ",
    "にんじん",
    "じゃがいも",
    "カレールー",
    "水",
    "塩",
    "こしょう",
    "サラダ油"
  ],
  "steps": [
    "鶏肉を一口大に切り、塩とこしょうをふる。",
    "玉ねぎを薄切りにし、にんじんとじゃがいもを一口大に切る。",
    "鍋にサラダ油を熱し、玉ねぎを炒めて透明になるまで炒める。",
    "鶏肉を加え、表面が白くなるまで炒める。",
    "にんじんとじゃがいもを加え、全体をよく混ぜる。",
    "水を加え、煮立ったらアクを取り、弱火で20分煮る。",
    "カレールーを加え、溶かしながらさらに10分煮る。",
    "味を見て、必要に応じて塩で調整する。",
    "ご飯と一緒に盛り付けて完成。"
  ]
}
"""

# recipe = output_parser.parse(ai_messages.content)
# print(type(recipe))
# print(recipe)
"""
<class '__main__.Recipe'>
ingredients=['鶏肉', '玉ねぎ', 'にんじん', 'じゃがいも', 'カレールー', '水', '塩', 'こしょう', 'サラダ油'] steps=['鶏肉を一口大に切り、塩とこしょうをふる。', '玉ねぎを薄切りにし、にんじんとじゃがいもを一口大に切る。', '鍋にサラダ油を熱し、玉ねぎを炒めて透明になるまで炒める。', '鶏肉を加え、表面が白くなるまで炒める。', 'にんじんとじゃがいもを加え、全体を混ぜる。', '水を加え、煮立ったらアクを取る。', '弱火にして、約20分煮込む。', 'カレールーを加え、溶かしながらさらに10分煮込む。', '味を見て、必要に応じて塩で調整する。', 'ご飯と一緒に盛り付けて完成。']
"""


# StrOutputParser
# ただ文字列として扱う
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

ai_message = AIMessage(content="こんにちは。私はAIアシスタントです。")
output = output_parser.invoke(ai_message)
print(type(output)) # <class 'str'>
print(output) # こんにちは。私はAIアシスタントです。


