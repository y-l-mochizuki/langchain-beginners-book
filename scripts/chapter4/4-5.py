# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "ユーザーが入力した料理のレシピを考えてください。"),
#     ("human", "{dish}")
# ])

# model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# chain = prompt | model

# ai_message = chain.invoke({"dish": "カレー"})
# print(ai_message.content)
# ai_message = Messageオブジェクト
"""
カレーのレシピをご紹介します。シンプルで美味しい基本のカレーを作りましょう。

### 材料（4人分）
- 鶏肉（もも肉または胸肉）: 400g
- 玉ねぎ: 2個
- にんじん: 1本
- じゃがいも: 2個
- カレールー: 1箱（約200g）
- サラダ油: 大さじ2
- 水: 800ml
- 塩: 適量
- 胡椒: 適量
- お好みでガーリックパウダーや生姜: 適量

### 作り方
1. **材料の下ごしらえ**:
   - 鶏肉は一口大に切り、塩と胡椒を振っておきます。
   - 玉ねぎは薄切り、にんじんは輪切り、じゃがいもは一口大に切ります。

2. **炒める**:
   - 大きめの鍋にサラダ油を熱し、玉ねぎを中火で炒めます。玉ねぎが透明になるまで炒めます。
   - 鶏肉を加え、表面が白くなるまで炒めます。

3. **野菜を加える**:
   - にんじんとじゃがいもを鍋に加え、全体をよく混ぜます。

4. **煮る**:
   - 水を加え、強火で煮立たせます。煮立ったら、アクを取り除き、中火にして蓋をし、約15分煮ます。

5. **カレールーを加える**:
   - カレールーを割り入れ、よく溶かします。さらに10分ほど煮込み、全体がなじんだら火を止めます。

6. **味を調える**:
   - お好みで塩や胡椒で味を調整します。

7. **盛り付け**:
   - ご飯と一緒に盛り付けて、お好みで福神漬けやらっきょうを添えて完成です。

### おすすめのトッピング
- チーズ
- 生卵（温泉卵や目玉焼き）
- 青ねぎやパセリの刻んだもの

この基本のカレーは、具材を変えたり、スパイスを追加したりすることでアレンジが可能です。お好みのスタイルで楽しんでください！
"""



# chain = prompt | model | StrOutputParser()
# output = chain.invoke({"dish": "カレー"})
# print(output)
"""
カレーのレシピをご紹介します。シンプルで美味しい基本のカレーを作りましょう。

### 材料（4人分）
- 鶏肉（もも肉または胸肉）: 400g
- 玉ねぎ: 2個
- にんじん: 1本
- じゃがいも: 2個
- カレールー: 1箱（約200g）
- サラダ油: 大さじ2
- 水: 800ml
- 塩: 適量
- 胡椒: 適量
- お好みでガーリックパウダーや生姜: 適量

### 作り方
1. **材料の下ごしらえ**:
   - 鶏肉は一口大に切り、塩と胡椒を振っておきます。
   - 玉ねぎは薄切り、にんじんは輪切り、じゃがいもは一口大に切ります。

2. **炒める**:
   - 大きめの鍋にサラダ油を熱し、玉ねぎを中火で炒めます。玉ねぎが透明になるまで炒めます。
   - 鶏肉を加え、表面が白くなるまで炒めます。

3. **野菜を加える**:
   - にんじんとじゃがいもを鍋に加え、全体をよく混ぜます。

4. **煮る**:
   - 水を加え、強火で煮立たせます。煮立ったら、アクを取り除き、中火にして蓋をし、約15分煮ます。

5. **カレールーを加える**:
   - カレールーを割り入れ、よく溶かします。さらに10分ほど煮込み、全体がなじんだら味を見て、必要に応じて塩や胡椒で調整します。

6. **仕上げ**:
   - お好みでガーリックパウダーや生姜を加えて風味をアップさせます。火を止めて、少し冷ますと味がなじみます。

7. **盛り付け**:
   - ご飯と一緒に盛り付けて、お好みで福神漬けやらっきょうを添えて完成です。

### おすすめのトッピング
- 煮卵
- チーズ
- パクチー

この基本のカレーはアレンジがしやすいので、野菜や肉を変えて楽しんでください！
"""


from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")

output_parser = PydanticOutputParser(pydantic_object=Recipe)



from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーが入力した料理のレシピを考えてください。\n\n{format_instructions}"),
    ("human", "{dish}")
])

prompt_with_format_instructions = prompt.partial(
    format_instructions=output_parser.get_format_instructions()
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(
    response_format={"type": "json_object"}
)




chain = prompt_with_format_instructions | model | output_parser

recipe = chain.invoke({"dish": "カレー"})
print(type(recipe)) # <class '__main__.Recipe'>
print(recipe)
"""
ingredients=['鶏肉 500g', '玉ねぎ 2個', 'にんじん 1本', 'じゃがいも 2個', 'カレールー 1箱', '水 800ml', 'サラダ油 大さじ2', '塩 適量', 'こしょう 適量'] steps=['鶏肉は一口大に切り、塩とこしょうをふる。', '玉ねぎは薄切り、にんじんとじゃがいもは一口大に切る。', '鍋にサラダ油を熱し、玉ねぎを炒めて透明になるまで炒める。', '鶏肉を加え、表面が白くなるまで炒める。', 'にんじんとじゃがいもを加え、全体を混ぜる。', '水を加え、沸騰したらアクを取り、弱火で20分煮る。', 'カレールーを加え、よく溶かしてさらに10分煮る。', '味を見て、必要に応じて塩で調整する。', 'ご飯と一緒に盛り付けて完成。']
"""



# PydanticOutputParserよりもwith_structured_outputを使った方が良い?
