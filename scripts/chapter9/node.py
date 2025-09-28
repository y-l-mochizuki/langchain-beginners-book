from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from scripts.chapter9.model import llm
from scripts.chapter9.prompts import ANSWERING_PROMPT, CHECK_PROMPT, SELECTION_PROMPT
from scripts.chapter9.role import ROLES
from scripts.chapter9.state import State


def selection_node(state: State) -> dict[str, Any]:
    # 質問内容に基づいて最適な回答担当ロールを選択する
    query = state.query
    # 利用可能なロールのリストを作成（番号付き）
    role_options = "\n".join([f"{k}.{v['name']}: {v['name']}" for k, v in ROLES.items()])
    prompt = ChatPromptTemplate.from_template(SELECTION_PROMPT)

    # LLMにロール番号を選択させる（1トークンのみ生成）
    chain = prompt | llm.with_config(configurable=dict(max_tokens=1)) | StrOutputParser()
    role_number = chain.invoke({
        "role_options": role_options,
        "query": query,
    })
    # 選択されたロール番号から実際のロール名を取得
    selected_role = ROLES[role_number.strip()]['name']
    return {"current_role": selected_role}


def answering_node(state: State) -> dict[str, Any]:
    # 選択されたロールに基づいて質問に回答する
    query = state.query
    role = state.current_role
    # 全ロールの詳細情報を作成（参考用）
    role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in ROLES.values()])
    prompt = ChatPromptTemplate.from_template(ANSWERING_PROMPT)

    # 指定されたロールとして回答を生成
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "role": role,
        "role_details": role_details,
        "query": query,
    })

    return {
        "messages": [answer],
    }


class Judgement(BaseModel):
    # 回答品質の判定結果を格納するモデル
    reason: str = Field(default="", description="判定理由")
    judge: bool = Field(default=False, description="判定結果")


def check_node(state: State) -> dict[str, Any]:
    # 回答の品質をチェックし、問題の有無を判定する
    query = state.query
    answer = state.messages[-1]
    prompt = ChatPromptTemplate.from_template(CHECK_PROMPT)

    # 構造化出力を使って判定結果と理由を取得
    chain = prompt | llm.with_structured_output(Judgement)
    result: Judgement = chain.invoke({
        "query": query,
        "answer": answer,
    })

    return {
        "current_judge": result.judge,
        "judgement_reason": result.reason,
    }
