import operator
from typing import Annotated, Any, Optional, cast

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# データ構造の定義


# ペルソナを表すデータモデル
class Persona(BaseModel):
    name: str = Field(..., description="ペルソナの名前")
    background: str = Field(..., description="ペルソナの持つ背景情報")


# ペルソナのリストを表すデータモデル
class Personas(BaseModel):
    personas: list[Persona] = Field(default_factory=list, description="ペルソナのリスト")


# インタビュー内容
class Interview(BaseModel):
    persona: Persona = Field(..., description="インタビュー対象のペルソナ")
    question: str = Field(..., description="インタビューでの質問")
    answer: str = Field(..., description="インタビューでの回答")


class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(default_factory=list, description="インタビューの結果一覧")


class EvaluationResult(BaseModel):
    reason: str = Field(..., description="判断の理由")
    is_sufficient: bool = Field(..., description="情報が十分かどうか")


class InterviewState(BaseModel):
    user_request: str = Field(..., description="ユーザーからのリクエスト")
    personas: Annotated[list[Persona], operator.add] = Field(default_factory=list,
                                                             description="生成されたペルソナのリスト")
    interviews: Annotated[list[Interview], operator.add] = Field(default_factory=list,
                                                                 description="実施されたインタビューリスト")
    requirements_doc: str = Field(default="", description="生成された要件定義")
    iteration: int = Field(default=0, description="ペルソナ生成とインタビューの反復回数")
    is_information_sufficient: bool = Field(default=False, description="情報が十分かどうか")


# 主要コンポーネントの実装


# ペルソナを生成するクラス
class PersonaGenerator:

    def __init__(self, llm: ChatOpenAI, k: int = 5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        # プロンプトテンプレートを定義
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "あなたはユーザーインタビュー用の多様なペルソナを作成する専門家です。",
            ),
            (
                "human",
                f"以下のユーザーリクエストに関するインタビュー用に、{self.k}人の多様なペルソナを生成してください。\n\n"
                "ユーザーリクエスト: {user_request}\n\n"
                "各ペルソナには名前と簡単な背景を含めてください。年齢、性別、職業、技術的専門知識において多様性を確保してください。",
            ),
        ])
        # ペルソナ生成のためのチェーンを作成
        chain = prompt | self.llm
        # ペルソナを生成
        result = chain.invoke({"user_request": user_request})

        if not isinstance(result, Personas):
            raise ValueError("LLMの出力がPersonas型ではありません")

        return result


# インタビューを実施するクラス
class InterviewConductor:

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        # ペルソナごとに質問を生成
        questions = self._generate_questions(user_request=user_request, personas=personas)
        # 各質問に対してペルソナとして回答を生成
        answers = self._generate_answers(personas=personas, questions=questions)
        # ペルソナ・質問・回答を組み合わせてインタビューオブジェクトを作成
        interviews = self._create_interviews(personas=personas,
                                             questions=questions,
                                             answers=answers)
        # インタビュー結果を返す
        return InterviewResult(interviews=interviews)

    def _generate_questions(self, user_request: str, personas: list[Persona]) -> list[str]:
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたはユーザー要件に基づいて適切な質問を生成する専門化です"),
            ("human", "以下のペルソナに関連するユーザーリクエストについて、一つの質問を生成してください。\n\n"
             "ユーザーリクエスト: {user_request}\n"
             "ペルソナ: {persona_name} - {persona_background}\n\n"
             "質問は具体的で、このペルソナの視点から重要な情報を引き出すように設計してください")
        ])

        # 質問生成のためのチェーン作成
        question_chain = question_prompt | self.llm | StrOutputParser()

        question_queries = [{
            "user_request": user_request,
            "persona_name": persona.name,
            "persona_background": persona.background
        } for persona in personas]

        # 1つのチェインに対して複数の入力をbatchで渡して並列的に実行する
        return question_chain.batch(question_queries)

    def _generate_answers(self, personas: list[Persona], questions: list[str]) -> list[str]:
        answer_prompt = ChatPromptTemplate.from_messages([(
            "system",
            "あなたは以下のペルソナとして回答しています: {persona_name} - {persona_background}",
        ), ("human", "質問: {question}")])

        # 回答生成のためのチェイン作成
        answer_chain = answer_prompt | self.llm | StrOutputParser()

        # 各ペルソナに対する回答クエリの生成
        answer_queries = [{
            "persona_name": persona.name,
            "persona_background": persona.background,
            "question": question
        } for persona, question in zip(personas, questions)]

        # 回答をバッチ処理で生成
        return answer_chain.batch(answer_queries)

    def _create_interviews(self, personas: list[Persona], questions: list[str],
                           answers: list[str]) -> list[Interview]:
        return [
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]


# インタビュー内容を評価するクラス
class InformationEvaluator:

    def __init__(self, llm: ChatOpenAI):
        # with_structured_output: LLMの出力を指定したPydanticモデル（ここでは EvaluationResult）の構造化データとして返す機能
        self.llm = llm.with_structured_output(EvaluationResult)

    def run(self, user_request: str, interviews: list[Interview]) -> EvaluationResult:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたは包括的な要件文書を作成するための情報の十分性を評価する専門家です。"),
            ("human", "以下のユーザーリクエストとインタビュー結果に基づいて、包括的な要件文書を作成するのに十分な情報が集まったかどうかを判断してください。\n\n"
             "ユーザーリクエスト: {user_request}\n"
             "インタビュー結果: {interview_results}")
        ])

        chain = prompt | self.llm

        result = chain.invoke({
            "user_request":
                user_request,
            "interview_results":
                "\n".join(f"ペルソナ: {i.persona.name} - {i.persona.background}\n"
                          f"質問: {i.question}\n回答: {i.answer}\n" for i in interviews)
        })

        if not isinstance(result, EvaluationResult):
            raise ValueError("LLMの出力がEvaluationResult型ではありません")

        return result


# 要件文書を生成するクラス
class RequirementsDocumentGenerator:

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたは収集した情報に基づいて要件文書を作成する専門家です"),
            (
                "human",
                "以下のユーザーリクエストと複数のペルソナからのインタビュー結果に基づいて、要件文書を作成してください\n\n"
                "ユーザーリクエスト: {user_request}\n"
                "インタビュー結果: {interview_results}\n"
                "要件文書には以下のセクションを含めてください\n"
                "1. プロジェクト概要\n"
                "2. 主要機能\n"
                "3. 非機能要件\n"
                "4. 制約条件\n"
                "5. ターゲットユーザー\n"
                "6. 優先順位\n"
                "7. リスクと軽減策\n\n"
                "出力は必ず日本語でお願いします。\n\n要件文書:",
            )
        ])

        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "user_request":
                user_request,
            "interview_results":
                "\n".join("\n".join(f"ペルソナ: {i.persona.name} - {i.persona.background}\n"
                                    f"質問: {i.question}\n回答: {i.answer}\n" for i in interviews))
        })


class DocumentationAgent:

    def __init__(self, llm: ChatOpenAI, k: Optional[int] = None):
        self.persona_generator = PersonaGenerator(llm=llm, k=k)  # 多様なペルソナを生成
        self.interview_conductor = InterviewConductor(llm=llm)  # ペルソナに対してインタビューを実施
        self.information_evaluator = InformationEvaluator(llm=llm)  # インタビュー結果が要件定義に十分か評価
        self.requirements_doc_generator = RequirementsDocumentGenerator(
            llm=llm)  # インタビュー結果から要件文書を生成

        self.graph = self._create_graph()

    def _create_graph(self) -> Runnable:
        # ステートグラフの初期化
        workflow = StateGraph(InterviewState)

        # ノードの追加
        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("evaluate_information", self._evaluate_information)
        workflow.add_node("generate_requirements", self._generate_requirements)

        # エントリーポイントの設定とエッジの追加
        workflow.set_entry_point("generate_personas")
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "evaluate_information")

        # 条件付きエッジの追加
        workflow.add_conditional_edges(
            "evaluate_information",
            lambda state: not state.is_information_sufficient and state.iteration < 5,
            {
                True: "generate_personas",
                False: "generate_requirements"
            },
        )

        workflow.add_edge("generate_requirements", END)  # ENDを追加することで明示的に終了を示す。ENDは必須
        return workflow.compile()

    def _generate_personas(self, state: InterviewState) -> dict[str, Any]:
        # ペルソナの作成
        new_personas: Personas = self.persona_generator.run(state.user_request)
        return {"personas": new_personas.personas, "iteration": state.iteration + 1}

    def _conduct_interviews(self, state: InterviewState) -> dict[str, Any]:
        new_interviews: InterviewResult = self.interview_conductor.run(
            state.user_request,
            state.personas[-5:]  # 最新の5つのペルソナでインタビュー
        )
        return {"interviews": new_interviews.interviews}

    def _evaluate_information(self, state: InterviewState) -> dict[str, Any]:
        evaluation_result: EvaluationResult = self.information_evaluator.run(
            state.user_request, state.interviews)

        return {
            "is_information_sufficient": evaluation_result.is_sufficient,
            "missing_information": evaluation_result.reason
        }

    def _generate_requirements(self, state: InterviewState) -> dict[str, Any]:
        requirements_doc: str = self.requirements_doc_generator.run(state.user_request,
                                                                    state.interviews)
        return {"requirements_doc": requirements_doc}

    def run(self, user_request: str) -> str:
        initial_state = InterviewState(user_request=user_request)
        final_state = self.graph.invoke(initial_state)
        return final_state['requirements_doc']


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ユーザー要求に基づいて要件定義を生成します")
    parser.add_argument("--task", type=str, help="作成したいアプリケーションについて記載してください")
    parser.add_argument("--k", type=str, default="5", help="生成するペルソナの数を設定してください（デフォルト: 5）")
    args = parser.parse_args()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    agent = DocumentationAgent(llm=llm, k=args.k)
    final_output = agent.run(user_request=args.task)

    # 最終的な出力を表示する
    print(final_output)


if __name__ == "__main__":
    main()

# python scripts/chapter10/main.py --task 'スマートフォン向けの健康管理アプリを開発したい' --k 5

# 出力結果
"""
# 要件文書

## 1. プロジェクト概要
本プロジェクトは、スマートフォン向けの健康管理アプリを開発することを目的としています。このアプリは、ユーザーが日常的に健康を管理し、フィットネスや栄養に関するデータを簡単に記録・分析できるようにすることを目指します。ユーザーの健康意識を高め、健康的なライフスタイルを促進するための機能を提供します。

## 2. 主要機能
以下の主要機能を実装します。

### 2.1 トラッキング機能
- 歩数、運動量、睡眠時間、栄養素の記録を自動または手動で行う機能。
- GPSを利用したランニングやサイクリングのルートトラッキング。

### 2.2 カスタマイズ可能なワークアウトプラン
- ユーザーの目標や体力に応じたトレーニングプランをカスタマイズできる機能。

### 2.3 栄養管理
- 食事の内容やカロリー、栄養素の記録・分析機能。
- マクロ栄養素のバランスを把握できる機能。

### 2.4 コミュニティ機能
- 他のユーザーとつながり、励まし合ったり情報を共有できる機能。

### 2.5 サポート機能
- 初心者向けの使い方ガイドやチュートリアル。
- FAQセクションやカスタマーサポート窓口の設置。

### 2.6 SNS連携機能
- ユーザーが進捗をSNSで簡単にシェアできる機能。

## 3. 非機能要件
- **ユーザーインターフェース**: シンプルで直感的なデザインを採用し、誰でも簡単に操作できるようにする。
- **パフォーマンス**: アプリの起動時間は3秒以内、データの記録・表示は1秒以内で行う。
- **セキュリティ**: ユーザーの個人情報や健康データを適切に保護するためのセキュリティ対策を実施する。

## 4. 制約条件
- 開発期間は6ヶ月以内とし、予算は500万円を上限とする。
- iOSおよびAndroidの両プラットフォームに対応する必要がある。

## 5. ターゲットユーザー
- 健康意識が高い30代から50代の男女。
- 忙しい日常を送る看護師や自営業の方々。
- 健康管理を楽しみたい大学生や若年層。

## 6. 優先順位
1. トラッキング機能
2. 栄養管理機能
3. カスタマイズ可能なワークアウトプラン
4. コミュニティ機能
5. サポート機能
6. SNS連携機能

## 7. リスクと軽減策
- **リスク**: ユーザーのプライバシーに関する懸念。
  - **軽減策**: データの暗号化やプライバシーポリシーの明確化を行う。
  
- **リスク**: 技術的な問題による開発遅延。
  - **軽減策**: 定期的な進捗確認と問題解決のためのミーティングを実施する。

- **リスク**: ユーザーのアプリ利用継続率が低い。
  - **軽減策**: ゲーミフィケーション要素を取り入れ、モチベーションを高める仕組みを導入する。

以上が、スマートフォン向け健康管理アプリの要件文書です。このアプリが多くのユーザーに支持され、健康的なライフスタイルの実現に寄与することを期待しています。
"""
