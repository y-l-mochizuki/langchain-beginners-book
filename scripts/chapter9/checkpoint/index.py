# Python標準ライブラリのインポート
import operator  # operator.addを使用してリストの結合操作を定義するため
import os  # 環境変数を扱うため
from pprint import pprint  # チェックポイントのデータを整形して表示するため
from typing import Annotated, Any, TypedDict  # 型ヒントを定義するため

# .envファイルから環境変数を読み込む
from dotenv import load_dotenv

# .envファイルから環境変数を読み込み
load_dotenv()
# OpenAI APIキーを環境変数から取得
openai_api_key = os.getenv("OPENAI_API_KEY")

# LangChain関連のインポート
from langchain_core.messages import HumanMessage  # チャット用のメッセージ型
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig  # LangChainの実行設定用
from langchain_openai import ChatOpenAI  # OpenAI APIと通信するためのLLMクラス
# LangGraph関連のインポート（ステートフルなワークフロー管理用）
from langgraph.checkpoint.base import BaseCheckpointSaver  # チェックポイントの基底クラス
from langgraph.checkpoint.memory import MemorySaver  # メモリベースのチェックポイント保存
from langgraph.graph import END, StateGraph  # グラフベースのワークフロー定義

# Pydanticのインポート（データバリデーション用）
# Pydanticは不要となったため削除


# LangGraphワークフローの状態を定義するクラス
# 「State（ステート）」= アプリケーションの現在の状態・データを管理するクラス
# ワークフロー実行中に各ノード間で共有・更新されるデータを保持する
# TypedDictを使用することで、LangGraphとの互換性を確保
class State(TypedDict):
    # ユーザーからの入力クエリ（質問・指示）を格納
    # 「query（クエリ）」= ユーザーが入力したテキスト（質問や指示）
    query: str

    # チャットの履歴メッセージを格納するリスト
    # 「messages」= 会話の履歴（システム、ユーザー、AI全ての発言を順番に保存）
    #
    # 「Annotated」= 型に追加情報（ここではリスト結合方法）を付与するPythonの型ヒント機能
    # 「list[BaseMessage]」= BaseMessage型のオブジェクトを格納するリスト
    # 「operator.add」= リストを結合する演算子（[1,2] + [3,4] = [1,2,3,4]）
    #
    # この組み合わせにより、複数のノード（関数）が返すメッセージリストが
    # 自動的に既存のmessagesリストに追加される仕組みが実現される
    # 例: 現在[msg1, msg2] → ノードが[msg3]を返す → 自動で[msg1, msg2, msg3]になる
    messages: Annotated[list[BaseMessage], operator.add]


# メッセージを追加するノード関数
# 「ノード（Node）」= グラフ構造の中の1つの処理単位（関数）
# LangGraphでは各ノードが順番に実行され、データ（State）を処理していく
# このノードはシステムメッセージとユーザーメッセージを状態に追加する役割
def add_message(state: dict) -> dict[str, Any]:
    # 新しく追加するメッセージのリストを初期化
    # 「BaseMessage」= LangChainで使用される全メッセージ型の親クラス
    # SystemMessage, HumanMessage, AIMessageなど全ての種類のメッセージを格納可能
    additional_messages: list[BaseMessage] = []  # 型エラー対策：BaseMessageを明示的に指定

    # 初回の会話の場合（メッセージ履歴が空の場合）、
    # 「SystemMessage」= AIの性格や振る舞いを指定するメッセージ（プロンプトエンジニアリング）
    # LLM（大規模言語モデル）に「どのように振る舞うべきか」を指示する
    if not state.get("messages", []):
        additional_messages.append(SystemMessage(content="あなたは最小限の応答をする対話エージェントです。"))

    # ユーザーのクエリをHumanMessageとして追加
    # 「HumanMessage」= 人間（ユーザー）からの入力メッセージ
    # LLMはこのメッセージに対して応答を生成する
    additional_messages.append(HumanMessage(content=state["query"]))

    # 状態更新用の辞書を返す（messagesフィールドが更新される）
    return {"messages": additional_messages}


# LLMから応答を生成するノード関数
# 「LLM（Large Language Model）」= 大規模言語モデル（ChatGPTなどのAI）
# 現在のメッセージ履歴を基にAIの応答を生成する処理ノード
def llm_response(state: dict) -> dict[str, Any]:
    # OpenAI APIを使用するLLMインスタンスを作成
    # 「ChatOpenAI」= OpenAIのAPIと通信してChatGPTを使用するためのクラス
    #
    # パラメータ説明：
    # 「model」: 使用するGPTモデル（gpt-4o-miniは高速・低コストな軽量版）
    # 「temperature」: 応答のランダム性を制御（0=決定的で同じ回答、1=創造的で多様な回答）
    #                0.5は中程度でバランスの取れた設定
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    # メッセージ履歴全体をLLMに送信して応答を生成
    # 「invoke」= LLMを実行（呼び出し）するメソッド
    # 全ての会話履歴を送信することで、文脈を理解した応答が可能になる
    ai_message = llm.invoke(state["messages"])

    # AI応答をリストに入れて返す
    # 「AIMessage」型のオブジェクトがリストとして返され、
    # operator.addの設定により自動的に既存のmessagesリストに追加される
    return {"messages": [ai_message]}


# チェックポイントの内容を表示するヘルパー関数
# 「チェックポイント」= ワークフローの特定時点での状態を保存したもの（セーブポイント）
# 「ヘルパー関数」= メインの処理を補助する便利な関数
# デバッグや状態確認に使用
def print_checkpoint_dump(checkpointer: BaseCheckpointSaver, config: RunnableConfig):
    # 指定されたconfigに対応するチェックポイントを取得
    # 「config」= 実行設定（thread_idなどのセッション情報を含む）
    # 「checkpoint_tuple」= チェックポイントのデータとメタデータをまとめたタプル（組）
    checkpoint_tuple = checkpointer.get_tuple(config)

    # チェックポイントが存在しない場合の処理
    if checkpoint_tuple is None:
        print("No checkpoint found.")
        return

    # チェックポイントの内容を整形して表示
    print("=== Checkpoint Dump ===")
    pprint(checkpoint_tuple.checkpoint)  # 保存された状態データ（Stateの内容）
    print("\nメタデータ:")
    pprint(checkpoint_tuple.metadata)  # メタデータ（作成時刻、ノード名などの補足情報）


# LangGraphワークフローの構築
# 「ワークフロー」= 一連の処理の流れ（処理手順）
# 「StateGraph」= 状態（State）を持つグラフ構造でワークフローを表現するクラス
# 「グラフ構造」= ノード（処理）とエッジ（つながり）で構成される図形的な構造
#
# このグラフは各ノードを順番に実行し、Stateを更新しながら処理を進める
graph = StateGraph(State)  # Stateクラスを使用してグラフを初期化

# ノードの登録（各ノードは特定の処理を行う関数）
# 「add_node」= グラフにノード（処理単位）を追加するメソッド
# 第1引数：ノードの名前（識別子）、第2引数：実行する関数
graph.add_node("add_message", add_message)  # メッセージ追加ノード
graph.add_node("llm_response", llm_response)  # LLM応答生成ノード

# ワークフローのエントリーポイントを設定
# 「エントリーポイント」= プログラムの実行開始地点（スタート地点）
graph.set_entry_point("add_message")  # 最初に実行されるノード

# ノード間のエッジ（接続）を定義
# 「エッジ（Edge）」= グラフ内のノード間の接続・矢印（処理の流れを表す）
# 「END」= LangGraphの特別な定数で、ワークフローの終了を表す
graph.add_edge("add_message", "llm_response")  # add_message → llm_response
graph.add_edge("llm_response", END)  # llm_response → 終了

# チェックポイント機能の設定
# 「MemorySaver」= コンピュータのメモリ（RAM）に状態を一時保存するクラス
# メモリ保存は高速だが、プログラム終了時にデータが消える
# 本番環境では「PostgresSaver」などのデータベース保存を使用して永続化する
checkpointer = MemorySaver()

# グラフをコンパイルして実行可能な形式に変換
# 「コンパイル」= グラフ定義を実際に実行可能な形式に変換すること
# checkpointerを指定することで、各ノード実行後の状態が自動的に保存される
# これにより会話の履歴が保持され、中断・再開が可能になる
compiled_graph = graph.compile(checkpointer=checkpointer)

# 実行設定の定義
# 「config」= 実行時の設定情報を格納する辞書
# 「thread_id」= 会話のセッション（一連の会話）を識別するID
# 同じthread_idを使用すると、前回の会話履歴が引き継がれる
# 異なるthread_idにすると、新しい会話として開始される
config = {"configurable": {"thread_id": "example-1"}}

# ユーザーのクエリを辞書として作成
# TypedDictベースのStateは辞書として扱う
# ユーザーの入力テキストをqueryフィールドに設定
user_query = {"query": "私の好きなものは、ずんだ餅です。覚えておいてね。", "messages": []}
first_response = compiled_graph.invoke(user_query, config)
print("初回応答:", first_response)

# 保存されたチェックポイントの一覧を表示
# 「checkpointer.list()」= 指定したセッション（thread_id）の全チェックポイントを取得
# 各ノード実行後の状態が時系列で確認でき、デバッグや動作確認に役立つ
# 各チェックポイントには、その時点でのStateとメタデータが含まれる
# for checkpoint in checkpointer.list(config):
#     print(checkpoint)

# チェックポイントの内容を表示する（ヘルパー関数を使用）
# print_checkpoint_dump(checkpointer, config)

# 2回目のクエリ実行（同じthread_idなので前回の会話履歴が保持される）
user_query = {"query": "私の好物は何か覚えてる？"}
# config = {"configurable": {"thread_id": "example-2"}}  # 別のthread_idで新規会話になるか試す -> 好きなもの覚えてなかった
second_response = compiled_graph.invoke(user_query, config)
print("2回目応答:", second_response)

#  同じスクリプト実行内であれば、configのthread_idをキーとして状態がMemorySaver()に保存されます。
