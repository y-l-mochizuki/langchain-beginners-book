import asyncio
from typing import Any

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision
from ragas.metrics.base import Metric, MetricWithEmbeddings, MetricWithLLM
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator

LLM_MODEL = "gpt-4o-mini"
EMBEDDINGS_MODEL = "text-embedding-3-small"
DATA_SET_NAME = "agent-book"


# GitHubリポジトリから特定の拡張子のファイルのみを対象にするフィルタ関数
def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


# 検索対象のドキュメントをGitHubから取得
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",  #
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter)

raw_documents = loader.load()

# トークンデカすぎてエラーになるので分割する
splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=120, separator="\n\n")
documents = splitter.split_documents(raw_documents)

# Ragasによる合成テストデータの作成
for document in documents:
    # Ragasが使用するメタデータ filename を設定する
    document.metadata["filename"] = document.metadata["source"]

# Google Colab等で非同期処理を実行できるようにする。（この環境では不要かも）
# nest_asyncio.apply()

# Ragasの合成テストデータ生成機能の初期化
generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model=LLM_MODEL),  # 合成テストデータの質問や回答を生成するLLM
    critic_llm=ChatOpenAI(model=LLM_MODEL),  # 生成されたテストデータの品質を評価するLLM
    embeddings=OpenAIEmbeddings())  # ドキュメントのベクトル化に使用する埋め込みモデル

testset = generator.generate_with_langchain_docs(
    documents,  # filenameメタデータが設定されたドキュメント群
    test_size=4,  # 生成するテストデータの数
    distributions={  # 生成するテストデータの種類と割合
        simple: 0.5,  # 単純な質問の割合
        reasoning: 0.25,  # 回答に推論が必要な質問の割合
        multi_context: 0.25,  # 回答に複雑な情報源が必要な質問の割合（RAG前提か）
    })

# testset.to_pandas()  # PandasのDataFrameという見やすい表形式に変換して表示し、その内容や品質を簡単に確認することができる

dataset_name = DATA_SET_NAME  # データセット名
client = Client()  # LangSmithのクライアントを初期化

if client.has_dataset(dataset_name=dataset_name):
    client.delete_dataset(dataset_name=dataset_name)  # 既存の同名データセットが存在する場合は削除

data_set = client.create_dataset(dataset_name=dataset_name)  # 新しいデータセットを作成

# Ragasで生成したデータをLangSmithのDatasetに保存する
inputs = []
outputs = []
metadatas = []

for testset_record in testset.test_data:
    inputs.append({
        "question": testset_record.question,
    })
    outputs.append({
        "contexts": testset_record.contexts,
        "ground_truth": testset_record.ground_truth,
    })
    metadatas.append({
        "source": testset_record.metadata[0]["source"],
        "evolution_type": testset_record.evolution_type
    })

# LangSmithのDatasetにデータを一括登録する
# Datasetに保存するデータの一件一件を Example と呼ぶ
client.create_examples(inputs=inputs, outputs=outputs, metadatas=metadatas, dataset_id=data_set.id)

# LangSmith上でRagasのテストデータセットを確認済み
# https://smith.langchain.com/o/83a6cd41-a15d-4f9d-881f-b385d5ab3acb/datasets/cd992b9c-96a9-445d-b1fc-b41265bef433?tab=1

# LangSmithとRagasを使ったオフライン評価の実装
# | 評価対象   | 評価メトリクス                                         | 概要                                                                                 | LLM を使用 | Embedding を使用 |
# | -------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------ | -------- | ---------------- |
# | 検索      | Context precision（コンテキストの適合率）                | 質問と期待する回答を踏まえて、実際の検索結果のうち有用だと LLM で推論される割合                   | ○        |                  |
# | 検索      | Context recall（コンテキストの再現率）                   | 期待する回答をいくつかの文章に分割したうち、実際の検索結果で説明できる割合                        | ○        |                  |
# | 検索      | Context entity recall（コンテキストのエンティティの再現率） | 期待する回答に含まれるエンティティ（物事）のうち、実際の検索結果に含まれる割合                    | ○        |                  |
# | 生成      | Answer relevancy（回答の関連性）                        | 実際の回答が質問にどれだけ関連するか                                                       | ○        | ○                |
# | 生成      | Faithfulness（忠実性）                                 | 実際の回答に含まれる主張のうち、コンテキストに基づく事実と一致している割合                        | ○        |                  |
# | 検索＋生成 | Answer similarity（回答の類似性）                       | 実際の回答と期待する回答の、埋め込みベクトルのコサイン類似度                                    |          | ○                |
# | 検索＋生成 | Answer correctness（回答の正確性）                      | 実際の回答と期待する回答の、事実性（faithfulness）と意味的類似性（Answer similarity）の加重平均  | ○        | ○                |


# Ragasのメトリクスを使ってLangSmithの実行結果を評価するクラス
class RagasMetricEvaluator:

    # 評価メトリクスの初期化
    def __init__(self, metric: Metric, llm: BaseChatModel, embeddings: Embeddings):
        self.metric = metric

        # メトリクスがLLMを必要とする場合、LangChain用のラッパーでLLMを設定
        if isinstance(self.metric, MetricWithLLM):
            self.metric.llm = LangchainLLMWrapper(llm)
        # メトリクスがembeddingsを必要とする場合、埋め込みモデルを設定
        if isinstance(self.metric, MetricWithEmbeddings):
            self.metric.embeddings = LangchainEmbeddingsWrapper(embeddings)

    # 実際のRAG実行結果とテストデータを比較して評価スコアを計算
    def evaluate(self, run: Run, example: Example) -> dict[str, Any]:
        if run.outputs is None:
            raise ValueError("Run outputs is None")
        if example.inputs is None:
            raise ValueError("Example inputs is None")
        if example.outputs is None:
            raise ValueError("Example outputs is None")

        # LangSmithの検索結果（Documentオブジェクト）からテキスト部分を抽出
        # context_strs = [doc.page_content for doc in run.outputs["contexts"]]
        # # Ragasメトリクスで評価実行（質問、実際の回答、検索結果、正解を渡す）
        # score = self.metric.score({
        #     "question": example.inputs["question"],  # 質問
        #     "answer": run.outputs["answer"],  # 実際の回答
        #     "contexts": context_strs,  # 実際の検索結果
        #     "ground_truth": example.outputs["ground_truth"],  # 期待する回答
        # })

        payload = {
            "question": example.inputs["question"],
            "answer": run.outputs["answer"],
            "contexts": [doc.page_content for doc in run.outputs["contexts"]],
            "ground_truth": example.outputs["ground_truth"],
        }

        # ★ ここで ascore を asyncio.run で回す
        if hasattr(self.metric, "ascore"):
            score = asyncio.run(self.metric.ascore(payload))
        else:
            # 非同期APIが無い場合のフォールバック
            score = self.metric.score(payload)

        # LangSmith用の評価結果形式で返す
        return {"key": self.metric.name, "score": score}


# 料金節約のため評価対象を絞る
metrics = [
    context_precision,  # 質問と期待する回答を踏まえて、実際の検索結果のうち有用だと LLM で推論される割合
    answer_relevancy  # 実際の回答が質問にどれだけ関連するか
]

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
# 各メトリクスに対してRagasMetricEvaluatorを作成し、その評価関数をリスト化
evaluators = [RagasMetricEvaluator(metric, llm, embeddings).evaluate for metric in metrics]
db = Chroma.from_documents(documents,
                           embeddings,
                           client_settings=Settings(anonymized_telemetry=False))
prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください
文脈: """
{context}
"""

質問: {question}
''')

model = ChatOpenAI(model=LLM_MODEL, temperature=0)
retriever = db.as_retriever()  # DBから検索機（retriever）を作成する
chain = RunnableParallel({
    "question": RunnablePassthrough(),
    "context": retriever,
}).assign(answer=prompt | model | StrOutputParser())


# evaluateが期待するキーを返す
def predict(inputs: dict[str, Any]) -> dict[str, Any]:
    question = inputs["question"]
    output = chain.invoke(question)
    return {
        "contexts": output["context"],
        "answer": output["answer"],
    }


evaluate(
    predict,
    data=DATA_SET_NAME,
    evaluators=evaluators,
    max_concurrency=1,
)
