import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import NotionDBLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
# filter_complex_metadataのインポートを削除（エラーの原因）

# ======================
# 0. .env読み込み
# ======================
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
notion_token = os.getenv("NOTION_TOKEN")
database_id = os.getenv("NOTION_DATABASE_ID")

# ======================
# 1. Notionからデータ取得
# ======================
loader = NotionDBLoader(
    integration_token=notion_token,
    database_id=database_id,
    request_timeout_sec=30
)

raw_docs = loader.load()
print(f"Loaded {len(raw_docs)} documents from Notion")

# デバッグ: raw_docsの内容を確認
print(f"Type of raw_docs: {type(raw_docs)}")
if raw_docs:
    print(f"Type of first item: {type(raw_docs[0])}")
    if isinstance(raw_docs[0], tuple):
        print(f"  First item is tuple with {len(raw_docs[0])} elements")
        if len(raw_docs[0]) > 0:
            print(f"  Element 0 type: {type(raw_docs[0][0])}")
        if len(raw_docs[0]) > 1:
            print(f"  Element 1 type: {type(raw_docs[0][1])}")

# ======================
# 2. raw_docsをDocumentに正規化
# ======================
docs = []
for i, item in enumerate(raw_docs):
    # すでにDocumentの場合
    if isinstance(item, Document):
        docs.append(item)
    # tupleの場合（古いバージョンのloader.load()の出力形式）
    elif isinstance(item, tuple) and len(item) == 2:
        page_content, metadata = item
        # metadataが辞書でない場合は空の辞書にする
        if not isinstance(metadata, dict):
            metadata = {}
        docs.append(Document(page_content=str(page_content), metadata=metadata))
    # その他の形式
    else:
        print(f"Warning: Unexpected format at index {i}: {type(item)}")
        docs.append(Document(page_content=str(item), metadata={}))

# デバッグ: 変換後の型を確認
print(f"After conversion: {len(docs)} documents")
for i, doc in enumerate(docs[:3]):  # 最初の3つだけ確認
    print(f"  Doc {i}: type={type(doc)}, has_metadata={hasattr(doc, 'metadata')}")

# ======================
# 3. メタデータをクリーンアップ（カスタム実装）
# ======================
def clean_metadata(metadata):
    """メタデータから複雑な型を除去"""
    if not isinstance(metadata, dict):
        return {}
    
    cleaned = {}
    for key, value in metadata.items():
        # 基本的な型のみを保持
        if isinstance(value, (str, int, float, bool, type(None))):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)):
            # リストやタプルは文字列に変換
            cleaned[key] = str(value)
        elif isinstance(value, dict):
            # 辞書は文字列に変換
            cleaned[key] = str(value)
        else:
            # その他の型は文字列に変換
            cleaned[key] = str(value)
    return cleaned

cleaned_docs = []
for i, doc in enumerate(docs):
    if isinstance(doc, Document):
        # メタデータをクリーンアップ
        clean_meta = clean_metadata(doc.metadata)
        cleaned_docs.append(Document(page_content=doc.page_content, metadata=clean_meta))
    else:
        print(f"Warning: Doc {i} is not a Document: {type(doc)}")
        cleaned_docs.append(Document(page_content=str(doc), metadata={}))
docs = cleaned_docs

print(f"Prepared {len(docs)} documents after cleaning")

# ======================
# 4. ベクトルDBに保存
# ======================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(docs, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})

# ======================
# 5. プロンプト定義
# ======================
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "あなたはCarat社の社内アシスタントです。次の情報を参考にユーザーの質問に答えてください。\n\n{context}"),
    ("human", "{question}")
])

# ======================
# 6. LCELでチェーンを構築
# ======================
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": retriever | format_docs, "question": lambda x: x["question"]}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

# ======================
# 7. 実行例
# ======================
query = "望月さんってCaratで何してる人？"
result = chain.invoke({"question": query})

print("=== 質問 ===")
print(query)
print("=== 回答 ===")
print(result.content)
