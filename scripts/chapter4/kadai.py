import os

from dotenv import load_dotenv
from langchain_community.document_loaders import NotionDBLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
notion_token = os.getenv("NOTION_TOKEN")
database_id = os.getenv("NOTION_DATABASE_ID")

loader = NotionDBLoader(
    integration_token=notion_token,
    database_id=database_id,
    request_timeout_sec=30
)

raw_docs = loader.load()
# print(document)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(raw_docs)
# print(len(docs))

def clean_metadata(doc):
    new_metadata = {}
    for k, v in doc.metadata.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            new_metadata[k] = v
        elif isinstance(v, list) and len(v) == 0:
            continue
        elif isinstance(v, list) and all(isinstance(item, (str, int, float, bool)) for item in v):
            new_metadata[k] = str(v)
        else:
            continue
    doc.metadata = new_metadata
    return doc

cleaned_docs = [clean_metadata(doc) for doc in docs]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(cleaned_docs, embeddings)

retriever = db.as_retriever()
context_docs = retriever.invoke("望月さんってCaratで何してる人？")
first_doc = context_docs[0]
print(f"len = {len(context_docs)}")
print(f"first_doc: {first_doc.page_content[:200]}")
