from langchain_community.document_loaders import GitLoader

def file_filter(file_path:str) -> bool:
    return file_path.endswith(".mdx")

loader = GitLoader(
    clone_url='https://github.com/langchain-ai/langchain',
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter
)

raw_docs = loader.load()

# print(len(raw_docs)) # 439 書籍より増えてる




# # Document transformer
from  langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(raw_docs)
# print(len(docs)) # 1569 raw_docsをチャンク分割



# Embedding model
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

query = "AWSのS3からデータを読み込むためのDocument Loaderはありますか？"

vector = embeddings.embed_query(query)
# print(len(vector)) # 1536
# print(vector)
# """
# [0.016827309504151344, -0.004892522003501654, 0.02605109103024006, -0.02810552529990673, 0.05337296426296234, 0.02393311634659767, -0.01697556860744953, 0.02175160124897957, 0.01913590356707573, -0.015905991196632385, 0.002160334261134267, -0.011511193588376045, -0.013978634029626846, -0.0016546678962185979, -0.01671082153916359, 0.05858318507671356, 0.03484068438410759, -0.003253738861531019, -0.04015680402517319, 0.030604736879467964, 0.029651647433638573, 0.04110989347100258, -0.04502814635634422, 0.02095736190676689, 0.023467160761356354, ...以下省略
# """



from langchain_chroma import Chroma
db = Chroma.from_documents(docs, embeddings)

retriever = db.as_retriever()

context_docs = retriever.invoke(query)
first_doc = context_docs[0]
# print(f"len = {len(context_docs)}")
# print(f"metadata = {first_doc.metadata}")
# print(first_doc.page_content)
"""
len = 4
metadata = {'file_type': '.mdx', 'source': 'docs/docs/integrations/providers/aws.mdx', 'file_name': 'aws.mdx', 'file_path': 'docs/docs/integrations/providers/aws.mdx'}
## Document loaders

### AWS S3 Directory and File

>[Amazon Simple Storage Service (Amazon S3)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-folders.html)
> is an object storage service.
>[AWS S3 Directory](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-folders.html)
>[AWS S3 Buckets](https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingBucket.html)

See a [usage example for S3DirectoryLoader](/docs/integrations/document_loaders/aws_s3_directory).

See a [usage example for S3FileLoader](/docs/integrations/document_loaders/aws_s3_file).

```python
from langchain_community.document_loaders import S3DirectoryLoader, S3FileLoader
```

### Amazon Textract

>[Amazon Textract](https://docs.aws.amazon.com/managedservices/latest/userguide/textract.html) is a machine
> learning (ML) service that automatically extracts text, handwriting, and data from scanned documents.

See a [usage example](/docs/integrations/document_loaders/amazon_textract).
"""


# LCELを使ったRAGのChainの実装
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("""\
以下の文脈だけを踏まえて質問に回答してください。

文脈: '''
{context}
'''

質問: {question}
""")

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

output = chain.invoke(query)
print(output)