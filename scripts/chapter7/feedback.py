from uuid import UUID

import ipywidgets as widgets
from IPython.display import display
from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tracers.context import collect_runs
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langsmith import Client


def file_filter(file_path: str) -> bool:
    return file_path.endswith('.mdx')


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
raw_documents = loader.load()
documents = text_splitter.split_documents(raw_documents)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
db = Chroma.from_documents(documents, embeddings)
prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈："""
{context}
"""

質問：{question}
''')

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
retriever = db.as_retriever()
chain: Runnable = ({
    "question": RunnablePassthrough(),
    "context": retriever,
}) | prompt | model | StrOutputParser()


def display_feedback_buttons(run_id: UUID) -> None:
    good_button = widgets.Button(
        description="Good",
        button_style="success",
        icon="thumbs-up",
    )
    bad_button = widgets.Button(
        description="Bad",
        button_style="danger",
        icon="thumbs-down",
    )

    def on_button_clicked(button: widgets.Button) -> None:
        if button.description == "Good":
            score = 1
        elif button.description == "Bad":
            score = 0
        else:
            raise ValueError("Unknown button: {button}")

        client = Client()
        client.create_feedback(run_id=run_id, key="thumbs", score=score)
        print("フィードバックを保存しました")

    good_button.on_click(on_button_clicked)
    bad_button.on_click(on_button_clicked)
    display(good_button, bad_button)


with collect_runs() as runs_cb:
    output = chain.invoke("LangChainの概要を教えて")
    print(output["answer"])
    run_id = runs_cb[0].traced_runs[0].id

display_feedback_buttons(run_id)
