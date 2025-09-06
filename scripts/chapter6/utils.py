from typing import Any

from langchain_cohere import CohereRerank
from langchain_core.documents import Document


def reciprocal_rank_fusion(
    retriever_outputs: list[list[Document]],
    k: int = 60,
) -> list[str]:
    content_score_mapping = {}

    for docs in retriever_outputs:
        for rank, doc in enumerate(docs):
            content = doc.page_content

            if content not in content_score_mapping:
                content_score_mapping[content] = 0.0

            content_score_mapping[content] += 1 / (rank + k)

    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)
    return [content for content, _ in ranked[:k]]


def rerank(input: dict[str, Any], top_n: int = 3) -> list[Document]:
    question = input["question"]
    documents = input["documents"]

    cohere_rerank = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)
    return cohere_rerank.compress_documents(documents, query=question)
