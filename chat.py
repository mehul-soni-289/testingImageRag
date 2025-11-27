"""
Terminal chatbot for querying Supabase-backed multi-vector retriever.
"""

from __future__ import annotations

import argparse

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

from .config import settings
from .docstore import LocalJSONDocStore


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant context found."
    parts = []
    for doc in docs:
        meta = doc.metadata or {}
        parts.append(
            (
                f"PDF: {meta.get('pdf_id')}\n"
                f"Page: {meta.get('page_no')}\n"
                f"Image Path: {meta.get('image_path')}\n"
                f"Summary:\n{doc.page_content}"
            )
        )
    return "\n\n---\n\n".join(parts)


def _build_vector_components():
    settings.validate_vector_store()

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
    )
    client = create_client(settings.supabase_url, settings.supabase_key)
    vectorstore = SupabaseVectorStore(
        client=client,
        table_name=settings.supabase_table,
        query_name=settings.supabase_query_fn,
        embedding=embeddings,
    )
    docstore = LocalJSONDocStore(settings.docstore_dir)
    return vectorstore, docstore


def _build_retrieval_lambda(vectorstore: SupabaseVectorStore, docstore: LocalJSONDocStore):
    def _retrieve(question: str) -> List[Document]:
        base_docs = vectorstore.similarity_search(
            question, k=settings.retriever_top_k, filter=None
        )
        doc_ids = [doc.metadata.get("doc_id") for doc in base_docs]
        store_docs = docstore.mget(doc_ids)

        merged: List[Document] = []
        for fallback, stored in zip(base_docs, store_docs):
            merged.append(stored or fallback)
        return merged

    return RunnableLambda(_retrieve)


def build_chain():
    vectorstore, docstore = _build_vector_components()
    retrieval_lambda = _build_retrieval_lambda(vectorstore, docstore)

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a business intelligence assistant. "
                "Use the provided context summaries and referenced page info to answer. "
                "If the answer is not explicitly available, say you do not know.",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    return (
        {
            "context": retrieval_lambda | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal chatbot for PDF summaries.")
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        help="Optional single question to ask. If omitted, enters interactive mode.",
    )
    args = parser.parse_args()

    chain = build_chain()

    if args.question:
        answer = chain.invoke(args.question)
        print("\nAnswer:\n")
        print(answer)
        return

    print("Interactive mode. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question or question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        answer = chain.invoke(question)
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()

