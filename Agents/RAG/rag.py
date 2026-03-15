"""
rag.py — PDF ingestion and retriever setup.
"""

import pathlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR   = pathlib.Path("data")
VECTOR_DIR = pathlib.Path("vectorstore")

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def build_retriever(k: int = 4):
    """
    Ingests all PDFs in DATA_DIR into a persisted Chroma vector store.
    Skips ingestion if the store already exists (idempotent).
    Returns a LangChain retriever.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Re-use existing store if already built
    if VECTOR_DIR.exists() and any(VECTOR_DIR.iterdir()):
        print("[RAG] Loading existing vector store...")
        db = Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embeddings)
        return db.as_retriever(search_kwargs={"k": k})

    # First-time ingestion
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"[RAG] No PDFs found in {DATA_DIR.resolve()}")

    print(f"[RAG] Ingesting {len(pdfs)} PDFs...")
    docs = []
    for pdf in pdfs:
        print(f"  📄 {pdf.name}")
        docs.extend(PyPDFLoader(str(pdf)).load())

    chunks = SPLITTER.split_documents(docs)
    print(f"[RAG] {len(chunks)} chunks → embedding & storing...")

    db = Chroma.from_documents(chunks, embeddings, persist_directory=str(VECTOR_DIR))
    print("[RAG] ✓ Vector store ready.")
    return db.as_retriever(search_kwargs={"k": k})