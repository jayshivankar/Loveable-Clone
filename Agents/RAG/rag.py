"""
RAG module — PDF ingestion and retriever setup.
Paths are anchored to __file__ so the module works regardless of cwd.
"""

import os
import pathlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths anchored to this file's location — works from any working directory
_HERE       = pathlib.Path(__file__).parent.resolve()
DATA_DIR    = _HERE / "Data"

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def build_retriever(k: int = 4):
    """
    Ingests all PDFs in DATA_DIR into a persisted PGVector store.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/codeforge",
    )
    url = url.replace("postgresql://", "postgresql+psycopg://")

    # Connect to the exact same Postgres DB used for episodic memory
    collection_name = "codeforge_kb"

    db = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=url,
        use_jsonb=True,
    )

    try:
        # Check if store already contains vectors
        dummy = db.similarity_search("test", k=1)
        is_empty = len(dummy) == 0
    except Exception:
        is_empty = True

    if not is_empty:
        print("[RAG] Loading existing PGVector store…")
        return db.as_retriever(search_kwargs={"k": k})

    # First-time ingestion
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[RAG] No PDFs found in {DATA_DIR} — RAG will be empty.")
        return db.as_retriever(search_kwargs={"k": k})

    print(f"[RAG] Ingesting {len(pdfs)} PDF(s)…")
    docs = []
    for pdf in pdfs:
        print(f"  📄 {pdf.name}")
        docs.extend(PyPDFLoader(str(pdf)).load())

    chunks = SPLITTER.split_documents(docs)
    print(f"[RAG] {len(chunks)} chunks → embedding & storing inside Postgres…")

    db.add_documents(chunks)
    print("[RAG] ✓ Vector store ready.")
    return db.as_retriever(search_kwargs={"k": k})