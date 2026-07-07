"""
ingest.py
---------
Handles the "front half" of the RAG pipeline:

    documents on disk  ->  loaded  ->  chunked  ->  embedded  ->  stored in ChromaDB

Run this once (or any time your source documents change) to (re)build the
vector store that the chatbot queries against at runtime.

Usage:
    python ingest.py                      # ingest everything in ./sample_docs
    python ingest.py --source ./my_docs   # ingest a different folder
    python ingest.py --reset              # wipe and rebuild the vector store
"""

import argparse
import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_chatbot_docs"

# Chunking parameters. These matter a lot in practice:
#   - too small -> chunks lose context, retrieval returns fragments
#   - too large -> chunks dilute relevance, waste context window, raise cost
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_documents(source_dir: str):
    """Load .txt, .md, and .pdf files from a directory into LangChain Documents."""
    loaders = [
        DirectoryLoader(source_dir, glob="**/*.txt", loader_cls=TextLoader,
                         show_progress=True),
        DirectoryLoader(source_dir, glob="**/*.md", loader_cls=TextLoader,
                         show_progress=True),
        DirectoryLoader(source_dir, glob="**/*.pdf", loader_cls=PyPDFLoader,
                         show_progress=True),
    ]
    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except ValueError:
            # DirectoryLoader raises if zero files match a given glob - that's fine
            continue

    if not documents:
        raise ValueError(
            f"No .txt, .md, or .pdf files found in '{source_dir}'. "
            "Add some documents and try again."
        )
    return documents


def chunk_documents(documents):
    """Split long documents into overlapping chunks sized for embedding + retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_vector_store(chunks, reset: bool = False):
    """Embed chunks and persist them to a local ChromaDB collection."""
    if reset and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"Cleared existing vector store at ./{CHROMA_DIR}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vector_store.add_documents(chunks)
    return vector_store


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument("--source", default="sample_docs", help="Folder of source documents")
    parser.add_argument("--reset", action="store_true", help="Wipe and rebuild the vector store")
    args = parser.parse_args()

    print(f"Loading documents from '{args.source}'...")
    documents = load_documents(args.source)
    print(f"Loaded {len(documents)} document(s).")

    print("Splitting into chunks...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunk(s) "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    print("Embedding chunks and writing to ChromaDB (this calls the OpenAI API)...")
    build_vector_store(chunks, reset=args.reset)
    print(f"Done. Vector store persisted at ./{CHROMA_DIR}")


if __name__ == "__main__":
    main()
