"""
rag_chain.py
------------
Handles the "back half" of the RAG pipeline:

    user question -> semantic retrieval -> grounded prompt -> LLM answer + sources

This is the retrieval-augmented generation logic itself, decoupled from the
Streamlit UI so it can be tested or reused independently (e.g. from a CLI or API).
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_chatbot_docs"

# This is the key lever for reducing hallucination: the prompt explicitly
# instructs the model to answer ONLY from retrieved context, and to say so
# when the context doesn't contain the answer, rather than guessing.
SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the \
provided context from the user's documents.

Rules:
- Answer strictly using the context below. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so plainly \
instead of guessing.
- When you use a fact from the context, keep your answer concise and directly tied \
to what's stated.
- Do not fabricate document names, numbers, or policies that are not in the context.

Context:
{context}
"""


def format_docs(docs):
    """Concatenate retrieved chunks into a single context block for the prompt."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def get_vector_store():
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"No vector store found at ./{CHROMA_DIR}. Run `python ingest.py` first."
        )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def build_rag_chain(k: int = 4, model: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    Build a LangChain LCEL chain that:
      1. Embeds the user's question
      2. Retrieves the top-k most similar chunks from ChromaDB (semantic search)
      3. Stuffs those chunks into a grounded prompt
      4. Sends it to the LLM and parses the response

    Returns (chain, retriever) so the caller can also inspect retrieved sources
    separately from the final answer (useful for citing sources in the UI).
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model=model, temperature=temperature)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def answer_question(question: str, k: int = 4):
    """Convenience wrapper: run the chain and return both the answer and source docs."""
    chain, retriever = build_rag_chain(k=k)
    sources = retriever.invoke(question)
    answer = chain.invoke(question)
    return answer, sources


if __name__ == "__main__":
    # Quick CLI smoke test: `python rag_chain.py`
    q = "How many PTO days do employees get, and how does that change over time?"
    answer, sources = answer_question(q)
    print("QUESTION:", q)
    print("\nANSWER:", answer)
    print("\nSOURCES USED:")
    for s in sources:
        print(" -", s.metadata.get("source", "unknown"))
