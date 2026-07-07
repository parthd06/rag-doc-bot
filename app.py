"""
app.py
------
Streamlit front-end for the RAG chatbot.

Lets a user:
  - Upload their own documents (.txt, .md, .pdf) or use the bundled sample docs
  - Ask natural-language questions about them
  - See the chatbot's grounded answer plus which source chunks it used

Run with:  streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from ingest import load_documents, chunk_documents, build_vector_store
from rag_chain import build_rag_chain

load_dotenv()

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Light styling pass — keeps default Streamlit widgets but tightens spacing,
# adds a subtle card look to chat bubbles, and gives the header some presence.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 2.2rem; max-width: 900px; }
        [data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.4rem;
        }
        .app-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.1rem;
        }
        .app-header h1 { font-size: 1.9rem; margin: 0; }
        .app-subtitle { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.2rem; }
        .stButton>button { border-radius: 8px; }
        [data-testid="stExpander"] { border-radius: 10px; }
        .example-chip {
            display: inline-block;
            background: #f1f5f9;
            border-radius: 999px;
            padding: 0.3rem 0.8rem;
            font-size: 0.85rem;
            margin: 0.2rem 0.3rem 0.2rem 0;
            color: #334155;
        }
    </style>
    <div class="app-header">
        <h1>📚 RAG Document Chatbot</h1>
    </div>
    <div class="app-subtitle">
        Ask questions about your documents. Every answer is grounded in retrieved
        passages, with sources shown for full transparency.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: document ingestion controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📄 Document Source")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set. Add it to a `.env` file (see README).")

    source_choice = st.radio(
        "Choose a document source",
        ["Use bundled sample docs", "Upload my own files"],
        label_visibility="collapsed",
    )

    uploaded_files = None
    if source_choice == "Upload my own files":
        uploaded_files = st.file_uploader(
            "Upload .txt, .md, or .pdf files",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    build_clicked = st.button("🔨 Build / Rebuild Vector Store", type="primary", use_container_width=True)

    if build_clicked:
        with st.spinner("Ingesting: loading → chunking → embedding → storing..."):
            try:
                if source_choice == "Use bundled sample docs":
                    docs = load_documents("sample_docs")
                else:
                    if not uploaded_files:
                        st.warning("Please upload at least one file first.")
                        st.stop()
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        for f in uploaded_files:
                            path = os.path.join(tmp_dir, f.name)
                            with open(path, "wb") as out:
                                out.write(f.getbuffer())
                        docs = load_documents(tmp_dir)

                chunks = chunk_documents(docs)
                build_vector_store(chunks, reset=True)
                st.session_state["vector_store_ready"] = True
                st.session_state["doc_count"] = len(docs)
                st.session_state["chunk_count"] = len(chunks)
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    if st.session_state.get("vector_store_ready"):
        c1, c2 = st.columns(2)
        c1.metric("Documents", st.session_state.get("doc_count", 0))
        c2.metric("Chunks", st.session_state.get("chunk_count", 0))
        st.success("Vector store ready ✓")

    st.divider()
    st.markdown("### ⚙️ Retrieval settings")
    k = st.slider("Chunks retrieved per question (k)", min_value=2, max_value=8, value=4)

    st.divider()
    with st.expander("🔧 How this works", expanded=False):
        st.markdown(
            "1. Load documents\n"
            "2. Split into overlapping chunks\n"
            "3. Embed chunks (OpenAI `text-embedding-3-small`)\n"
            "4. Store vectors in ChromaDB\n"
            "5. Retrieve top-k similar chunks per question\n"
            "6. Generate grounded answer (`gpt-4o-mini`)"
        )

# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

vector_store_exists = os.path.exists("chroma_db")

if not vector_store_exists:
    st.info("👈 Build the vector store from the sidebar first — sample docs work out of the box, no upload needed.")
else:
    if not st.session_state["messages"]:
        st.markdown("**Try asking:**")
        example_questions = [
            "Summarize this document in 3 sentences",
            "What are the key details I should know?",
            "Is there anything about dates or deadlines?",
        ]
        cols = st.columns(len(example_questions))
        clicked_example = None
        for col, eq in zip(cols, example_questions):
            if col.button(eq, use_container_width=True):
                clicked_example = eq
    else:
        clicked_example = None

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources used"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['source']}**")
                        st.text(s["snippet"])

    question = st.chat_input("Ask a question about your documents...") or clicked_example

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages and generating an answer..."):
                try:
                    chain, retriever = build_rag_chain(k=k)
                    retrieved_docs = retriever.invoke(question)
                    answer = chain.invoke(question)

                    st.markdown(answer)

                    sources = []
                    with st.expander("📎 Sources used"):
                        for doc in retrieved_docs:
                            source_name = doc.metadata.get("source", "unknown")
                            snippet = doc.page_content[:300] + (
                                "..." if len(doc.page_content) > 300 else ""
                            )
                            st.markdown(f"**{source_name}**")
                            st.text(snippet)
                            sources.append({"source": source_name, "snippet": snippet})

                    st.session_state["messages"].append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
