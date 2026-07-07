# RAG Document Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you query and summarize
document collections through natural-language, context-grounded conversation.

**Stack:** Python · LangChain · OpenAI API · ChromaDB · Streamlit

## What it does

1. **Ingestion** (`ingest.py`) — loads `.txt`/`.md`/`.pdf` files, splits them into
   overlapping chunks, embeds each chunk with OpenAI's `text-embedding-3-small`,
   and persists the vectors in a local ChromaDB collection.
2. **Retrieval + generation** (`rag_chain.py`) — embeds the user's question,
   runs a semantic similarity search against ChromaDB to pull the top-k most
   relevant chunks, then stuffs those chunks into a prompt that instructs the
   LLM (`gpt-4o-mini`) to answer *only* from that context — this is what grounds
   the output and cuts down on hallucination.
3. **UI** (`app.py`) — a Streamlit chat interface where you can upload your own
   documents (or use the bundled samples), ask questions, and see exactly which
   source chunks the answer was based on.

## Setup

```bash
git clone <this repo>
cd rag-chatbot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

## Run it

**Option A — CLI (fastest way to see the pipeline work):**

```bash
python ingest.py            # builds the vector store from sample_docs/
python rag_chain.py         # asks a sample question, prints answer + sources
```

**Option B — Streamlit app (the full experience):**

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints, click **"Build / Rebuild Vector Store"**
in the sidebar (sample docs work out of the box), and start asking questions like:

- "How many PTO days do employees get?"
- "What's the API rate limit on the Business tier?"
- "What happens if uptime falls below the SLA?"

Or upload your own PDFs/text files and ask questions about those instead.

## Project structure

```
rag-chatbot/
├── app.py              # Streamlit UI
├── ingest.py           # Load → chunk → embed → store pipeline
├── rag_chain.py        # Retrieval + grounded-prompt generation chain
├── requirements.txt
├── sample_docs/        # Example documents so it runs out of the box
└── chroma_db/          # Created at runtime — persisted vector store
```

## Design decisions worth knowing for an interview

- **Chunking (1000 chars, 150 overlap):** big enough to preserve context within
  a chunk, small enough that embeddings stay topically focused. Overlap prevents
  losing information that straddles a chunk boundary.
- **`RecursiveCharacterTextSplitter`:** splits on paragraph → line → sentence →
  word boundaries in priority order, so chunks break at natural points instead
  of mid-sentence.
- **Grounding prompt:** the system prompt explicitly tells the model to answer
  only from retrieved context and to admit when it doesn't know — this is the
  main lever for hallucination reduction, separate from retrieval quality itself.
- **Source display:** every answer shows which chunks it was built from, so a
  user (or you, demoing it) can verify the answer against the original text.
- **LCEL (`|` chain syntax):** the retrieval → prompt → LLM → parser chain is
  built with LangChain Expression Language, which is the current idiomatic way
  to compose LangChain pipelines (as opposed to older `RetrievalQA` chain
  classes, which are being phased out).

## Possible extensions (good talking points for "what would you add next")

- Swap in `langgraph` for multi-step retrieval (e.g., query rewriting, re-ranking,
  multi-hop retrieval across document sets).
- Add conversation memory so follow-up questions can reference prior turns.
- Add a re-ranking step (e.g., Cohere rerank or cross-encoder) after initial
  vector retrieval to improve precision on the top-k results.
- Swap OpenAI for a local model (via Ollama) or add a provider toggle (OpenAI /
  Claude / Gemini) to show multi-LLM-API experience.
- Add evaluation (e.g., RAGAS) to measure retrieval precision/recall and
  answer faithfulness against a labeled test set.
# rag-doc-bot
