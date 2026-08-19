# Advanced AI Agent / RAG System
> LLM agent and RAG pipeline for niche use cases — intelligent document Q&A and multi-step reasoning

![Status](https://img.shields.io/badge/status-In%20Progress-yellow)
![Month](https://img.shields.io/badge/roadmap-Month%204-purple)

## 📋 Overview

A working RAG (retrieval-augmented generation) pipeline plus a multi-step
tool-using AI agent, built on Claude. Upload documents, ask grounded
questions with citations, or hand the agent an open-ended task that requires
chaining document search, arithmetic, and (optionally) live web search.

Part of a 5-month Cybersecurity + AI + Automation roadmap targeting $2K MRR.

## ✨ Features

- **Document ingestion** — PDF, DOCX, TXT, MD → cleaned, sentence-aware overlapping chunks
- **Vector indexing** — persistent ChromaDB collection, free local embeddings by default (sentence-transformers), or swap in OpenAI embeddings
- **RAG pipeline** — retrieval + Claude generation, answers grounded in context with `[n]` inline citations, refuses to hallucinate outside the retrieved passages
- **Multi-step agent** — Claude's native tool-use loop chaining `retrieve_documents`, `calculator`, `web_search` (optional), and `list_knowledge_base` across up to N reasoning steps, with a visible reasoning trace
- **FastAPI backend** — `/ingest`, `/query`, `/agent`, `/status`, `/documents/{source}`, auto docs at `/docs`
- **Streamlit UI** — tabs for ingesting docs, asking RAG questions, running the agent, and viewing knowledge-base status
- **CLI bulk ingest** — point it at a folder and index everything at once
- **Tests** — chunking logic and the sandboxed calculator tool

## 🛠️ Tech Stack

- Anthropic Claude API (generation + native tool use)
- ChromaDB (persistent local vector store)
- sentence-transformers (default local embeddings, no API key) — swappable for OpenAI embeddings
- FastAPI + Uvicorn
- Streamlit
- pypdf / python-docx for document parsing

## 🚀 Getting Started

```bash
git clone https://github.com/malikatemz/advanced-ai-agent-rag.git
cd advanced-ai-agent-rag

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY

# Run the API
python main.py
# → http://localhost:8000/docs

# In a second terminal, run the UI
streamlit run streamlit_app.py
```

### Docker (one-command deploy)

```bash
cp .env.example .env   # set ANTHROPIC_API_KEY first
docker compose up --build
# API   → http://localhost:8000/docs
# UI    → http://localhost:8501
```

The vector DB persists in a named Docker volume (`chroma_data`) across
restarts. Drop files into `./data/documents` and they're mounted straight
into the API container if you want to bulk-ingest with the CLI inside it:

```bash
docker compose exec api python scripts/ingest_cli.py ./data/documents
```

### Bulk-ingesting a folder of documents

```bash
python scripts/ingest_cli.py ./data/documents
```

### Ingesting via the API directly

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/report.pdf"

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What were the key findings?"}'

curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Find the revenue figure in the report and calculate a 15% increase on it."}'
```

## ⚙️ Configuration

All configuration lives in `.env` (see `.env.example`):

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Required — Claude API key | — |
| `CLAUDE_MODEL` | Model for generation/agent | `claude-sonnet-4-6` |
| `EMBEDDING_PROVIDER` | `local` or `openai` | `local` |
| `EMBEDDING_MODEL` | sentence-transformers model name | `all-MiniLM-L6-v2` |
| `TAVILY_API_KEY` | Enables the `web_search` agent tool | unset (tool disabled) |
| `CHROMA_PERSIST_DIR` | Where the vector DB is stored on disk | `./data/chroma` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters (characters) | `800` / `120` |
| `TOP_K` | Passages retrieved per query | `5` |

## 📂 Project Structure

```
advanced-ai-agent-rag/
├── main.py                  # FastAPI app (entry point)
├── streamlit_app.py         # Streamlit UI
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py            # Settings (pydantic-settings)
│   ├── ingestion.py         # Document loading + chunking
│   ├── embeddings.py        # Local / OpenAI embedding providers
│   ├── vectorstore.py       # ChromaDB wrapper
│   ├── rag.py                # RAG query pipeline (retrieve + generate + cite)
│   ├── agent.py              # Multi-step tool-using agent loop
│   └── tools/
│       └── definitions.py    # Tool schemas + implementations
├── scripts/
│   └── ingest_cli.py         # Bulk-ingest a directory
├── tests/
│   ├── test_ingestion.py
│   └── test_tools.py
├── data/
│   ├── documents/             # Drop files here for CLI ingestion
│   └── chroma/                 # Persistent vector DB (gitignored)
├── Dockerfile                  # API container
├── Dockerfile.streamlit        # UI container
├── docker-compose.yml          # api + ui, one command
└── .github/workflows/ci.yml    # lint + tests + docker build on push/PR
```

## 🧠 How it works

**RAG path** (`/query`): embed the question → similarity search in Chroma →
stuff the top-k passages into a system prompt that instructs Claude to answer
*only* from that context, with `[n]` citations → return answer + source list.

**Agent path** (`/agent`): Claude is given tool definitions and loops using
Anthropic's native tool-use protocol — it decides which tool(s) to call, we
execute them locally and feed results back, and it repeats until it has
enough information for a final answer (or hits `max_iterations`). The full
reasoning trace (thoughts, tool calls, tool results) is returned so you can
see exactly how it got there.

## 🗺️ Roadmap

- [x] Core RAG pipeline (ingest → chunk → embed → retrieve → generate)
- [x] Multi-step tool-using agent
- [x] FastAPI backend + Streamlit UI
- [x] Unit tests for chunking and the calculator tool
- [x] Docker Compose deployment
- [x] CI (GitHub Actions: lint, tests, docker build)
- [ ] Auth / multi-tenant document isolation
- [ ] Streaming responses in the UI
- [ ] Niche use-case fine-tuning (target vertical TBD)

## 👤 Author

**malikatemz** — [GitHub](https://github.com/malikatemz)

---
*Part of the [5-Month Cyber + AI + Automation Roadmap](https://github.com/malikatemz/github-portfolio-launch)*
