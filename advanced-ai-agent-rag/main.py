"""
FastAPI entry point for the RAG + Agent system.

Run with:
    uvicorn main:app --reload --port 8000
or simply:
    python main.py
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agent import run_agent
from src.config import get_settings
from src.ingestion import ingest_file
from src.rag import answer_query
from src.vectorstore import get_vectorstore

app = FastAPI(
    title="Advanced AI Agent / RAG System",
    description="Document Q&A (RAG) and multi-step tool-using agent, powered by Claude.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- 
# Schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    filter_source: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str


class AgentRequest(BaseModel):
    query: str
    max_iterations: int | None = 6


class AgentResponse(BaseModel):
    answer: str
    steps: list[dict]


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int


class StatusResponse(BaseModel):
    documents: list[str]
    total_chunks: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["meta"])
def root():
    return {
        "name": "Advanced AI Agent / RAG System",
        "endpoints": ["/ingest", "/query", "/agent", "/status", "/documents/{source}"],
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
def health():
    settings = get_settings()
    return {"status": "ok", "model": settings.claude_model, "embedding_provider": settings.embedding_provider}


@app.post("/ingest", response_model=IngestResponse, tags=["documents"])
async def ingest(file: UploadFile = File(...)):
    """Upload and ingest a document (.pdf, .txt, .md, .docx) into the vector store."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".docx"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Preserve the original filename as the source label.
        target = tmp_path.with_name(file.filename)
        shutil.move(str(tmp_path), str(target))
        chunks = ingest_file(target)
        store = get_vectorstore()
        added = store.add_chunks(chunks)
    finally:
        try:
            target.unlink(missing_ok=True)
        except Exception:
            pass

    return IngestResponse(filename=file.filename, chunks_added=added)


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest):
    """Ask a question answered strictly from the ingested documents (RAG)."""
    result = answer_query(req.query, top_k=req.top_k, filter_source=req.filter_source)
    return QueryResponse(answer=result.answer, sources=result.sources, query=result.query)


@app.post("/agent", response_model=AgentResponse, tags=["agent"])
def agent(req: AgentRequest):
    """Run the multi-step tool-using agent (retrieval + calculator + web search)."""
    result = run_agent(req.query, max_iterations=req.max_iterations or 6)
    steps = [
        {"type": s.type, "content": s.content, "tool_name": s.tool_name, "tool_input": s.tool_input}
        for s in result.steps
    ]
    return AgentResponse(answer=result.final_answer, steps=steps)


@app.get("/status", response_model=StatusResponse, tags=["documents"])
def status():
    store = get_vectorstore()
    return StatusResponse(documents=store.list_sources(), total_chunks=store.count())


@app.delete("/documents/{source}", tags=["documents"])
def delete_document(source: str):
    store = get_vectorstore()
    deleted = store.delete_source(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for source '{source}'")
    return {"source": source, "chunks_deleted": deleted}


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=True)
