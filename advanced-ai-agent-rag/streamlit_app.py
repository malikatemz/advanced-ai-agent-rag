"""
Simple Streamlit front-end for the RAG + Agent system.
Run with:
    streamlit run streamlit_app.py

Talks to the FastAPI backend (default http://localhost:8000). Set API_BASE_URL
env var to point elsewhere.
"""
import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Agent / RAG System", page_icon="🧠", layout="wide")
st.title("🧠 Advanced AI Agent / RAG System")
st.caption("Document Q&A and multi-step reasoning, powered by Claude.")

tab_ingest, tab_query, tab_agent, tab_status = st.tabs(
    ["📥 Ingest", "🔍 Ask (RAG)", "🤖 Agent", "📊 Status"]
)

# --- Ingest tab -------------------------------------------------------------
with tab_ingest:
    st.subheader("Upload a document")
    uploaded = st.file_uploader("PDF, TXT, MD, or DOCX", type=["pdf", "txt", "md", "docx"])
    if uploaded and st.button("Ingest document", type="primary"):
        with st.spinner("Chunking, embedding, and indexing..."):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            try:
                resp = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Ingested '{data['filename']}' — {data['chunks_added']} chunks added.")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

# --- RAG query tab -----------------------------------------------------------
with tab_query:
    st.subheader("Ask a question about your documents")
    query = st.text_input("Question", key="rag_query")
    top_k = st.slider("Passages to retrieve", 1, 10, 5)
    if st.button("Ask", type="primary", key="rag_ask"):
        if not query.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Retrieving and generating..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/query", json={"query": query, "top_k": top_k}, timeout=60
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.markdown("### Answer")
                    st.write(data["answer"])
                    if data["sources"]:
                        st.markdown("### Sources")
                        for s in data["sources"]:
                            st.markdown(
                                f"**[{s['index']}] {s['source']}** (score: {s['score']:.3f})\n\n"
                                f"> {s['preview']}"
                            )
                except Exception as e:
                    st.error(f"Query failed: {e}")

# --- Agent tab ----------------------------------------------------------------
with tab_agent:
    st.subheader("Multi-step reasoning agent")
    st.caption("Can chain document retrieval, calculator, and web search across multiple steps.")
    agent_query = st.text_input("Task or question", key="agent_query")
    max_iter = st.slider("Max reasoning steps", 1, 12, 6)
    if st.button("Run agent", type="primary", key="agent_run"):
        if not agent_query.strip():
            st.warning("Enter a task first.")
        else:
            with st.spinner("Agent is reasoning..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/agent",
                        json={"query": agent_query, "max_iterations": max_iter},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.markdown("### Final Answer")
                    st.write(data["answer"])
                    with st.expander("🔎 Reasoning trace"):
                        for step in data["steps"]:
                            if step["type"] == "text":
                                st.markdown(f"**💭 Reasoning:** {step['content']}")
                            elif step["type"] == "tool_call":
                                st.markdown(f"**🔧 Called `{step['tool_name']}`** with `{step['tool_input']}`")
                            elif step["type"] == "tool_result":
                                st.markdown(f"**📄 Result from `{step['tool_name']}`:**")
                                st.code(step["content"])
                except Exception as e:
                    st.error(f"Agent run failed: {e}")

# --- Status tab -----------------------------------------------------------
with tab_status:
    st.subheader("Knowledge base status")
    if st.button("Refresh status"):
        pass
    try:
        resp = requests.get(f"{API_BASE_URL}/status", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        st.metric("Total chunks indexed", data["total_chunks"])
        st.markdown("### Documents")
        if not data["documents"]:
            st.info("No documents ingested yet.")
        for doc in data["documents"]:
            col1, col2 = st.columns([4, 1])
            col1.write(doc)
            if col2.button("Delete", key=f"del_{doc}"):
                requests.delete(f"{API_BASE_URL}/documents/{doc}")
                st.rerun()
    except Exception as e:
        st.error(f"Could not reach backend at {API_BASE_URL}: {e}")
