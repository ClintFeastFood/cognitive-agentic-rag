"""
app.py - Cognitive RAG Pipeline: Chainlit User Interface & FastAPI Backend

Provides a responsive Chainlit chat environment with:
  - Async PDF upload via drag-and-drop on chat start
  - Token-by-token response streaming
  - Expandable inline citations (cl.Text) for every context source
  - FastAPI health-check and status endpoints
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import chainlit as cl
from chainlit.element import Text as ClText
from fastapi import FastAPI
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ingestion import IngestionPipeline
from retrieval import RetrievalEngine, RetrievalResult, RankedFragment

logger = logging.getLogger(__name__)
#Initialize the 70B DeepSeek-R1 cloud layer via my free Groq token
Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY")
)

#Set up local embeddings layers to keep document processing fast
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# ---------------------------------------------------------------------------
# FastAPI sub-application for health / status
# ---------------------------------------------------------------------------
api = FastAPI(title="Cognitive RAG Pipeline", version="1.0.0")


@api.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@api.get("/status")
async def status() -> dict[str, object]:
    return {
        "pipeline_loaded": _pipeline is not None,
        "child_nodes": len(_pipeline.child_nodes) if _pipeline else 0,
        "parent_nodes": len(_pipeline.parent_nodes) if _pipeline else 0,
    }


# ---------------------------------------------------------------------------
# Module-level state (per-process, safe for single-instance deployments)
# ---------------------------------------------------------------------------
_pipeline: Optional[IngestionPipeline] = None
_engine: Optional[RetrievalEngine] = None


async def _ensure_pipeline() -> tuple[IngestionPipeline, RetrievalEngine]:
    """Lazily initialise the ingestion pipeline and retrieval engine."""
    global _pipeline, _engine
    if _pipeline is None:
        _pipeline = IngestionPipeline()
        await _pipeline.load_existing_index()
    if _engine is None:
        _engine = RetrievalEngine(pipeline=_pipeline)
    return _pipeline, _engine


# ---------------------------------------------------------------------------
# System prompt for the LLM
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = (
    "You are a precise, well-sourced research assistant. "
    "Answer the user's question using ONLY the provided context fragments. "
    "Cite sources inline as [1], [2], etc. corresponding to the fragment order. "
    "If the context does not contain enough information, say so explicitly."
)


def _build_context_block(fragments: list[RankedFragment]) -> str:
    """Format ranked fragments into a numbered context block for the LLM."""
    lines: list[str] = []
    for i, frag in enumerate(fragments, start=1):
        source_label = frag.source_file or "uploaded document"
        page_label = f", page {frag.page_number}" if frag.page_number else ""
        lines.append(
            f"[{i}] ({source_label}{page_label}, score={frag.score:.4f}, "
            f"type={frag.chunk_type}):\n{frag.text}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Chainlit event handlers
# ---------------------------------------------------------------------------
@cl.on_chat_start
async def on_chat_start() -> None:
    """Handle new chat session: display welcome message and offer PDF upload."""
    pipeline, engine = await _ensure_pipeline()

    await cl.Message(
        content=(
            "Welcome to the **Cognitive RAG Pipeline**.\n\n"
            "Upload one or more PDF documents to get started. "
            "You can drag and drop files directly into this chat, "
            "or click the attachment button below.\n\n"
            "Once documents are indexed, ask any question and I will "
            "retrieve, re-rank, and cite the most relevant passages."
        ),
    ).send()

    # Offer a file upload action.
    files = None
    while not files:
        files = await cl.AskFileMessage(
            content="Please upload a PDF to begin.",
            accept=["application/pdf"],
            max_size_mb=50,
            timeout=600,
        ).send()

    await _process_uploaded_files(files)


async def _process_uploaded_files(files: list) -> None:
    """Ingest uploaded PDFs and refresh the retrieval engine."""
    pipeline, engine = await _ensure_pipeline()

    msg = cl.Message(content="Processing uploaded documents...")
    await msg.send()

    # Save uploaded files to a temp directory for LlamaIndex to read.
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths: list[str] = []
        for f in files:
            dest = os.path.join(tmp_dir, f.name)
            # Chainlit file objects expose .path for the local path.
            source_path = getattr(f, "path", None)
            if source_path and Path(source_path).exists():
                file_paths.append(source_path)
            else:
                # Fallback: write content to temp file.
                content = await f.read() if hasattr(f, "read") else b""
                if content:
                    with open(dest, "wb") as fh:
                        fh.write(content)
                    file_paths.append(dest)

        if not file_paths:
            await cl.Message(content="No valid files detected. Please try again.").send()
            return

        total_nodes = 0
        for i, fp in enumerate(file_paths, start=1):
            try:
                n = await pipeline.ingest_pdf(fp)
                total_nodes += n
                await msg.stream_token(f"\n  - File {i}: {Path(fp).name} ({n} chunks)")
            except (FileNotFoundError, ValueError) as exc:
                await msg.stream_token(f"\n  - File {i}: SKIPPED ({exc})")

    # Refresh the retrieval engine so BM25 and dense indexes are current.
    engine.refresh()

    await msg.stream_token(
        f"\n\nIngestion complete. **{total_nodes}** child nodes indexed. "
        "You can now ask questions about your documents."
    )
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle an incoming user message: retrieve, synthesise, and stream."""
    pipeline, engine = await _ensure_pipeline()
    query = message.content.strip()

    if not query:
        await cl.Message(content="Please enter a question.").send()
        return

    # Step 1: Retrieve and re-rank.
    result: RetrievalResult = await engine.retrieve(query)

    if not result.fragments:
        await cl.Message(
            content="I could not find any relevant passages. "
            "Please upload documents first or rephrase your query."
        ).send()
        return

    # Step 2: Build context and prompt the LLM.
    # --- FIXED COGNITIVE LLM GENERATION & INLINE CITATIONS STREAMING ---
    # --- FIXED COGNITIVE LLM GENERATION & INLINE CITATIONS STREAMING ---
    from llama_index.core.llms import ChatMessage
    from llama_index.llms.groq import Groq
    import os

    # Force a fresh, explicit Groq instance using your key environment string
    local_groq_llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY")
    )

    # 1. Structure your context nodes as standard ChatMessage layers
    context_str = "\n\n".join([f"[Source {i+1}]: {frag.text}" for i, frag in enumerate(result.fragments)])
    
    system_instruction = (
        "You are a precise, well-sourced enterprise research assistant. "
        "Answer the user's question using ONLY the provided context fragments. "
        "Cite sources inline as [Source 1], [Source 2], corresponding directly to the fragment index. "
        "If the context does not contain enough information, say so explicitly. Do not extrapolate."
    )

    messages = [
        ChatMessage(role="system", content=system_instruction),
        ChatMessage(role="user", content=f"Context Sources:\n{context_str}\n\nQuery: {query}")
    ]

    # 2. Instantiate and transmit parent message frame
    msg = cl.Message(content="")
    await msg.send()

    # 3. Stream response tokens cleanly from Groq cloud infrastructure using local_groq_llm
    try:
        response_stream = await local_groq_llm.astream_chat(messages)
        async for chunk in response_stream:
            if chunk.delta:
                await msg.stream_token(chunk.delta)
    except Exception as e:
        logger.error(f"LLM streaming engine failed: {str(e)}")
        msg.content = f"⚠️ Prompt optimization error: {str(e)}"
        await msg.update()
        return
    
    # 4. Build and attach inline citations safely underneath the text block
    text_elements = []
    for i, frag in enumerate(result.fragments):
        source_label = f"Source {i+1}"
        text_elements.append(
            cl.Text(
                name=source_label,
                content=f"Structural Extraction Layer: {frag.chunk_type}\n\nData Passage Content:\n{frag.text}",
                display="inline"
            )
        )
    
    msg.elements = text_elements
    await msg.update()

# ---------------------------------------------------------------------------
# Chainlit entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
