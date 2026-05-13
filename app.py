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
    model="deepseek-r1-distill-llama-70b",
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
    context_block = _build_context_block(result.fragments)
    full_prompt = f"{SYSTEM_PROMPT}\n\n--- CONTEXT ---\n{context_block}\n\n--- QUESTION ---\n{query}"

    # Step 3: Stream the LLM response token by token.
    response_msg = cl.Message(content="")
    await response_msg.send()

    llm = Settings.llm
    full_response: str = ""

    # Use the LLM's streaming interface for token-by-token output.
    try:
        response_stream = await asyncio.to_thread(
            llm.stream, full_prompt
        )
        for token in response_stream:
            full_response += token
            await response_msg.stream_token(token)
    except Exception as exc:
        logger.error("LLM streaming failed: %s", exc)
        await response_msg.stream_token(
            f"\n\n[Error generating response: {exc}]"
        )

    await response_msg.update()

    # Step 4: Attach expandable citation cards for each context source.
    for i, frag in enumerate(result.fragments, start=1):
        source_label = frag.source_file or "uploaded document"
        page_info = f" | Page: {frag.page_number}" if frag.page_number else ""
        meta_label = (
            f"Score: {frag.score:.4f} | Type: {frag.chunk_type}{page_info}"
        )
        citation = ClText(
            name=f"Source [{i}] - {source_label}",
            content=frag.text,
            display="inline",
        )
        await citation.send()
        # Append a compact reference line under the message.
        await response_msg.stream_token(
            f"\n> **[{i}]** {source_label} — {meta_label}"
        )

    intent_label = "Summary" if result.intent.value == "summary" else "Granular (Hybrid)"
    await response_msg.stream_token(
        f"\n\n_Retrieval path: {intent_label} | Fragments: {len(result.fragments)}_"
    )
    await response_msg.update()


# ---------------------------------------------------------------------------
# Chainlit entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
