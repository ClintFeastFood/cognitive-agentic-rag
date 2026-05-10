# Cognitive RAG Pipeline

A production-grade Retrieval-Augmented Generation system built entirely in Python. Combines LlamaIndex hierarchical document parsing, hybrid dense+sparse retrieval with Reciprocal Rank Fusion, and cross-encoder re-ranking via flashrank, served through a Chainlit chat interface with FastAPI health endpoints.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Chainlit UI (app.py)               │
│  PDF Upload → Async Ingestion → Streaming Response   │
│  Expandable Citations (cl.Text) per Source Fragment  │
└──────────────┬──────────────────────┬────────────────┘
               │                      │
       ┌───────▼────────┐    ┌────────▼────────┐
       │  Ingestion     │    │  Retrieval      │
       │  Pipeline      │    │  Engine         │
       │  (ingestion.py)│    │  (retrieval.py) │
       └───────┬────────┘    └──┬─────────┬────┘
               │                │         │
   ┌───────────▼──────┐  ┌─────▼───┐ ┌───▼──────┐
   │ Hierarchical     │  │ Dense   │ │  BM25    │
   │ Node Parser      │  │ Vector  │ │  Sparse  │
   │ 1024t Parent     │  │ Search  │ │  Search  │
   │ 256t Child       │  └────┬────┘ └────┬─────┘
   │ 10% overlap      │       │           │
   └───────────┬──────┘  ┌────▼───────────▼────┐
               │         │  Reciprocal Rank    │
   ┌───────────▼──────┐  │  Fusion (RRF)       │
   │ Qdrant Vector    │  └────────┬────────────┘
   │ Store + Summary  │           │
   │ Index            │  ┌───────▼─────────┐
   └──────────────────┘  │ FlashRerank     │
                         │ Cross-Encoder   │
                         │ Top-20 → Top-5  │
                         └─────────────────┘
```

## Query Routing

| Intent | Detection | Retrieval Path |
|--------|-----------|----------------|
| **Summary** | Keywords: summarize, overview, key points, gist, recap | SummaryIndex (parent nodes) |
| **Granular** | Keywords: specific, detail, number, data, find, define | Hybrid dense + BM25 with RRF fusion and cross-encoder re-ranking |

## Key Design Decisions

1. **Hierarchical Node Parsing**: 1024-token parent nodes preserve broad context for summary queries; 256-token child nodes with 10% overlap provide precision for factual retrieval. The parent-child mapping enables future "auto-merge" retrieval where child hits can pull in their parent context.

2. **Reciprocal Rank Fusion (RRF)**: Merges dense (cosine similarity) and sparse (BM25) result sets without requiring score normalization. RRF's reciprocal rank formula `1/(k + rank)` naturally handles differing score scales.

3. **FlashRerank Cross-Encoder**: The ms-marco-MiniLM-L-12-v2 model provides a fast, lightweight re-ranking pass that dramatically improves precision over raw retrieval. Candidates are reduced from 20 to 5 before LLM synthesis.

4. **Async Throughout**: All I/O-bound operations (file reading, embedding, LLM calls, re-ranking) use `asyncio.to_thread` to avoid blocking the Chainlit event loop.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Required for embeddings and LLM | — |
| `QDRANT_URL` | Qdrant server URL (empty = in-memory) | (in-memory) |
| `QDRANT_API_KEY` | Qdrant API key for cloud instances | — |
| `RAG_PERSIST_DIR` | Local directory for index persistence | `./storage` |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI key
export OPENAI_API_KEY=sk-...

# 3. Launch the Chainlit application
chainlit run app.py
```

The UI will be available at `http://localhost:8000`. FastAPI health endpoints are mounted alongside at `/health` and `/status`.

## File Reference

| File | Responsibility |
|------|---------------|
| `app.py` | Chainlit chat UI, PDF upload handler, streaming LLM responses, citation cards |
| `ingestion.py` | Hierarchical node parsing, Qdrant vector storage, SummaryIndex construction |
| `retrieval.py` | Query intent classification, BM25 sparse retrieval, RRF fusion, FlashRerank pipeline |
| `requirements.txt` | Pinned production dependencies |

## Ingestion Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Parent chunk size | 1024 tokens | Broad context for summaries and auto-merge |
| Child chunk size | 256 tokens | Fine-grained retrieval for factual queries |
| Child chunk overlap | 26 tokens (~10%) | Prevents boundary-spanning facts from being split |
| Embedding model | text-embedding-ada-002 | 1536-dim, cost-effective, high quality |
| LLM | gpt-4o-mini | Fast, inexpensive, sufficient for grounded QA |
