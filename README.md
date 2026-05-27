# Enterprise Cognitive RAG Platform

An enterprise-grade, highly performant document intelligence pipeline built to eliminate hallucination vectors during deep analysis of complex corporate datasets. This system ingests unstructured data schemas, executes multi-stage hybrid retrieval loops, and streams grounded text generations with microsecond latency.

## 🛠️ Production Architecture Stack

- Orchestration Engine: Python, LlamaIndex, Pydantic, Asyncio
- Vector Infrastructure: Qdrant Vector Database (In-Memory Clustering Matrix)
- LLM Acceleration: Groq Cloud LPU Framework (llama-3.3-70b-versatile)
- Information Retrieval: Hybrid Search (Dense Vector + BM25 Lexical Keyword matching)
- Embedding Model: BAAI/bge-small-en-v1.5 (384-Dimensional Tensor Space)
- Context Re-ranking: FlashRank Cross-Encoder (ms-marco-MiniLM-L-6-v2)
- Streaming UI Dashboard: Chainlit (Python-Native WebSocket Wrapper)

---

## 🎯 Core Cognitive Features

### 📐 1. Adaptive Layout-Aware Chunking Strategy

Unlike standard uniform text splitters that cause severe context fragmentation, this pipeline dynamically reads document structural primitives before committing nodes to the database:

- **standard_narrative:** Content blocks are assigned lar512-token windowsws\*\* with rolling overlaps to preserve semantic flow and continuous logic.
- **tabular_matrix:** Complex data tables and rows are automatically routed int256-token windowsindows0 overlapverlap\*\* to preserve table column integrity.

### 🔄 2. Decoupled Multi-Stage Retrieval & Re-ranking

Protects the LLM context window from irrelevant filler content through a tiered search pipeliBroad Hybrid Fetch:Fetch:** Pulls the top 15 candidate text blocks using a combination of dense semantic distance scores and sparse keyword matchiCross-Encoder Re-ranking:nking:** Feeds candidates through a secondary neural cross-encoder layer to calculate a deep query-to-passage attention weight layoFocused Window Extraction:ction:\*\* Drops low-scoring fragments and extracts only the top 4 highly relevant data frames to feed the generation prompt.

### 🛡️ 3. Zero-Trust Hallucination Guardrail

An integrated prompt-enclosed constraint layer acts as a zero-trust gatekeeper. If the ingested context matrices lack concrete facts to prove the user's question, the engine triggers a clean, deterministic fallback notification string instead of manufacturing responses.

### 🔗 4. Grounded Interface Citations

Every response token streams directly into the web dashboard while dynamically compiling interactive inline source citation cards. Users can expand cards to instantly view confidence tracking metrics and structural extraction source attributes (tabular_matrix vs. standard_narrative).

---

## 🛠️ Local Installation & Deployment

### 1. Initialize Your Environment

Clone the repository workspace and spin up a clean Python virtual environment:
git clone https://github.com
cd cognitive-agentic-rag/project
python -m venv venv
./venv/Scripts/Activate.ps1 # On Windows PowerShell

### 2. Install Project Dependencies

Deploy all required runtime footings and compiled binary wheels:
pip install -r requirements.txt

### 3. Configure Infrastructure Tokens

Inject your free high-speed Groq API key into your local terminal memory space:
\$env:GROQ_API_KEY="your-actual-gsk-token-key-here"

### 4. Boot the Production Server

Launch the application wrapper using hot-reload tracing parameters:
chainlit run app.py

---

## 🎙️ Sample Architectural Interview Defense

> _"I didn't design a fragile, naive RAG wrapper that blindly throws prompts at a vector store. I engineered an Adaptive Layout-Aware Ingestion Pipeline paired with a Multi-Stage Retrieval Matrix. By isolating tabular data rows into distinct, zero-overlap token boundaries and passing filtered candidate arrays through a secondary FlashRank Cross-Encoder re-ranker, the system filters out structural noise before it reaches the Llama-3.3 layer.This enforces absolute source grounding and ensures near-zero hallucination rates during complex enterprise data evaluations."_
