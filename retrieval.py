"""
retrieval.py - Cognitive RAG Pipeline: Hybrid Retrieval & Re-ranking Engine

Implements an automated Query Router that dispatches to either a SummaryIndex
or a Hybrid Search pipeline (dense cosine + sparse BM25).  Top-20 candidates
are re-ranked via flashrank cross-encoder and filtered to the top 5.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from flashrank import Ranker, RerankRequest
from llama_index.core.schema import NodeWithScore, TextNode
from rank_bm25 import BM25Okapi

from ingestion import IngestionPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANDIDATE_TOP_K: int = 20
RERANK_TOP_N: int = 5
BM25_WINDOW_SIZE: int = 5


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------
class QueryIntent(str, Enum):
    """Classification labels for incoming queries."""
    SUMMARY = "summary"
    GRANULAR = "granular"


_SUMMARY_KEYWORDS: set[str] = {
    "summarize", "summary", "overview", "high-level", "main idea",
    "key points", "tldr", "brief", "overall", "describe", "explain briefly",
    "gist", "recap", "abstract", "conclusion",
}

_GRANULAR_KEYWORDS: set[str] = {
    "specific", "detail", "exact", "number", "data", "statistic", "metric",
    "value", "find", "what is", "how much", "how many", "where", "when",
    "who", "which", "define", "calculate",
}


def classify_intent(query: str) -> QueryIntent:
    """Classify a user query into SUMMARY or GRANULAR intent.

    Uses a simple keyword heuristic.  The word 'summarize' or its synonyms
    force SUMMARY; otherwise GRANULAR is the default since most RAG queries
    target specific facts.

    Args:
        query: The raw user query string.

    Returns:
        A QueryIntent enum value.
    """
    lowered = query.lower()
    tokens = set(re.findall(r"\b\w+\b", lowered))

    summary_score = len(tokens & _SUMMARY_KEYWORDS)
    granular_score = len(tokens & _GRANULAR_KEYWORDS)

    if summary_score > 0 and summary_score >= granular_score:
        return QueryIntent.SUMMARY
    return QueryIntent.GRANULAR


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------
@dataclass
class RankedFragment:
    """A single context fragment after re-ranking, ready for the LLM."""
    text: str
    score: float
    node_id: str
    source_file: str = ""
    page_number: int | None = None
    chunk_type: str = "child"


@dataclass
class RetrievalResult:
    """The complete output of a retrieval pass."""
    intent: QueryIntent
    fragments: list[RankedFragment] = field(default_factory=list)
    query: str = ""


# ---------------------------------------------------------------------------
# BM25 sparse retriever
# ---------------------------------------------------------------------------
class BM25Retriever:
    """Sparse keyword retrieval over ingested child nodes using BM25Okapi."""

    def __init__(self, nodes: list) -> None:
        self._nodes: list = nodes
        self._corpus: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        if nodes:
            self._build_index(nodes)

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r"\b\w+\b", text.lower())

    def _build_index(self, nodes: list) -> None:
        """Build the BM25 index from node texts."""
        self._corpus = [self._tokenize(n.get_content()) for n in nodes]
        self._bm25 = BM25Okapi(self._corpus, k1=1.5, b=0.75)
        logger.info("BM25 index built over %d documents", len(nodes))

    def retrieve(self, query: str, top_k: int = CANDIDATE_TOP_K) -> list[NodeWithScore]:
        """Return top-k BM25 results as NodeWithScore objects.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of NodeWithScore sorted by BM25 score (descending).
        """
        if self._bm25 is None or not self._nodes:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        scored: list[tuple[int, float]] = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results: list[NodeWithScore] = []
        for idx, score in scored:
            if score > 0:
                results.append(
                    NodeWithScore(node=self._nodes[idx], score=float(score))
                )
        return results


# ---------------------------------------------------------------------------
# Cross-encoder re-ranker via flashrank
# ---------------------------------------------------------------------------
class FlashReranker:
    """Re-ranks candidate chunks using a flashrank cross-encoder model."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2") -> None:
        # The Ranker downloads the model on first use and caches it.
        self._ranker = Ranker(model_name=model_name, cache_dir="/tmp/flashrank_cache")

    async def rerank(
        self, query: str, candidates: list[NodeWithScore], top_n: int = RERANK_TOP_N
    ) -> list[RankedFragment]:
        """Re-rank candidate chunks and return the top-N fragments.

        Args:
            query: The original user query.
            candidates: Candidate chunks from the hybrid retrieval stage.
            top_n: Number of top fragments to return.

        Returns:
            Sorted list of RankedFragment objects (descending by score).
        """
        if not candidates:
            return []

        # Build the passage list expected by flashrank.
        passages = [
            {"id": i, "text": c.node.get_content(), "meta": c.node.metadata}
            for i, c in enumerate(candidates)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)

        # Run the potentially CPU-intensive re-ranking in a thread.
        results = await asyncio.to_thread(self._ranker.rerank, rerank_request)

        ranked_fragments: list[RankedFragment] = []
        for item in results[:top_n]:
            meta = item.get("meta", {}) or {}
            ranked_fragments.append(
                RankedFragment(
                    text=item["text"],
                    score=item["score"],
                    node_id=str(item["id"]),
                    source_file=meta.get("file_name", meta.get("filename", "")),
                    page_number=meta.get("page_label", None),
                    chunk_type=meta.get("chunk_type", "child"),
                )
            )

        logger.info("Re-ranked %d candidates to %d fragments", len(candidates), len(ranked_fragments))
        return ranked_fragments


# ---------------------------------------------------------------------------
# Main retrieval orchestrator
# ---------------------------------------------------------------------------
class RetrievalEngine:
    """Orchestrates query routing, hybrid search, and re-ranking.

    Usage::

        engine = RetrievalEngine(pipeline=ingestion_pipeline)
        result = await engine.retrieve("What are the key findings?")
    """

    def __init__(self, pipeline: IngestionPipeline) -> None:
        self._pipeline = pipeline
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._reranker = FlashReranker()
        self._refresh_sparse_index()

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------
    def _refresh_sparse_index(self) -> None:
        """Rebuild the BM25 index from current child nodes."""
        if self._pipeline.child_nodes:
            self._bm25_retriever = BM25Retriever(self._pipeline.child_nodes)
        else:
            self._bm25_retriever = None

    def _merge_hybrid_results(
        self,
        dense: list[NodeWithScore],
        sparse: list[NodeWithScore],
    ) -> list[NodeWithScore]:
        """Merge dense and sparse results with Reciprocal Rank Fusion (RRF).

        RRF formula: score = sum(1 / (k + rank_i)) for each list i.

        Args:
            dense: Results from vector similarity search.
            sparse: Results from BM25 keyword search.

        Returns:
            Deduplicated, RRF-scored list of NodeWithScore.
        """
        k: int = 60  # RRF smoothing constant

        rrf_scores: dict[str, float] = {}
        node_map: dict[str, NodeWithScore] = {}

        for rank, nws in enumerate(dense, start=1):
            nid = nws.node.node_id
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (k + rank)
            node_map[nid] = nws

        for rank, nws in enumerate(sparse, start=1):
            nid = nws.node.node_id
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (k + rank)
            node_map[nid] = nws

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)  # type: ignore[arg-type]
        return [NodeWithScore(node=node_map[nid].node, score=rrf_scores[nid]) for nid in sorted_ids]

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    async def retrieve(self, query: str) -> RetrievalResult:
        """Route, retrieve, and re-rank for a given user query.

        Args:
            query: The natural-language user query.

        Returns:
            A RetrievalResult containing the intent label and ranked fragments.
        """
        intent = classify_intent(query)
        logger.info("Query classified as %s: '%s'", intent.value, query[:80])

        # ---- SUMMARY path: use SummaryIndex retriever ----
        if intent == QueryIntent.SUMMARY:
            try:
                retriever = self._pipeline.get_summary_retriever()
                nodes: list[NodeWithScore] = await asyncio.to_thread(
                    retriever.retrieve, query
                )
                fragments = [
                    RankedFragment(
                        text=n.node.get_content(),
                        score=n.score or 0.0,
                        node_id=n.node.node_id,
                        source_file=n.node.metadata.get("file_name", ""),
                        page_number=n.node.metadata.get("page_label", None),
                        chunk_type="parent",
                    )
                    for n in nodes[:RERANK_TOP_N]
                ]
                return RetrievalResult(intent=intent, fragments=fragments, query=query)
            except RuntimeError:
                logger.warning("Summary retriever unavailable; falling back to granular.")
                intent = QueryIntent.GRANULAR

        # ---- GRANULAR path: hybrid search + re-ranking ----
        dense_results: list[NodeWithScore] = []
        sparse_results: list[NodeWithScore] = []

        # Dense vector search
        try:
            retriever = self._pipeline.get_retriever(similarity_top_k=CANDIDATE_TOP_K)
            dense_results = await asyncio.to_thread(retriever.retrieve, query)
        except RuntimeError:
            logger.error("No vector index available for retrieval.")
            return RetrievalResult(intent=intent, query=query)

        # Sparse BM25 search
        if self._bm25_retriever is not None:
            sparse_results = self._bm25_retriever.retrieve(query, top_k=CANDIDATE_TOP_K)

        # Fuse results
        fused = self._merge_hybrid_results(dense_results, sparse_results)
        top_candidates = fused[:CANDIDATE_TOP_K]

        # Re-rank with cross-encoder
        fragments = await self._reranker.rerank(query, top_candidates, top_n=RERANK_TOP_N)

        return RetrievalResult(intent=intent, fragments=fragments, query=query)

    def refresh(self) -> None:
        """Rebuild internal indexes after a new ingestion pass."""
        self._refresh_sparse_index()
        logger.info("RetrievalEngine refreshed with updated node data.")
