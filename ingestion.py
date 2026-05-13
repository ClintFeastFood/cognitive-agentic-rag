"""
ingestion.py - Cognitive RAG Pipeline: Document Ingestion Engine

Processes uploaded PDFs through LlamaIndex's Hierarchical Node Parser,
splitting documents into 1024-token Parent Nodes mapped to 256-token
Child Nodes with 10% overlap, then seeds vectors into Qdrant or an
in-memory VectorStoreIndex.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
    SimpleDirectoryReader,
    SummaryIndex,
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)
# Force the ingestion engine to use my free cloud Groq cluster
Settings.llm = Groq(
    model="deepseek-r1-distill-llama-70b",
    api_key=os.environ.get("GROQ_API_KEY")
)

#Switch embeddings to compute locally on my computer using HuggingFace
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
COLLECTION_NAME: str = "cognitive_rag"
PARENT_CHUNK_SIZE: int = 1024
CHILD_CHUNK_SIZE: int = 256
CHILD_CHUNK_OVERLAP: int = 26  # ~10% of 256
PERSIST_DIR: str = os.getenv("RAG_PERSIST_DIR", "./storage")
EMBED_DIM: int = 384  # text-embedding-ada-002 dimension


# ---------------------------------------------------------------------------
# LlamaIndex global settings bootstrap
# ---------------------------------------------------------------------------
def configure_settings() -> None:
    """Global LlamaIndex settings are bootstrapped at the module level (Line 36)"""
    Settings.chunk_size = PARENT_CHUNK_SIZE
    Settings.chunk_overlap = CHILD_CHUNK_OVERLAP


# ---------------------------------------------------------------------------
# Qdrant client factory
# ---------------------------------------------------------------------------
def _get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client - local in-memory unless QDRANT_URL is set."""
    url = os.getenv("QDRANT_URL")
    if url:
        api_key = os.getenv("QDRANT_API_KEY")
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(":memory:")


def _ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it does not already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'", COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------
class IngestionPipeline:
    """End-to-end document ingestion: PDF -> hierarchical nodes -> vector store.

    Attributes:
        index: The LlamaIndex VectorStoreIndex populated after ingestion.
        summary_index: A SummaryIndex built from parent (coarse) nodes.
        parent_nodes: List of 1024-token parent nodes for context enrichment.
        child_nodes: List of 256-token leaf nodes used for retrieval.
    """

    def __init__(self) -> None:
        configure_settings()
        self._qdrant_client: QdrantClient = _get_qdrant_client()
        _ensure_collection(self._qdrant_client)

        self._vector_store: QdrantVectorStore = QdrantVectorStore(
            client=self._qdrant_client,
            collection_name=COLLECTION_NAME,
        )
        self.index: Optional[VectorStoreIndex] = None
        self.summary_index: Optional[SummaryIndex] = None
        self.parent_nodes: list[BaseNode] = []
        self.child_nodes: list[BaseNode] = []

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    async def ingest_pdf(self, file_path: str | Path) -> int:
        """Ingest a single PDF into the vector store.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Number of child (leaf) nodes created.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ValueError: If no documents can be parsed from the file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.info("Starting ingestion for '%s'", path.name)

        # Step 1: Read documents from the PDF.
        documents = await asyncio.to_thread(
            SimpleDirectoryReader(input_files=[str(path)]).load_data
        )
        if not documents:
            raise ValueError(f"No documents parsed from '{path.name}'")

        # Step 2: Apply the hierarchical node parser.
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE],
            chunk_overlap=CHILD_CHUNK_OVERLAP,
        )
        nodes: list[BaseNode] = await asyncio.to_thread(parser.get_nodes_from_documents, documents)

        # Step 3: Separate parent (coarse) and child (leaf) nodes.
        leaf_nodes = get_leaf_nodes(nodes)
        self.parent_nodes = [n for n in nodes if n not in leaf_nodes]
        self.child_nodes = leaf_nodes

        # Step 4: Build the VectorStoreIndex from leaf nodes with a
        #         StorageContext that writes to Qdrant.
        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )
        self.index = await asyncio.to_thread(
            VectorStoreIndex,
            nodes=self.child_nodes,
            storage_context=storage_context,
        )

        # Step 5: Build a SummaryIndex from parent nodes for high-level queries.
        self.summary_index = await asyncio.to_thread(
            SummaryIndex, nodes=self.parent_nodes
        )

        # Step 6: Persist metadata to local disk for hot-restart.
        self.index.storage_context.persist(persist_dir=PERSIST_DIR)

        logger.info(
            "Ingestion complete: %d parent nodes, %d child nodes",
            len(self.parent_nodes),
            len(self.child_nodes),
        )
        return len(self.child_nodes)

    async def ingest_multiple(self, file_paths: list[str | Path]) -> int:
        """Ingest multiple PDFs sequentially.

        Args:
            file_paths: List of PDF file paths.

        Returns:
            Total number of child nodes across all documents.
        """
        total = 0
        for fp in file_paths:
            total += await self.ingest_pdf(fp)
        return total

    # -----------------------------------------------------------------------
    # Index reload
    # -----------------------------------------------------------------------
    async def load_existing_index(self) -> Optional[VectorStoreIndex]:
        """Attempt to load a previously persisted index from disk.

        Returns:
            The loaded VectorStoreIndex, or None if no persisted data exists.
        """
        if not Path(PERSIST_DIR).exists():
            return None
        try:
            storage_context = StorageContext.from_defaults(
                vector_store=self._vector_store,
                persist_dir=PERSIST_DIR,
            )
            self.index = await asyncio.to_thread(
                load_index_from_storage, storage_context
            )
            logger.info("Loaded existing index from '%s'", PERSIST_DIR)
            return self.index
        except Exception as exc:
            logger.warning("Failed to load persisted index: %s", exc)
            return None

    def get_retriever(self, similarity_top_k: int = 20):
        """Return a VectorStoreRetriever from the current index.

        Args:
            similarity_top_k: Number of candidate chunks to retrieve.

        Raises:
            RuntimeError: If no index has been built yet.
        """
        if self.index is None:
            raise RuntimeError("No index available. Ingest documents first.")
        return self.index.as_retriever(similarity_top_k=similarity_top_k)

    def get_summary_retriever(self):
        """Return a SummaryIndex retriever for high-level queries.

        Raises:
            RuntimeError: If no summary index has been built yet.
        """
        if self.summary_index is None:
            raise RuntimeError("No summary index available. Ingest documents first.")
        return self.summary_index.as_retriever()
