"""
Vector Store
FAISS-based local vector store for document retrieval.
Stores document embeddings and supports similarity search.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a single retrieval result."""

    def __init__(self, doc_id: str, content: str, metadata: Dict[str, Any], score: float):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata
        self.score = score

    def __repr__(self) -> str:
        return f"<SearchResult id={self.doc_id} score={self.score:.4f}>"


class VectorStore:
    """
    FAISS-backed vector store.
    Builds or loads an index of document embeddings and answers similarity queries.
    """

    INDEX_FILE = "faiss.index"
    META_FILE = "metadata.pkl"

    def __init__(self, index_path: str = "data/embeddings/faiss_index"):
        self.index_path = Path(index_path)
        self._faiss_index = None
        self._documents: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def build_index(
        self,
        documents: List[Any],  # List[Document] from data_processor
        embeddings: np.ndarray,
    ) -> None:
        """
        Build a flat FAISS index from documents and their embeddings.

        Args:
            documents: list of Document objects (must have .doc_id, .content, .metadata)
            embeddings: float32 numpy array of shape (N, dim)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")

        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Document count ({len(documents)}) must match "
                f"embedding rows ({embeddings.shape[0]})"
            )

        dim = embeddings.shape[1]
        # Inner-product index on L2-normalised vectors == cosine similarity
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(embeddings)

        self._documents = [
            {"doc_id": d.doc_id, "content": d.content, "metadata": d.metadata}
            for d in documents
        ]
        logger.info(
            f"FAISS index built: {self._faiss_index.ntotal} vectors, dim={dim}"
        )

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")

        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_index, str(self.index_path / self.INDEX_FILE))

        with open(self.index_path / self.META_FILE, "wb") as f:
            pickle.dump(self._documents, f)

        logger.info(f"Vector store saved to {self.index_path}")

    def load(self) -> bool:
        """
        Load a previously saved FAISS index from disk.
        Returns True if successful, False otherwise.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")

        index_file = self.index_path / self.INDEX_FILE
        meta_file = self.index_path / self.META_FILE

        if not index_file.exists() or not meta_file.exists():
            logger.info("No existing vector store found. Will build a new one.")
            return False

        self._faiss_index = faiss.read_index(str(index_file))
        with open(meta_file, "rb") as f:
            self._documents = pickle.load(f)

        logger.info(
            f"Vector store loaded: {self._faiss_index.ntotal} vectors from {self.index_path}"
        )
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        """
        Retrieve the top-k most similar documents for a query embedding.

        Args:
            query_embedding: shape (1, dim) float32 array
            top_k: number of results to return
            similarity_threshold: minimum cosine similarity score to include
            filter_metadata: optional dict of metadata key-value filters

        Returns:
            List of SearchResult objects sorted by descending score
        """
        if self._faiss_index is None:
            raise RuntimeError("Vector store not initialised. Call build_index() or load() first.")

        # Retrieve more candidates if metadata filtering is applied
        k_candidates = top_k * 3 if filter_metadata else top_k
        k_candidates = min(k_candidates, self._faiss_index.ntotal)

        scores, indices = self._faiss_index.search(query_embedding, k_candidates)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < similarity_threshold:
                continue

            doc = self._documents[idx]

            # Apply metadata filters
            if filter_metadata:
                match = all(
                    doc["metadata"].get(k) == v for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            results.append(
                SearchResult(
                    doc_id=doc["doc_id"],
                    content=doc["content"],
                    metadata=doc["metadata"],
                    score=float(score),
                )
            )

            if len(results) >= top_k:
                break

        return results

    @property
    def is_ready(self) -> bool:
        return self._faiss_index is not None and len(self._documents) > 0

    @property
    def document_count(self) -> int:
        return len(self._documents)
