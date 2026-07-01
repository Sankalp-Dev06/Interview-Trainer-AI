"""
Embedding Engine
Generates and manages vector embeddings using IBM Slate model via watsonx.ai.
Falls back to SentenceTransformers when IBM credentials are not available (dev mode).
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Generates vector embeddings using the IBM Slate model (watsonx.ai).
    Automatically falls back to a local SentenceTransformer model if
    IBM credentials are not configured (useful for local development).
    """

    IBM_SLATE_MODEL = "ibm/slate-125m-english-rtrvr"
    FALLBACK_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, config: dict):
        self.config = config
        self._model = None
        self._mode = None   # "ibm" or "local"
        self._embed_config = config.get("watsonx", {}).get("embedding", {})
        self._ibm_config = config.get("watsonx", {})

    def _initialize(self) -> None:
        """Initialize the embedding model — IBM Slate or local fallback."""
        if self._model is not None:
            return

        api_key = self._ibm_config.get("api_key", "")
        project_id = self._ibm_config.get("project_id", "")
        url = self._ibm_config.get("url", "https://us-south.ml.cloud.ibm.com")

        # Check if real IBM credentials are present
        is_valid_ibm = (
            api_key
            and not api_key.startswith("MISSING_")
            and project_id
            and not project_id.startswith("MISSING_")
        )

        if is_valid_ibm:
            self._init_ibm_slate(api_key, project_id, url)
        else:
            logger.warning(
                "IBM credentials not found or incomplete. "
                "Falling back to local SentenceTransformer model for embeddings."
            )
            self._init_local_model()

    def _init_ibm_slate(self, api_key: str, project_id: str, url: str) -> None:
        """Initialize IBM Slate embedding model via watsonx.ai SDK."""
        try:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import Embeddings
            from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

            credentials = Credentials(api_key=api_key, url=url)
            model_id = self._embed_config.get("model_id", self.IBM_SLATE_MODEL)

            self._model = Embeddings(
                model_id=model_id,
                credentials=credentials,
                project_id=project_id,
            )
            self._mode = "ibm"
            logger.info(f"IBM Slate embedding model initialized: {model_id}")
        except ImportError:
            logger.warning("ibm-watsonx-ai package not installed. Falling back to local model.")
            self._init_local_model()
        except Exception as e:
            logger.error(f"Failed to initialize IBM Slate model: {e}. Falling back to local.")
            self._init_local_model()

    def _init_local_model(self) -> None:
        """Initialize a local SentenceTransformer as fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.FALLBACK_MODEL)
            self._mode = "local"
            logger.info(f"Local SentenceTransformer model loaded: {self.FALLBACK_MODEL}")
        except ImportError:
            raise ImportError(
                "Neither ibm-watsonx-ai nor sentence-transformers is installed. "
                "Please install at least one: pip install sentence-transformers"
            )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of text strings.
        Returns a numpy array of shape (len(texts), embedding_dim).
        """
        self._initialize()
        batch_size = self._embed_config.get("batch_size", 32)

        if self._mode == "ibm":
            return self._embed_with_ibm(texts, batch_size)
        else:
            return self._embed_with_local(texts, batch_size)

    def _embed_with_ibm(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Batch-embed texts using IBM Slate via watsonx.ai."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            response = self._model.embed_documents(texts=batch)
            # SDK shape varies by version:
            #   ≥1.1  → list of vectors  [[0.1, ...], ...]
            #   <1.1  → {"results": [{"embedding": [...]}, ...]}
            if isinstance(response, list):
                batch_vecs = [
                    item["embedding"] if isinstance(item, dict) else item
                    for item in response
                ]
            else:
                batch_vecs = [item["embedding"] for item in response["results"]]
            all_embeddings.extend(batch_vecs)
            logger.debug(f"Embedded batch {i//batch_size + 1} ({len(batch)} texts)")

        embeddings = np.array(all_embeddings, dtype=np.float32)
        if self._embed_config.get("normalize_embeddings", True):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms
        return embeddings

    def _embed_with_local(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Embed texts using local SentenceTransformer."""
        normalize = self._embed_config.get("normalize_embeddings", True)
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 50,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1, embedding_dim)."""
        return self.embed_texts([query])

    @property
    def mode(self) -> str:
        self._initialize()
        return self._mode
