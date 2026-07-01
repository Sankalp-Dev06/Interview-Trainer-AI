"""
RAG Pipeline
Orchestrates data loading, embedding, indexing, retrieval, and generation
for the Interview Trainer Agent.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_loader import ConfigLoader
from .data_processor import DataProcessor
from .embedding_engine import EmbeddingEngine
from .llm_client import GraniteLLMClient
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class InterviewRAGPipeline:
    """
    End-to-end RAG pipeline for interview preparation.

    Usage:
        pipeline = InterviewRAGPipeline()
        pipeline.initialize()
        result = pipeline.generate_prep_plan(name="Alice", role="software_engineer", level="mid")
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load()

        self.data_processor = DataProcessor(
            data_dir=self.config.get("data", {}).get("raw_dir", "data/raw")
        )
        self.embedding_engine = EmbeddingEngine(self.config)
        self.vector_store = VectorStore(
            index_path=self.config.get("rag", {})
            .get("vector_store", {})
            .get("index_path", "data/embeddings/faiss_index")
        )
        self.llm_client = GraniteLLMClient(self.config)

        self._initialized = False
        self._rag_config = self.config.get("rag", {})

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, force_rebuild: bool = False) -> None:
        """
        Set up the pipeline:
        1. Try to load existing FAISS index.
        2. If none found (or force_rebuild=True), build it from raw data.
        """
        if self._initialized and not force_rebuild:
            return

        # Try loading an existing index first
        if not force_rebuild and self.vector_store.load():
            logger.info(
                f"Loaded existing vector store with {self.vector_store.document_count} documents."
            )
        else:
            logger.info("Building vector store from raw datasets...")
            self._build_index()

        self._initialized = True
        logger.info(
            f"Pipeline ready | mode: embedding={self.embedding_engine.mode}, "
            f"llm={self.llm_client.mode}"
        )

    def _build_index(self) -> None:
        """Load data, embed documents, build FAISS index, and persist it."""
        # 1. Load all documents
        documents = self.data_processor.load_all()
        if not documents:
            raise RuntimeError("No documents loaded from data/raw. Check your dataset files.")

        # 2. Save processed docs for inspection
        self.data_processor.save_processed()

        # 3. Embed all documents
        logger.info(f"Embedding {len(documents)} documents...")
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_engine.embed_texts(texts)

        # 4. Build and persist the FAISS index
        self.vector_store.build_index(documents, embeddings)
        self.vector_store.save()
        logger.info("Vector store built and saved.")

    # ------------------------------------------------------------------
    # Core RAG operations
    # ------------------------------------------------------------------

    def retrieve_context(
        self,
        query: str,
        role: Optional[str] = None,
        level: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Retrieve the most relevant documents for a query.
        Returns a formatted context string for prompt injection.
        """
        if not self._initialized:
            self.initialize()

        rag_cfg = self._rag_config.get("vector_store", {})
        k = top_k or rag_cfg.get("top_k", 5)
        threshold = rag_cfg.get("similarity_threshold", 0.0)

        # Build metadata filter if role/level provided
        metadata_filter: Dict[str, str] = {}
        if role:
            metadata_filter["role"] = role

        query_vec = self.embedding_engine.embed_query(query)
        results = self.vector_store.search(
            query_embedding=query_vec,
            top_k=k,
            similarity_threshold=threshold,
            filter_metadata=metadata_filter if metadata_filter else None,
        )

        # Fallback: if role-filtered search returns nothing, search without filter
        if not results and metadata_filter:
            logger.info("Role-filtered search returned no results; retrying without filter.")
            results = self.vector_store.search(
                query_embedding=query_vec,
                top_k=k,
                similarity_threshold=threshold,
            )

        if not results:
            return "No relevant context found in the knowledge base."

        # Format results as a numbered context block
        context_parts = []
        for i, res in enumerate(results, 1):
            source = res.metadata.get("source", "unknown")
            context_parts.append(
                f"[{i}] (Source: {source}, Score: {res.score:.3f})\n{res.content}"
            )

        return "\n\n---\n\n".join(context_parts)

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def generate_prep_plan(
        self,
        name: str,
        role: str,
        level: str,
    ) -> Dict[str, Any]:
        """
        Generate a full personalised interview preparation plan.

        Returns a dict with:
          - name, role, level
          - context_retrieved (raw retrieved docs)
          - preparation_plan (AI-generated text)
          - metadata (mode, doc_count)
        """
        if not self._initialized:
            self.initialize()

        role_norm = role.lower().replace(" ", "_")
        level_norm = level.lower()

        query = (
            f"Interview preparation questions and strategy for a {level_norm} "
            f"{role.replace('_', ' ')} position including technical, behavioral, and HR questions"
        )

        context = self.retrieve_context(query, role=role_norm, level=level_norm, top_k=6)

        plan = self.llm_client.generate_question_set(
            name=name,
            role=role.replace("_", " ").title(),
            level=level_norm.title(),
            context=context,
        )

        return {
            "name": name,
            "role": role,
            "level": level,
            "context_retrieved": context,
            "preparation_plan": plan,
            "metadata": {
                "embedding_mode": self.embedding_engine.mode,
                "llm_mode": self.llm_client.mode,
                "documents_in_store": self.vector_store.document_count,
            },
        }

    def get_question_guidance(
        self,
        name: str,
        role: str,
        level: str,
        question: str,
    ) -> Dict[str, Any]:
        """
        Get detailed guidance for a specific interview question.
        """
        if not self._initialized:
            self.initialize()

        role_norm = role.lower().replace(" ", "_")
        level_norm = level.lower()

        context = self.retrieve_context(
            query=question,
            role=role_norm,
            level=level_norm,
            top_k=4,
        )

        guidance = self.llm_client.generate_answer_guidance(
            name=name,
            role=role.replace("_", " ").title(),
            level=level_norm.title(),
            question=question,
            context=context,
        )

        return {
            "question": question,
            "guidance": guidance,
            "context_retrieved": context,
        }

    def evaluate_candidate_answer(
        self,
        name: str,
        role: str,
        level: str,
        question: str,
        candidate_answer: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a candidate's answer and provide structured feedback.
        """
        if not self._initialized:
            self.initialize()

        role_norm = role.lower().replace(" ", "_")
        level_norm = level.lower()

        context = self.retrieve_context(
            query=f"{question} {candidate_answer}",
            role=role_norm,
            level=level_norm,
            top_k=4,
        )

        feedback = self.llm_client.evaluate_answer(
            name=name,
            role=role.replace("_", " ").title(),
            level=level_norm.title(),
            question=question,
            candidate_answer=candidate_answer,
            context=context,
        )

        return {
            "question": question,
            "candidate_answer": candidate_answer,
            "feedback": feedback,
            "context_retrieved": context,
        }

    def get_role_info(self, role: str, level: str) -> Dict[str, Any]:
        """
        Return structured role and level information from the knowledge base.
        """
        import json

        role_norm = role.lower().replace(" ", "_")
        level_norm = level.lower()

        roles_file = Path(self.config.get("data", {}).get("raw_dir", "data/raw")) / "job_roles.json"
        if roles_file.exists():
            with open(roles_file, "r", encoding="utf-8") as f:
                roles_data = json.load(f)
            role_data = roles_data.get("roles", {}).get(role_norm, {})
            level_data = role_data.get("levels", {}).get(level_norm, {})
            prep_data = role_data.get("preparation_strategy", {}).get(level_norm, {})
            return {
                "role": role_data.get("display_name", role),
                "level_info": level_data,
                "preparation_strategy": prep_data,
            }
        return {}

    # ------------------------------------------------------------------
    # Pipeline status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return current pipeline status."""
        return {
            "initialized": self._initialized,
            "documents_indexed": self.vector_store.document_count,
            "embedding_mode": self.embedding_engine.mode if self._initialized else "not_started",
            "llm_mode": self.llm_client.mode if self._initialized else "not_started",
            "index_path": str(self.vector_store.index_path),
        }
