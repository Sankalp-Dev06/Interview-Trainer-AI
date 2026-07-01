"""
Unit Tests for Interview Trainer Agent
Tests core components: data processor, vector store, and RAG pipeline.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processor import DataProcessor, Document
from src.vector_store import VectorStore


class TestDocument(unittest.TestCase):
    """Tests for the Document data class."""

    def test_document_creation(self):
        doc = Document(content="Test content", metadata={"role": "software_engineer"})
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.metadata["role"], "software_engineer")
        self.assertIsNotNone(doc.doc_id)

    def test_document_id_generation(self):
        doc1 = Document(content="Same content", metadata={})
        doc2 = Document(content="Same content", metadata={})
        self.assertEqual(doc1.doc_id, doc2.doc_id)  # same content → same hash

    def test_document_explicit_id(self):
        doc = Document(content="Content", metadata={}, doc_id="SE-001")
        self.assertEqual(doc.doc_id, "SE-001")

    def test_document_repr(self):
        doc = Document(content="c", metadata={"role": "devops_engineer", "level": "mid", "question_type": "technical"})
        r = repr(doc)
        self.assertIn("devops_engineer", r)
        self.assertIn("mid", r)


class TestDataProcessor(unittest.TestCase):
    """Tests for the DataProcessor class."""

    def setUp(self):
        # Minimal valid interview questions fixture
        self.sample_questions = {
            "version": "1.0",
            "categories": {
                "software_engineer": {
                    "entry": {
                        "technical": [
                            {
                                "id": "SE-E-T-001",
                                "question": "What is a stack?",
                                "category": "Data Structures",
                                "difficulty": "easy",
                                "model_answer": "LIFO structure.",
                                "follow_up": ["Can you implement it?"],
                                "keywords": ["LIFO", "stack"],
                                "improvement_tip": "Give real-world examples."
                            }
                        ]
                    }
                }
            }
        }
        self.sample_hr = {
            "hr_guidelines": {
                "interview_process_stages": {
                    "phone_screen": {
                        "duration": "30 minutes",
                        "what_to_expect": ["Walk me through your resume"]
                    }
                },
                "common_hr_questions": [
                    {
                        "question": "Tell me about yourself.",
                        "framework": "Present-Past-Future",
                        "guidance": "Keep it under 2 minutes."
                    }
                ],
                "industry_competency_frameworks": {
                    "IBM": {
                        "framework": "IBM Values",
                        "values": ["Client success", "Innovation"],
                        "preparation_tip": "Research IBM's AI strategy."
                    }
                },
                "body_language_and_communication": {
                    "virtual_interview": ["Test your camera"]
                }
            }
        }
        self.sample_roles = {
            "roles": {
                "software_engineer": {
                    "display_name": "Software Engineer",
                    "levels": {
                        "entry": {
                            "label": "Entry",
                            "yoe_range": "0-2",
                            "typical_titles": ["Junior Dev"],
                            "core_skills": ["Python"],
                            "soft_skills": ["Teamwork"],
                            "interview_rounds": ["Coding"],
                            "key_companies": ["IBM"]
                        }
                    },
                    "preparation_strategy": {
                        "entry": {
                            "focus_areas": ["DSA"],
                            "timeline_weeks": 6,
                            "resources": ["LeetCode"],
                            "daily_plan": {"weekday": "2 problems", "weekend": "mock interview"}
                        }
                    }
                }
            }
        }
        self.sample_insights = {
            "industry_trends_2024_2025": {
                "software_engineering": {
                    "hot_skills": ["AI"],
                    "hiring_trends": ["Remote interviews common"]
                }
            },
            "top_companies_interview_styles": {
                "IBM": {
                    "style": "Competency-based",
                    "rounds": 3,
                    "process": ["HR", "Technical", "Final"],
                    "tips": ["Know IBM values"],
                    "duration_days": "2-4 weeks"
                }
            },
            "interview_success_metrics": {
                "common_failure_reasons": ["Jumping to code without clarifying"]
            }
        }

    def _write_fixtures(self, tmp_dir: Path):
        fixtures = {
            "interview_questions.json": self.sample_questions,
            "hr_guidelines.json": self.sample_hr,
            "job_roles.json": self.sample_roles,
            "industry_insights.json": self.sample_insights,
        }
        for fname, data in fixtures.items():
            (tmp_dir / fname).write_text(json.dumps(data))

    def test_load_all_returns_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_fixtures(tmp_path)
            processor = DataProcessor(data_dir=str(tmp_path))
            docs = processor.load_all()
            self.assertGreater(len(docs), 0)
            for d in docs:
                self.assertIsInstance(d, Document)
                self.assertTrue(d.content)
                self.assertIsInstance(d.metadata, dict)

    def test_documents_have_required_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_fixtures(tmp_path)
            processor = DataProcessor(data_dir=str(tmp_path))
            processor.load_all()
            interview_docs = [d for d in processor.documents if d.metadata.get("source") == "interview_questions"]
            self.assertGreater(len(interview_docs), 0)
            for d in interview_docs:
                self.assertIn("role", d.metadata)
                self.assertIn("level", d.metadata)
                self.assertIn("question_type", d.metadata)

    def test_filter_by_role_and_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_fixtures(tmp_path)
            processor = DataProcessor(data_dir=str(tmp_path))
            processor.load_all()
            result = processor.get_documents_by_role_level("software_engineer", "entry")
            self.assertGreater(len(result), 0)
            for d in result:
                self.assertEqual(d.metadata.get("role"), "software_engineer")
                self.assertEqual(d.metadata.get("level"), "entry")


class TestVectorStore(unittest.TestCase):
    """Tests for VectorStore (requires faiss-cpu)."""

    def test_vector_store_build_and_search(self):
        try:
            import faiss
            import numpy as np
        except ImportError:
            self.skipTest("faiss-cpu not installed")

        docs = [
            Document("machine learning interview question", {"role": "data_scientist", "level": "entry"}),
            Document("system design distributed systems", {"role": "software_engineer", "level": "mid"}),
            Document("behavioral STAR leadership example", {"role": "software_engineer", "level": "mid"}),
        ]
        embeddings = np.random.rand(3, 128).astype(np.float32)
        # L2-normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        with tempfile.TemporaryDirectory() as tmp:
            store = VectorStore(index_path=str(Path(tmp) / "test_index"))
            store.build_index(docs, embeddings)

            self.assertTrue(store.is_ready)
            self.assertEqual(store.document_count, 3)

            query = np.random.rand(1, 128).astype(np.float32)
            query = query / np.linalg.norm(query)
            results = store.search(query, top_k=2)
            self.assertEqual(len(results), 2)
            self.assertGreaterEqual(results[0].score, results[1].score)

    def test_vector_store_save_and_load(self):
        try:
            import faiss
            import numpy as np
        except ImportError:
            self.skipTest("faiss-cpu not installed")

        docs = [Document("test content", {"source": "test"})]
        embeddings = np.random.rand(1, 64).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            store = VectorStore(index_path=str(Path(tmp) / "idx"))
            store.build_index(docs, embeddings)
            store.save()

            store2 = VectorStore(index_path=str(Path(tmp) / "idx"))
            loaded = store2.load()
            self.assertTrue(loaded)
            self.assertEqual(store2.document_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
