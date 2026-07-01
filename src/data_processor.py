"""
Data Processor
Loads raw JSON datasets and converts them into flat document chunks
suitable for embedding and RAG retrieval.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Document:
    """Represents a single retrievable document chunk."""

    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None,
    ):
        self.content = content
        self.metadata = metadata
        self.doc_id = doc_id or self._generate_id()

    def _generate_id(self) -> str:
        import hashlib
        return hashlib.md5(self.content.encode()).hexdigest()[:12]

    def __repr__(self) -> str:
        role = self.metadata.get("role", "unknown")
        level = self.metadata.get("level", "")
        q_type = self.metadata.get("question_type", "")
        return f"<Document id={self.doc_id} role={role} level={level} type={q_type}>"


class DataProcessor:
    """
    Loads all raw JSON datasets and converts them into Document objects
    ready for embedding into the vector store.
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.documents: List[Document] = []

    def load_all(self) -> List[Document]:
        """Load all datasets and return flattened list of Document objects."""
        self.documents = []

        dataset_loaders = [
            ("interview_questions.json", self._process_interview_questions),
            ("hr_guidelines.json", self._process_hr_guidelines),
            ("job_roles.json", self._process_job_roles),
            ("industry_insights.json", self._process_industry_insights),
        ]

        for filename, loader in dataset_loaders:
            file_path = self.data_dir / filename
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                docs = loader(data)
                self.documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {filename}")
            else:
                logger.warning(f"Dataset file not found: {file_path}")

        logger.info(f"Total documents loaded: {len(self.documents)}")
        return self.documents

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _process_interview_questions(self, data: Dict) -> List[Document]:
        """Convert interview questions JSON into Document objects."""
        docs = []
        categories = data.get("categories", {})

        for role, levels in categories.items():
            for level, question_types in levels.items():
                for q_type, questions in question_types.items():
                    for q in questions:
                        # Build rich text content for embedding
                        content_parts = [
                            f"Role: {role.replace('_', ' ').title()}",
                            f"Level: {level.title()}",
                            f"Question Type: {q_type.title()}",
                            f"Category: {q.get('category', '')}",
                            f"Difficulty: {q.get('difficulty', '')}",
                            f"\nInterview Question: {q['question']}",
                            f"\nModel Answer: {q['model_answer']}",
                        ]

                        if q.get("follow_up"):
                            content_parts.append(
                                f"\nFollow-up Questions: {'; '.join(q['follow_up'])}"
                            )

                        if q.get("improvement_tip"):
                            content_parts.append(
                                f"\nImprovement Tip: {q['improvement_tip']}"
                            )

                        if q.get("keywords"):
                            content_parts.append(
                                f"\nKey Concepts: {', '.join(q['keywords'])}"
                            )

                        content = "\n".join(content_parts)
                        metadata = {
                            "source": "interview_questions",
                            "role": role,
                            "level": level,
                            "question_type": q_type,
                            "question_id": q.get("id", ""),
                            "category": q.get("category", ""),
                            "difficulty": q.get("difficulty", ""),
                        }
                        docs.append(Document(content=content, metadata=metadata, doc_id=q.get("id")))

        return docs

    def _process_hr_guidelines(self, data: Dict) -> List[Document]:
        """Convert HR guidelines into Document objects."""
        docs = []
        hr = data.get("hr_guidelines", {})

        # Process interview stages
        stages = hr.get("interview_process_stages", {})
        for stage_name, stage_data in stages.items():
            content_parts = [f"HR Interview Stage: {stage_name.replace('_', ' ').title()}"]
            for key, val in stage_data.items():
                if isinstance(val, list):
                    content_parts.append(f"\n{key.replace('_', ' ').title()}:")
                    for item in val:
                        content_parts.append(f"  - {item}")
                elif isinstance(val, str):
                    content_parts.append(f"\n{key.replace('_', ' ').title()}: {val}")

            content = "\n".join(content_parts)
            docs.append(Document(
                content=content,
                metadata={"source": "hr_guidelines", "stage": stage_name, "type": "interview_stage"}
            ))

        # Process common HR questions
        hr_questions = hr.get("common_hr_questions", [])
        for q in hr_questions:
            content = (
                f"HR Question: {q['question']}\n"
                f"Answer Framework: {q.get('framework', '')}\n"
                f"Guidance: {q.get('guidance', '')}"
            )
            docs.append(Document(
                content=content,
                metadata={"source": "hr_guidelines", "type": "common_hr_question"}
            ))

        # Process company competency frameworks
        frameworks = hr.get("industry_competency_frameworks", {})
        for company, fw_data in frameworks.items():
            content_parts = [f"Company: {company}", f"Interview Framework: {fw_data.get('framework', '')}"]
            for key, val in fw_data.items():
                if key == "framework":
                    continue
                if isinstance(val, list):
                    content_parts.append(f"\n{key.replace('_', ' ').title()}: {', '.join(val)}")
                elif isinstance(val, str):
                    content_parts.append(f"\n{key.replace('_', ' ').title()}: {val}")

            content = "\n".join(content_parts)
            docs.append(Document(
                content=content,
                metadata={"source": "hr_guidelines", "company": company, "type": "company_framework"}
            ))

        # Process communication tips
        comms = hr.get("body_language_and_communication", {})
        for interview_type, tips in comms.items():
            content = (
                f"Interview Communication Tips - {interview_type.replace('_', ' ').title()}:\n"
                + "\n".join(f"  - {tip}" for tip in tips)
            )
            docs.append(Document(
                content=content,
                metadata={"source": "hr_guidelines", "type": "communication_tips", "interview_type": interview_type}
            ))

        return docs

    def _process_job_roles(self, data: Dict) -> List[Document]:
        """Convert job role definitions into Document objects."""
        docs = []
        roles = data.get("roles", {})

        for role_key, role_data in roles.items():
            role_name = role_data.get("display_name", role_key)

            # Per-level documents
            levels = role_data.get("levels", {})
            for level, level_data in levels.items():
                content_parts = [
                    f"Job Role: {role_name}",
                    f"Level: {level_data.get('label', level)}",
                    f"Years of Experience: {level_data.get('yoe_range', '')}",
                    f"Typical Titles: {', '.join(level_data.get('typical_titles', []))}",
                    f"\nCore Skills Required: {', '.join(level_data.get('core_skills', []))}",
                    f"\nSoft Skills Required: {', '.join(level_data.get('soft_skills', []))}",
                    f"\nInterview Rounds: {', '.join(level_data.get('interview_rounds', []))}",
                    f"\nKey Companies Hiring: {', '.join(level_data.get('key_companies', []))}",
                ]
                content = "\n".join(content_parts)
                docs.append(Document(
                    content=content,
                    metadata={"source": "job_roles", "role": role_key, "level": level, "type": "role_definition"}
                ))

            # Preparation strategy documents
            prep = role_data.get("preparation_strategy", {})
            for level, strategy in prep.items():
                content_parts = [
                    f"Preparation Strategy for {role_name} - {level.title()} Level",
                    f"\nFocus Areas: {', '.join(strategy.get('focus_areas', []))}",
                    f"\nTimeline: {strategy.get('timeline_weeks', '')} weeks",
                ]
                if "daily_plan" in strategy:
                    plan = strategy["daily_plan"]
                    content_parts.append(
                        f"\nWeekday Plan: {plan.get('weekday', '')}"
                        f"\nWeekend Plan: {plan.get('weekend', '')}"
                    )
                content_parts.append(f"\nRecommended Resources: {', '.join(strategy.get('resources', []))}")
                content = "\n".join(content_parts)
                docs.append(Document(
                    content=content,
                    metadata={"source": "job_roles", "role": role_key, "level": level, "type": "preparation_strategy"}
                ))

        return docs

    def _process_industry_insights(self, data: Dict) -> List[Document]:
        """Convert industry insights into Document objects."""
        docs = []

        # Industry trends
        trends = data.get("industry_trends_2024_2025", {})
        for sector, trend_data in trends.items():
            content_parts = [f"Industry Trends - {sector.replace('_', ' ').title()} (2024-2025)"]
            for key, val in trend_data.items():
                label = key.replace("_", " ").title()
                if isinstance(val, list):
                    content_parts.append(f"\n{label}:")
                    for item in val:
                        content_parts.append(f"  - {item}")
                elif isinstance(val, dict):
                    content_parts.append(f"\n{label}:")
                    for sub_key, sub_val in val.items():
                        content_parts.append(f"  {sub_key}: {sub_val}")
            content = "\n".join(content_parts)
            docs.append(Document(
                content=content,
                metadata={"source": "industry_insights", "sector": sector, "type": "trends"}
            ))

        # Company interview styles
        companies = data.get("top_companies_interview_styles", {})
        for company, style_data in companies.items():
            content_parts = [
                f"Interview Process at {company}",
                f"Style: {style_data.get('style', '')}",
                f"Number of Rounds: {style_data.get('rounds', '')}",
                f"Process: {' → '.join(style_data.get('process', []))}",
                f"\nInterview Tips:",
            ]
            for tip in style_data.get("tips", []):
                content_parts.append(f"  - {tip}")
            content_parts.append(f"\nTypical Duration: {style_data.get('duration_days', '')} days")
            content = "\n".join(content_parts)
            docs.append(Document(
                content=content,
                metadata={"source": "industry_insights", "company": company, "type": "company_interview_style"}
            ))

        # Failure reasons
        failure_reasons = (
            data.get("interview_success_metrics", {}).get("common_failure_reasons", [])
        )
        if failure_reasons:
            content = (
                "Common Interview Failure Reasons to Avoid:\n"
                + "\n".join(f"  - {r}" for r in failure_reasons)
            )
            docs.append(Document(
                content=content,
                metadata={"source": "industry_insights", "type": "failure_reasons"}
            ))

        return docs

    def get_documents_by_role_level(
        self,
        role: str,
        level: str,
        question_type: Optional[str] = None,
    ) -> List[Document]:
        """Filter loaded documents by role, level, and optionally question type."""
        if not self.documents:
            self.load_all()

        result = []
        for doc in self.documents:
            if (
                doc.metadata.get("role") == role
                and doc.metadata.get("level") == level
            ):
                if question_type is None or doc.metadata.get("question_type") == question_type:
                    result.append(doc)
        return result

    def save_processed(self, output_path: str = "data/processed/documents.json") -> None:
        """Persist the processed document list to a JSON file."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        serialized = [
            {"doc_id": d.doc_id, "content": d.content, "metadata": d.metadata}
            for d in self.documents
        ]
        with open(output, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
        logger.info(f"Processed documents saved to {output_path}")
