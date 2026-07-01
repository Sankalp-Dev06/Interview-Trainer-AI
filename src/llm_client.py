"""
IBM Granite LLM Client
Provides a unified interface to IBM Granite models via watsonx.ai.
Falls back to a local response generator for development/demo use.
"""

import logging
import textwrap
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Prompt Templates
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Interview Trainer AI powered by IBM Granite.
    Your mission is to help job candidates prepare thoroughly and confidently for interviews.

    Guidelines:
    - Provide specific, actionable advice tailored to the candidate's role and experience level.
    - Use STAR (Situation, Task, Action, Result) framework for behavioral questions.
    - Include model answers that are detailed yet concise.
    - Highlight improvement tips and common mistakes to avoid.
    - Reference industry-specific expectations and company culture when relevant.
    - Be encouraging and professional in tone.
""")

QUESTION_SET_TEMPLATE = """\
{system_prompt}

Candidate Profile:
- Name: {name}
- Job Role: {role}
- Experience Level: {level}

Retrieved Context from Interview Knowledge Base:
{context}

Task: Generate a comprehensive, personalized interview preparation set for this candidate.
Structure your response with:
1. PERSONALIZED QUESTION SET (8-10 questions mix of technical, behavioral, and HR)
2. MODEL ANSWERS for each question
3. IMPROVEMENT TIPS specific to this role and level
4. PREPARATION STRATEGY with timeline and resources
5. KEY AREAS TO FOCUS ON based on the role

Response:"""

SINGLE_QUESTION_TEMPLATE = """\
{system_prompt}

Candidate Profile:
- Name: {name}
- Job Role: {role}
- Experience Level: {level}

Retrieved Context:
{context}

Question Asked: {question}

Provide:
1. A detailed model answer
2. Key points the interviewer is evaluating
3. Common mistakes to avoid
4. A follow-up question to expect

Response:"""

ANSWER_EVALUATION_TEMPLATE = """\
{system_prompt}

Candidate Profile:
- Name: {name}
- Job Role: {role}
- Experience Level: {level}

Interview Question: {question}

Candidate's Answer: {candidate_answer}

Retrieved Context for Reference:
{context}

Evaluate the candidate's answer and provide:
1. SCORE (1-10) with justification
2. STRENGTHS of the answer
3. AREAS FOR IMPROVEMENT
4. AN IMPROVED VERSION of the answer
5. TIPS for better delivery

Response:"""


# ------------------------------------------------------------------ #
# LLM Client
# ------------------------------------------------------------------ #

class GraniteLLMClient:
    """
    Wraps IBM Granite (watsonx.ai) for generation tasks.
    Automatically falls back to a rule-based stub if IBM credentials
    are absent, so the notebook runs end-to-end without credentials.
    """

    def __init__(self, config: dict):
        self.config = config
        self._client = None
        self._mode: Optional[str] = None  # "ibm" or "stub"
        self._llm_config = config.get("watsonx", {}).get("llm", {})
        self._ibm_config = config.get("watsonx", {})

    def _initialize(self) -> None:
        if self._client is not None:
            return

        api_key = self._ibm_config.get("api_key", "")
        project_id = self._ibm_config.get("project_id", "")
        url = self._ibm_config.get("url", "https://us-south.ml.cloud.ibm.com")

        is_valid_ibm = (
            api_key
            and not api_key.startswith("MISSING_")
            and project_id
            and not project_id.startswith("MISSING_")
        )

        if is_valid_ibm:
            self._init_ibm(api_key, project_id, url)
        else:
            logger.warning(
                "IBM watsonx.ai credentials not found. "
                "Running in STUB mode — responses will use retrieved context directly."
            )
            self._mode = "stub"

    def _init_ibm(self, api_key: str, project_id: str, url: str) -> None:
        """Initialize IBM Granite via watsonx.ai SDK."""
        try:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

            credentials = Credentials(api_key=api_key, url=url)
            model_id = self._llm_config.get("model_id", "ibm/granite-13b-chat-v2")
            params = self._llm_config.get("parameters", {})

            generate_params = {
                GenParams.MAX_NEW_TOKENS: params.get("max_new_tokens", 1024),
                GenParams.MIN_NEW_TOKENS: params.get("min_new_tokens", 50),
                GenParams.TEMPERATURE: params.get("temperature", 0.7),
                GenParams.TOP_P: params.get("top_p", 0.9),
                GenParams.TOP_K: params.get("top_k", 50),
                GenParams.REPETITION_PENALTY: params.get("repetition_penalty", 1.1),
                GenParams.DECODING_METHOD: params.get("decoding_method", "sample"),
            }

            self._client = ModelInference(
                model_id=model_id,
                credentials=credentials,
                project_id=project_id,
                params=generate_params,
            )
            self._mode = "ibm"
            logger.info(f"IBM Granite model initialized: {model_id}")
        except ImportError:
            logger.warning("ibm-watsonx-ai not installed. Running in STUB mode.")
            self._mode = "stub"
        except Exception as e:
            logger.error(f"IBM Granite initialization failed: {e}. Running in STUB mode.")
            self._mode = "stub"

    def generate(self, prompt: str) -> str:
        """Generate a response from IBM Granite or the stub fallback."""
        self._initialize()

        if self._mode == "ibm":
            return self._generate_ibm(prompt)
        else:
            return self._generate_stub(prompt)

    def _generate_ibm(self, prompt: str) -> str:
        """Call IBM Granite via watsonx.ai SDK."""
        try:
            response = self._client.generate_text(prompt=prompt)
            return response
        except Exception as e:
            logger.error(f"IBM Granite generation error: {e}")
            return self._generate_stub(prompt)

    def _generate_stub(self, prompt: str) -> str:
        """
        Stub mode: returns a formatted response built from the context
        embedded in the prompt. Used when IBM credentials are absent.
        """
        # Extract context section from the prompt for display
        if "Retrieved Context" in prompt and "Task:" in prompt:
            start = prompt.find("Retrieved Context")
            end = prompt.find("Task:", start)
            context_section = prompt[start:end].strip()
            header = (
                "**[STUB MODE — Connect IBM watsonx.ai for full AI responses]**\n\n"
                "Based on the retrieved knowledge base context:\n\n"
            )
            return header + context_section + "\n\n_Configure your IBM Cloud API key and watsonx.ai Project ID to receive full AI-generated answers from IBM Granite._"
        return (
            "**[STUB MODE]** IBM watsonx.ai credentials not configured.\n"
            "Please set IBM_CLOUD_API_KEY and WATSONX_PROJECT_ID environment variables "
            "to enable full IBM Granite-powered responses."
        )

    def generate_question_set(
        self,
        name: str,
        role: str,
        level: str,
        context: str,
    ) -> str:
        """Generate a personalised interview question set."""
        self._initialize()
        prompt = QUESTION_SET_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            name=name,
            role=role,
            level=level,
            context=context,
        )
        return self.generate(prompt)

    def generate_answer_guidance(
        self,
        name: str,
        role: str,
        level: str,
        question: str,
        context: str,
    ) -> str:
        """Generate detailed guidance for a specific interview question."""
        self._initialize()
        prompt = SINGLE_QUESTION_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            name=name,
            role=role,
            level=level,
            question=question,
            context=context,
        )
        return self.generate(prompt)

    def evaluate_answer(
        self,
        name: str,
        role: str,
        level: str,
        question: str,
        candidate_answer: str,
        context: str,
    ) -> str:
        """Evaluate a candidate's answer and provide structured feedback."""
        self._initialize()
        prompt = ANSWER_EVALUATION_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            name=name,
            role=role,
            level=level,
            question=question,
            candidate_answer=candidate_answer,
            context=context,
        )
        return self.generate(prompt)

    @property
    def mode(self) -> str:
        self._initialize()
        return self._mode
