# 🎯 Interview Trainer Agent
### Powered by IBM Granite (watsonx.ai) + RAG | Problem Statement No. 22

An AI-powered Interview Trainer built on **IBM Cloud** using **IBM Granite 4.0** foundation models and a **RAG (Retrieval-Augmented Generation)** pipeline. It prepares candidates for job interviews by generating tailored question sets, model answers, improvement tips, and personalised preparation strategies — all served through a **chat-first Streamlit web app**.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **💬 Chat Coach** | Conversational interface — ask anything, get instant AI coaching with chat bubbles |
| **📋 Personalised Prep Plan** | Full question set + model answers + study timeline tailored to your role and level |
| **💡 Question Guidance** | Deep-dive on any interview question: model answer, what's being tested, follow-ups |
| **🏆 Answer Evaluator** | Score (1–10) + strengths + improvement tips + improved answer version |
| **🗺️ Role Explorer** | Browse skill requirements, interview rounds, key companies, and prep strategies |
| **Multi-Role Support** | Software Engineer · Data Scientist · Product Manager · DevOps Engineer |
| **Multi-Level Support** | Entry (0–2 yr) · Mid (3–6 yr) · Senior (7+ yr) |
| **STAR Framework** | All behavioural answers coached with Situation-Task-Action-Result structure |
| **Graceful Fallback** | Runs in stub/local mode without IBM credentials — full pipeline still functional |

---

## 🏗️ Architecture

```
User (Streamlit Web App)
        |
        v
  ┌───────────────────────────────────────────────────────────┐
  │                    RAG Pipeline                           │
  │                                                           │
  │  1. Embed query  ──►  IBM Slate                           │
  │                        ibm/slate-125m-english-rtrvr-v2    │
  │  2. Retrieve     ──►  FAISS Vector Store                  │
  │                        (Interview Q&A, HR Guidelines,     │
  │                         Job Roles, Industry Insights)     │
  │  3. Augment prompt with retrieved context                 │
  │  4. Generate     ──►  IBM Granite 4.0                     │
  │                        ibm-granite/granite-4.0-h-small    │
  └───────────────────────────────────────────────────────────┘
        |
        v
  Chat Response · Prep Plan · Scored Feedback · Role Overview
```

---

## 📁 Project Structure

```
interview_trainer_agent/
│
├── streamlit_app.py               ← ✅ Streamlit web app (chat-first UI)
│
├── src/
│   ├── __init__.py
│   ├── config_loader.py           ← YAML config + ${ENV_VAR} resolver
│   ├── data_processor.py          ← JSON datasets → Document objects
│   ├── embedding_engine.py        ← IBM Slate embeddings / local fallback
│   ├── vector_store.py            ← FAISS build / save / load / search
│   ├── llm_client.py              ← IBM Granite inference + prompt templates
│   └── rag_pipeline.py            ← End-to-end RAG orchestrator
│
├── data/
│   ├── raw/
│   │   ├── interview_questions.json  ← 40+ Q&A: Technical, Behavioral, HR
│   │   ├── hr_guidelines.json        ← Interview stages, company frameworks
│   │   ├── job_roles.json            ← Role definitions & prep strategies
│   │   └── industry_insights.json   ← 2025 trends, company interview styles
│   ├── processed/                    ← Auto-generated document chunks
│   └── embeddings/faiss_index/       ← Auto-generated vector index (first run)
│
├── config/
│   └── config.yaml                ← IBM Cloud & RAG configuration (reads from .env)
│
├── notebooks/
│   └── Interview_Trainer_Agent.ipynb  ← IBM Watson Studio exploration notebook
│
├── tests/
│   └── test_pipeline.py           ← Unit tests for core components
│
├── outputs/
│   ├── sessions/                  ← Per-session prep plans & evaluations
│   └── logs/                      ← Application logs
│
├── .env                           ← Local credentials (never committed)
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone <your-repo-url>
cd interview_trainer_agent

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure credentials

Create a `.env` file in the project root:

```env
WATSONX_API_KEY="your-ibm-cloud-api-key"
WATSONX_PROJECT_ID="your-watsonx-project-id"
WATSONX_URL="https://eu-de.ml.cloud.ibm.com"
GRANITE_MODEL_ID="ibm-granite/granite-4.0-h-small"
EMBEDDING_MODEL_ID="ibm/slate-125m-english-rtrvr-v2"

# RAG settings
TOP_K=5

# Generation parameters
MAX_NEW_TOKENS=900
DECODING_TEMPERATURE=0.4
```

> **No credentials?** The app still runs fully in stub/local mode — the RAG pipeline, vector store, and all UI features work. Only the LLM responses are replaced with context passthroughs until credentials are provided.

### 3. Launch the web app

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

**First launch** (~30–60 sec): The pipeline embeds all documents and builds the FAISS index automatically. Subsequent launches load the cached index in seconds.

---

## 🖥️ Web App — Tabs Overview

| Tab | What you can do |
|-----|----------------|
| **💬 Chat Coach** | Free-form conversation with your AI coach — ask questions, get a plan, or request feedback via quick-action buttons |
| **📋 Prep Plan** | One-click personalised prep plan with download option |
| **💡 Question Guide** | Paste any interview question and get expert guidance |
| **🏆 Answer Evaluator** | Type your answer side-by-side with the question, get scored feedback |
| **🗺️ Role Explorer** | Visual skill pills, interview rounds, company list, and prep strategy for any role/level |
| **⚙️ Status** | Live pipeline status, LLM/embedding mode badges, rebuild vector store |

The **sidebar** holds your profile (name, role, level) and system status badges that are shared across all tabs.

---

## 🔧 IBM Cloud Credentials

### Getting your credentials

1. **API Key** — [IBM Cloud → Manage → Access (IAM) → API Keys → Create](https://cloud.ibm.com/iam/apikeys)
2. **Project ID** — [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com) → Your project → **Manage → General → Project ID**
3. **Region URL** — Choose the URL that matches your IBM Cloud region:

| Region | `WATSONX_URL` |
|--------|---------------|
| EU Frankfurt | `https://eu-de.ml.cloud.ibm.com` |
| US South | `https://us-south.ml.cloud.ibm.com` |
| EU London | `https://eu-gb.ml.cloud.ibm.com` |
| Tokyo | `https://jp-tok.ml.cloud.ibm.com` |

---

## 🤖 Models Used

| Model | ID | Purpose |
|-------|----|---------|
| **IBM Granite 4.0** | `ibm-granite/granite-4.0-h-small` | LLM — prep plans, guidance, evaluation |
| **IBM Slate Embeddings** | `ibm/slate-125m-english-rtrvr-v2` | Semantic retrieval from knowledge base |
| **Fallback LLM** | `ibm/granite-3-8b-instruct` | Automatic fallback if primary unavailable |
| **Local embeddings** | `all-MiniLM-L6-v2` | SentenceTransformers fallback (no IBM creds) |

All IBM models run on the watsonx.ai **Lite (free)** plan.

---

## 📚 Knowledge Base

| File | Contents |
|------|---------|
| `interview_questions.json` | 40+ Q&A across 4 roles × 3 levels × 3 types (Technical, Behavioural, HR) with model answers, follow-ups, and improvement tips |
| `hr_guidelines.json` | Interview stages, common HR questions, IBM / Amazon / Google / Microsoft competency frameworks, communication tips |
| `job_roles.json` | Role definitions, required core & soft skills, interview rounds, 4–8 week preparation strategies with daily plans |
| `industry_insights.json` | 2024–2025 hiring trends, company interview styles (FAANG, IBM, etc.), common failure reasons |

---

## 💻 Programmatic Usage

The pipeline can also be used directly in Python without the web app:

```python
from src.rag_pipeline import InterviewRAGPipeline

pipeline = InterviewRAGPipeline(config_path="config/config.yaml")
pipeline.initialize()  # builds FAISS index on first run, loads cache after

# Generate a full personalised prep plan
result = pipeline.generate_prep_plan(
    name="Alice Johnson",
    role="software_engineer",
    level="mid"
)
print(result["preparation_plan"])

# Get guidance on a specific question
guidance = pipeline.get_question_guidance(
    name="Alice Johnson",
    role="software_engineer",
    level="mid",
    question="Design a rate limiter for a public API."
)
print(guidance["guidance"])

# Evaluate a practice answer
feedback = pipeline.evaluate_candidate_answer(
    name="Alice Johnson",
    role="software_engineer",
    level="mid",
    question="What is the CAP theorem?",
    candidate_answer="CAP theorem says a distributed system can only guarantee two of three..."
)
print(feedback["feedback"])

# Explore role information
info = pipeline.get_role_info(role="data_scientist", level="senior")
print(info["preparation_strategy"])
```

---

## 🧪 Running Tests

```bash
python -m pytest tests/test_pipeline.py -v
```

Tests that require `faiss-cpu` are automatically skipped if the package is not installed.

---

## 🔍 Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Pipeline initialisation failed: list indices must be integers or slices, not str` | IBM SDK ≥1.1 changed `embed_documents()` return shape | Already fixed in `src/embedding_engine.py` — delete `data/embeddings/faiss_index/` and restart |
| `LLM mode: stub` after setting credentials | Credentials not loaded from `.env` | Ensure `.env` exists in project root; `python-dotenv` must be installed |
| `ModuleNotFoundError: faiss` | `faiss-cpu` not installed | `pip install faiss-cpu` |
| `ModuleNotFoundError: ibm_watsonx_ai` | SDK not installed | `pip install ibm-watsonx-ai` |
| `No documents loaded` | JSON files missing from `data/raw/` | Ensure all 4 JSON files are present |
| Slow first startup | FAISS index being built (normal) | Wait 30–60 sec; subsequent starts are instant |
| Wrong region error | `WATSONX_URL` points to wrong region | Update `WATSONX_URL` in `.env` to match your IBM Cloud region |

---

## 🗺️ Watson Studio (Notebook) Usage

The project also ships with a Jupyter notebook for IBM Watson Studio exploration:

```
Project → Manage → Git Integration → connect repo → Sync
```

Open `notebooks/Interview_Trainer_Agent.ipynb` and set credentials in the environment cell, then:

```
Kernel → Restart & Run All
```

> For day-to-day use, the **Streamlit web app is recommended** over the notebook.

---

*Interview Trainer Agent v1.0 · IBM Granite 4.0 · watsonx.ai · FAISS · Streamlit*
