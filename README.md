# Interview Trainer Agent
### Powered by IBM Granite (watsonx.ai) + RAG | Problem Statement No. 22

An AI-powered Interview Trainer Agent built on **IBM Cloud** using **IBM Granite** foundation models and a **RAG (Retrieval-Augmented Generation)** architecture. It prepares users for job interviews by generating tailored question sets, model answers, improvement tips, and personalized preparation strategies based on their profile, experience level, and target job role.

---

## Architecture

```
User Input (Name, Role, Level, Question)
        |
        v
  ┌─────────────────────────────────────────────────────────┐
  │                     RAG Pipeline                         │
  │                                                          │
  │  1. Embed query  ──►  IBM Slate (slate-125m-rtrvr)       │
  │  2. Retrieve     ──►  FAISS Vector Store                 │
  │                        (Interview Q's, HR Guidelines,    │
  │                         Job Roles, Industry Insights)    │
  │  3. Augment prompt with retrieved context                │
  │  4. Generate     ──►  IBM Granite (granite-13b-chat-v2)  │
  └─────────────────────────────────────────────────────────┘
        |
        v
  Personalized Prep Plan · Model Answers · Scored Feedback
```

---

## Project Structure

```
interview_trainer_agent/
│
├── notebooks/
│   └── Interview_Trainer_Agent.ipynb  ← Main IBM Watson Studio notebook (19 cells)
│
├── src/
│   ├── __init__.py
│   ├── config_loader.py               ← YAML config + ${ENV_VAR} resolver
│   ├── data_processor.py              ← JSON datasets → Document objects
│   ├── embedding_engine.py            ← IBM Slate embeddings / local fallback
│   ├── vector_store.py                ← FAISS build / save / load / search
│   ├── llm_client.py                  ← IBM Granite inference + prompt templates
│   └── rag_pipeline.py                ← End-to-end RAG orchestrator
│
├── data/
│   ├── raw/
│   │   ├── interview_questions.json   ← 40+ Q&A: Technical, Behavioral, HR
│   │   ├── hr_guidelines.json         ← Interview stages, company frameworks
│   │   ├── job_roles.json             ← Role definitions & prep strategies
│   │   └── industry_insights.json     ← 2025 trends, salary data, company styles
│   ├── processed/                     ← Auto-generated document chunks
│   └── embeddings/faiss_index/        ← Auto-generated vector index
│
├── config/
│   └── config.yaml                    ← IBM Cloud & RAG configuration
│
├── tests/
│   └── test_pipeline.py               ← Unit tests for core components
│
├── outputs/
│   ├── sessions/                      ← Per-session prep plans & evaluations
│   └── logs/                          ← Application logs
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start on IBM Watson Studio

### Prerequisites
- IBM Cloud account (free): [cloud.ibm.com/registration](https://cloud.ibm.com/registration)
- IBM Watson Studio instance (Lite — free)
- IBM watsonx.ai project (free tier)

### Step 1 — Get IBM Cloud credentials

1. Go to [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com) → **New project**
2. Copy **Project ID** from: Project → Manage → General → Project ID
3. Create **API Key**: IBM Cloud → Manage → Access (IAM) → API Keys → Create

### Step 2 — Upload project to Watson Studio

**Option A — Git integration (recommended)**
```
Project → Manage → Git Integration → connect your repo → Sync
```

**Option B — Manual upload**
```
Project → Assets → New asset → Upload files
Upload: notebooks/, src/, data/raw/, config/config.yaml, requirements.txt
```

### Step 3 — Set credentials in the notebook

Open `notebooks/Interview_Trainer_Agent.ipynb`, Cell 3:

```python
import os
os.environ['IBM_CLOUD_API_KEY']  = 'your-api-key-here'
os.environ['WATSONX_PROJECT_ID'] = 'your-project-id-here'
```

Or set them permanently via **Project → Manage → Environments → Environment variables**.

### Step 4 — Run the notebook

```
Kernel → Restart & Run All
```

- **Cell 1** installs all packages — restart kernel after it completes
- **Cell 6** builds the FAISS index on first run (~30–60 seconds), then caches it
- **Cell 9** calls IBM Granite and generates your personalized prep plan

> **Stub mode:** Without credentials, every pipeline feature (indexing, retrieval, analytics, sessions) still runs fully. Only LLM responses are replaced by context passthrough stubs. Configure credentials to activate IBM Granite generation.

---

## Features

| Feature | Description |
|---------|-------------|
| **Personalized Prep Plan** | Full question set + model answers + strategy tailored to name, role, and level |
| **Question Deep-Dive** | Detailed guidance on any specific question with follow-ups and tips |
| **Answer Evaluator** | Score (1–10) + strengths + improvement tips + improved answer version |
| **Multi-Role Support** | Software Engineer, Data Scientist, Product Manager, DevOps Engineer |
| **Multi-Level Support** | Entry (0–2 yr), Mid (3–6 yr), Senior (7+ yr) |
| **STAR Framework** | All behavioral questions coached with Situation-Task-Action-Result structure |
| **Industry Insights** | Salary ranges, hot skills, hiring trends, company-specific interview styles |
| **Visual Analytics** | Knowledge base analytics chart + personalized preparation roadmap |
| **Session Reports** | Auto-saved Markdown reports to `outputs/sessions/` |

---

## Knowledge Base

| File | Contents |
|------|---------|
| `interview_questions.json` | 40+ Q&A across 4 roles × 3 levels × 3 types (Technical, Behavioral, HR) with model answers, follow-ups, and improvement tips |
| `hr_guidelines.json` | Interview stages, common HR questions, IBM / Amazon / Google / Microsoft competency frameworks |
| `job_roles.json` | Role definitions, required skills, interview rounds, 4–8 week preparation strategies |
| `industry_insights.json` | 2024–2025 hiring trends, salary ranges, company interview styles, common failure reasons |

---

## IBM Cloud Services & Models

| Service / Model | ID / Plan | Purpose |
|-----------------|-----------|---------|
| IBM Watson Studio | Lite (free) | Jupyter notebook hosting |
| IBM Granite LLM | `ibm/granite-13b-chat-v2` | Personalized answer & plan generation |
| IBM Granite fallback | `ibm/granite-3-8b-instruct` | Lighter, faster variant |
| IBM Slate Embeddings | `ibm/slate-125m-english-rtrvr` | Semantic document retrieval |
| IBM Cloud Object Storage | Lite (25 GB free) | Optional: FAISS index persistence |

All models are available on the watsonx.ai **Lite (free)** plan.

---

## Region Configuration

The default region is `us-south`. Update `config/config.yaml` line 14 if your account is in a different region:

| Region | URL |
|--------|-----|
| US South (default) | `https://us-south.ml.cloud.ibm.com` |
| EU Frankfurt | `https://eu-de.ml.cloud.ibm.com` |
| EU London | `https://eu-gb.ml.cloud.ibm.com` |
| Tokyo | `https://jp-tok.ml.cloud.ibm.com` |

---

## Local Development

```bash
# Clone the project
git clone <your-repo-url>
cd interview_trainer_agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set credentials as environment variables
export IBM_CLOUD_API_KEY=your-api-key-here
export WATSONX_PROJECT_ID=your-project-id-here
# Windows: set IBM_CLOUD_API_KEY=your-api-key-here

# Run unit tests
python -m pytest tests/ -v

# Launch notebook
jupyter notebook notebooks/Interview_Trainer_Agent.ipynb
```

---

## Programmatic Usage

```python
from src.rag_pipeline import InterviewRAGPipeline

pipeline = InterviewRAGPipeline(config_path='config/config.yaml')
pipeline.initialize()  # builds FAISS index on first run, loads from cache after

# Generate a full personalized prep plan
result = pipeline.generate_prep_plan(
    name="Arjun Sharma",
    role="software_engineer",
    level="mid"
)
print(result['preparation_plan'])

# Get guidance on a specific question
guidance = pipeline.get_question_guidance(
    name="Arjun Sharma",
    role="software_engineer",
    level="mid",
    question="Design a rate limiter for a public API."
)
print(guidance['guidance'])

# Evaluate a practice answer
feedback = pipeline.evaluate_candidate_answer(
    name="Arjun Sharma",
    role="software_engineer",
    level="mid",
    question="What is the CAP theorem?",
    candidate_answer="CAP theorem says a distributed system can only guarantee two of three..."
)
print(feedback['feedback'])
```

---

## Running Tests

```bash
python -m pytest tests/test_pipeline.py -v
```

Test requiring `faiss-cpu` are automatically skipped if the package is not installed.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: ibm_watsonx_ai` | Cell 1 not run or kernel not restarted | Run Cell 1 → Kernel → Restart |
| `ModuleNotFoundError: faiss` | Same | Same fix |
| `LLM mode: stub` after setting credentials | Extra whitespace in key/ID | Re-copy credentials without leading/trailing spaces |
| `FileNotFoundError: config/config.yaml` | Wrong working directory | Cell 2 runs `os.chdir(PROJECT_ROOT)` automatically |
| `No documents loaded` | JSON files not at `data/raw/` | Upload all 4 JSON files to the correct path |
| `IBM credentials invalid` | Wrong region endpoint | Update `watsonx.url` in `config/config.yaml` |

---

*Interview Trainer Agent v1.0 | IBM Cloud + IBM Granite + watsonx.ai*
