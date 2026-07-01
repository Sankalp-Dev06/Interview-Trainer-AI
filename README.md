# 🎓 Interview Trainer Agent
### Powered by IBM Granite (watsonx.ai) + RAG | Problem Statement No. 22

---

An AI-powered Interview Trainer Agent built on **IBM Cloud** using **IBM Granite** foundation models and a **RAG (Retrieval-Augmented Generation)** architecture. It prepares users for job interviews by generating tailored question sets, model answers, improvement tips, and personalized preparation strategies based on their profile, experience level, and target job role.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Interview Trainer Agent (IBM Cloud)                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  User Input (Name, Role, Level, Question)                            │
│       │                                                               │
│       ▼                                                               │
│  ┌──────────────────┐    ┌─────────────────────────────────────────┐ │
│  │   RAG Pipeline   │    │        IBM watsonx.ai                   │ │
│  │                  │    │  ┌─────────────────────────────────┐    │ │
│  │  1. Query        │───▶│  │  IBM Slate Embedding Model       │    │ │
│  │  2. Embed Query  │    │  │  (slate-125m-english-rtrvr)      │    │ │
│  │  3. Retrieve     │    │  └─────────────────────────────────┘    │ │
│  │  4. Augment      │    │  ┌─────────────────────────────────┐    │ │
│  │  5. Generate     │───▶│  │  IBM Granite LLM                 │    │ │
│  └──────────────────┘    │  │  (granite-13b-chat-v2)           │    │ │
│       │                   │  └─────────────────────────────────┘    │ │
│       │                   └─────────────────────────────────────────┘ │
│       ▼                                                               │
│  ┌──────────────────────────────────────┐                            │
│  │         FAISS Vector Store            │                            │
│  │   (Interview Q's, HR Guidelines,     │                            │
│  │    Job Roles, Industry Insights)     │                            │
│  └──────────────────────────────────────┘                            │
│                                                                       │
│  Output: Personalized Interview Prep Plan, Scored Feedback, Tips     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
interview_trainer_agent/
│
├── notebooks/
│   └── Interview_Trainer_Agent.ipynb  ← Main IBM Watson Studio notebook
│
├── src/                               ← Python source modules
│   ├── __init__.py
│   ├── config_loader.py               ← YAML config + env var resolver
│   ├── data_processor.py              ← JSON dataset → Document objects
│   ├── embedding_engine.py            ← IBM Slate / SentenceTransformer
│   ├── vector_store.py                ← FAISS vector index (build/load/search)
│   ├── llm_client.py                  ← IBM Granite inference client
│   └── rag_pipeline.py                ← End-to-end RAG orchestrator
│
├── data/
│   ├── raw/                           ← Knowledge base (JSON datasets)
│   │   ├── interview_questions.json   ← 50+ Q&A: Technical, Behavioral, HR
│   │   ├── hr_guidelines.json         ← Interview stages, HR tips, frameworks
│   │   ├── job_roles.json             ← Role definitions & prep strategies
│   │   └── industry_insights.json     ← Trends, salary data, company styles
│   ├── processed/                     ← Processed document chunks (auto-generated)
│   └── embeddings/                    ← FAISS index files (auto-generated)
│       └── faiss_index/
│
├── config/
│   └── config.yaml                    ← IBM Cloud & RAG configuration
│
├── tests/
│   └── test_pipeline.py               ← Unit tests for core components
│
├── outputs/                           ← Generated reports (auto-created)
│   ├── sessions/                      ← Per-session prep plans & evaluations
│   └── logs/                          ← Application logs
│
├── requirements.txt                   ← Python dependencies
├── setup.py                           ← Package setup
├── .env.example                       ← Environment variable template
├── DEPLOYMENT.md                      ← IBM Cloud deployment guide
└── README.md                          ← This file
```

---

## 🚀 Quick Start on IBM Watson Studio

### Prerequisites
- IBM Cloud account (free): [cloud.ibm.com/registration](https://cloud.ibm.com/registration)
- IBM Watson Studio project (Lite — free)
- IBM watsonx.ai project (free tier)

### Step 1: Get IBM Cloud Credentials
```
1. Create watsonx.ai project → copy the Project ID
2. Create IBM Cloud API Key: Manage → Access (IAM) → API Keys
```

### Step 2: Upload Project to Watson Studio
```
Watson Studio → New Project → Add Assets → Upload all project files
```

### Step 3: Set Environment Variables in Notebook
```python
import os
os.environ['IBM_CLOUD_API_KEY']  = 'your-api-key'
os.environ['WATSONX_PROJECT_ID'] = 'your-project-id'
```

### Step 4: Run the Notebook
```
Open notebooks/Interview_Trainer_Agent.ipynb
→ Kernel → Restart & Run All
```

> **Stub Mode:** If credentials are not set, the notebook runs in stub mode — all retrieval, indexing, and session features work; only IBM Granite generation is replaced by context-forwarding stubs. This lets you explore the full pipeline before configuring IBM credentials.

---

## 💡 Features

| Feature | Description |
|---------|-------------|
| **Personalized Prep Plan** | Full interview plan (questions + model answers + strategy) tailored to name, role, level |
| **Question Deep-Dive** | Detailed guidance on any specific question with follow-ups and tips |
| **Answer Evaluator** | Score (1-10) + strengths + improvement tips + improved answer version |
| **Multi-Role Support** | Software Engineer, Data Scientist, Product Manager, DevOps Engineer |
| **Multi-Level Support** | Entry (0-2yr), Mid (3-6yr), Senior (7+yr) |
| **STAR Framework** | Behavioral questions guided by Situation-Task-Action-Result structure |
| **Industry Insights** | Salary ranges, hot skills, hiring trends, company-specific interview styles |
| **Visual Analytics** | Knowledge base analytics and personalized preparation roadmap charts |
| **Session Reports** | Automatically saved Markdown reports for each session |
| **IBM Granite LLM** | Primary: `ibm/granite-13b-chat-v2` via watsonx.ai |
| **IBM Slate Embeddings** | `ibm/slate-125m-english-rtrvr` for semantic retrieval |

---

## 🗄️ Knowledge Base

| Dataset | Contents |
|---------|---------|
| `interview_questions.json` | 50+ Q&A with model answers across 4 roles × 3 levels × 3 types |
| `hr_guidelines.json` | Interview stages, common HR questions, company frameworks (IBM, Amazon, Google, Microsoft) |
| `job_roles.json` | Role definitions, required skills, interview rounds, preparation strategies |
| `industry_insights.json` | 2024-2025 hiring trends, salary ranges, company interview styles, success metrics |

---

## 🧑‍💻 IBM Granite Models Used

| Purpose | Model ID | Notes |
|---------|----------|-------|
| **Primary LLM** | `ibm/granite-13b-chat-v2` | Main generation model |
| **Fallback LLM** | `ibm/granite-3-8b-instruct` | Lighter, faster variant |
| **Embeddings** | `ibm/slate-125m-english-rtrvr` | Semantic retrieval |

All models are available on IBM watsonx.ai **Lite (free)** plan.

---

## 🔧 Local Development

```bash
# Clone / download the project
git clone <your-repo-url>
cd interview_trainer_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your IBM Cloud credentials

# Run unit tests
python -m pytest tests/ -v

# Launch Jupyter notebook
jupyter notebook notebooks/Interview_Trainer_Agent.ipynb
```

---

## 🏃 Programmatic Usage

```python
from src.rag_pipeline import InterviewRAGPipeline

# Initialize pipeline
pipeline = InterviewRAGPipeline(config_path='config/config.yaml')
pipeline.initialize()

# Generate a full prep plan
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

# Evaluate your practice answer
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

## 📊 IBM Cloud Services Used

| Service | Plan | Purpose |
|---------|------|---------|
| **IBM Watson Studio** | Lite (free) | Jupyter notebook hosting |
| **IBM watsonx.ai** | Lite (free) | IBM Granite LLM + IBM Slate Embeddings |
| **IBM Cloud Object Storage** | Lite (25 GB free) | Optional: index persistence |

---

## 📋 Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for full step-by-step IBM Cloud deployment instructions including Watson Studio setup, environment variables, and IBM Code Engine containerization.

---

## 🧪 Running Tests

```bash
python -m pytest tests/test_pipeline.py -v
```

Expected: All tests pass. Tests that require `faiss-cpu` are auto-skipped if the package is not installed.

---

*Interview Trainer Agent v1.0 | IBM Cloud + IBM Granite + watsonx.ai*
#   I n t e r v i e w - T r a i n e r - A I  
 