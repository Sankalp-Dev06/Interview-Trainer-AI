# 🎯 Interview Trainer Agent
### Powered by IBM Granite (watsonx.ai) + RAG | Problem Statement No. 22

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-red.svg)](https://streamlit.io)
[![IBM Granite](https://img.shields.io/badge/IBM-Granite%204.0-0f62fe.svg)](https://watsonx.ai)
[![Streamlit Cloud](https://img.shields.io/badge/deploy-Streamlit%20Cloud-ff4b4b.svg)](https://share.streamlit.io)

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
| **🌗 Light / Dark Mode** | One-click theme toggle in the sidebar — preference remembered for the session |
| **Multi-Role Support** | Software Engineer · Data Scientist · Product Manager · DevOps Engineer |
| **Multi-Level Support** | Entry (0–2 yr) · Mid (3–6 yr) · Senior (7+ yr) |
| **STAR Framework** | All behavioural answers coached with Situation-Task-Action-Result structure |
| **Graceful Fallback** | Runs in stub/local mode without IBM credentials — full UI still functional |

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
├── streamlit_app.py               ← Streamlit web app (chat-first UI + light/dark mode)
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
│   │   ├── interview_questions.json
│   │   ├── hr_guidelines.json
│   │   ├── job_roles.json
│   │   └── industry_insights.json
│   ├── processed/                 ← Auto-generated document chunks
│   └── embeddings/faiss_index/    ← Auto-generated vector index (first run)
│
├── config/
│   └── config.yaml                ← IBM Cloud & RAG configuration
│
├── .streamlit/
│   └── config.toml                ← Streamlit server + theme config
│
├── app.json                       ← Deployment env-var manifest
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Local Development

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

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
WATSONX_API_KEY="your-ibm-cloud-api-key"
WATSONX_PROJECT_ID="your-watsonx-project-id"
WATSONX_URL="https://eu-de.ml.cloud.ibm.com"
GRANITE_MODEL_ID="ibm-granite/granite-4.0-h-small"
EMBEDDING_MODEL_ID="ibm/slate-125m-english-rtrvr-v2"
```

> **No credentials?** The app still runs fully in stub/local mode — all UI features work, only LLM responses use context pass-through until real credentials are provided.

### 3. Run

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501**

**First launch** (~30–60 sec): the pipeline embeds all documents and builds the FAISS index automatically. Subsequent launches load the cached index in seconds.

---

## ☁️ Deploy to Streamlit Community Cloud

### Step-by-step

**1. Push your repo to GitHub**

```bash
git add .
git commit -m "ready for deployment"
git push origin main
```

**2. Go to [share.streamlit.io](https://share.streamlit.io)**

- Sign in with your GitHub account
- Click **"New app"**

**3. Fill in the deploy form**

| Field | Value |
|-------|-------|
| Repository | `your-github-username/interview-trainer-agent` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |

**4. Add your IBM credentials as Secrets**

Click **"Advanced settings"** → **"Secrets"** and paste:

```toml
WATSONX_API_KEY = "your-ibm-cloud-api-key"
WATSONX_PROJECT_ID = "your-watsonx-project-id"
WATSONX_URL = "https://eu-de.ml.cloud.ibm.com"
GRANITE_MODEL_ID = "ibm-granite/granite-4.0-h-small"
EMBEDDING_MODEL_ID = "ibm/slate-125m-english-rtrvr-v2"
```

> Secrets are stored encrypted — never put them in your code or `.env` file that is pushed to GitHub.

**5. Click "Deploy!"**

Streamlit Cloud will install dependencies from `requirements.txt` and launch the app. The first boot takes ~2–3 minutes while the FAISS index is built.

### What gets deployed

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main app entry point |
| `requirements.txt` | All Python dependencies |
| `.streamlit/config.toml` | Server settings + default theme |
| `config/config.yaml` | RAG pipeline config (reads secrets as env vars) |
| `src/` | Pipeline source code |
| `data/raw/` | Knowledge base JSON files |
| `app.json` | Documents the required env variables |

### What you do NOT need on Streamlit Cloud

- `Dockerfile` / `docker-compose.yml` — not used
- `.env` — use Streamlit Secrets instead
- Local `venv/` folder — Streamlit Cloud builds its own environment

---

## 🌗 Light / Dark Mode

The app ships with a full CSS theming system and a toggle in the sidebar.

| Mode | Appearance |
|------|-----------|
| 🌙 **Dark** (default) | Deep navy surfaces, IBM blue accents |
| ☀️ **Light** | Clean white surfaces, soft grey backgrounds |

Click **"🌙 Dark mode · Switch to Light"** in the sidebar to toggle. Choice persists for the session.

---

## 🖥️ Web App — Tabs Overview

| Tab | What you can do |
|-----|----------------|
| **💬 Chat Coach** | Free-form conversation — ask questions, get a plan, request feedback |
| **📋 Prep Plan** | One-click personalised prep plan with download option |
| **💡 Question Guide** | Paste any interview question and get expert guidance |
| **🏆 Answer Evaluator** | Type your answer side-by-side with the question, get scored feedback |
| **🗺️ Role Explorer** | Skill pills, interview rounds, company list, prep strategy |
| **⚙️ Status** | Live pipeline status, LLM/embedding mode badges, rebuild vector store |

The **sidebar** holds your profile (name, role, level), the light/dark toggle, and system status badges.

---

## 🔧 IBM Cloud Credentials

### Getting your credentials

1. **API Key** — [IBM Cloud → Manage → Access (IAM) → API Keys → Create](https://cloud.ibm.com/iam/apikeys)
2. **Project ID** — [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com) → Your project → **Manage → General → Project ID**
3. **Region URL** — Choose the URL matching your IBM Cloud region:

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
| `interview_questions.json` | 40+ Q&A across 4 roles × 3 levels × 3 types (Technical, Behavioural, HR) |
| `hr_guidelines.json` | Interview stages, common HR questions, IBM / Amazon / Google / Microsoft frameworks |
| `job_roles.json` | Role definitions, required skills, interview rounds, 4–8 week prep strategies |
| `industry_insights.json` | 2024–2025 hiring trends, company interview styles |

---

## 🧪 Running Tests

```bash
python -m pytest tests/test_pipeline.py -v
```

---

## 🔍 Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `LLM mode: stub` after setting credentials | Credentials not loaded | On Streamlit Cloud, check **App settings → Secrets**; locally ensure `.env` exists |
| `ModuleNotFoundError: faiss` | `faiss-cpu` not installed | `pip install faiss-cpu` |
| `No documents loaded` | JSON files missing from `data/raw/` | Ensure all 4 JSON files are present in the repo |
| Slow first startup | FAISS index being built (normal) | Wait 60–90 sec on Streamlit Cloud; subsequent reboots are faster |
| Wrong region error | `WATSONX_URL` points to wrong region | Update the secret to match your IBM Cloud region |

---

*Interview Trainer Agent v1.0 · IBM Granite 4.0 · watsonx.ai · FAISS · Streamlit*
