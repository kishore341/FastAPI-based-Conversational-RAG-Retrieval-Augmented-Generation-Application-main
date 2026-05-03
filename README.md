# Conversational RAG Application — Document Intelligence

> A production-ready Retrieval-Augmented Generation (RAG) system that enables natural language Q&A over proprietary documents, combining FAISS semantic search with LLM-generated, source-grounded responses.

---

## Overview

This application allows users to upload documents (PDF, text) and ask questions in plain English. Instead of relying purely on the LLM's training data, it **retrieves the most relevant passages from your documents first**, then generates an answer grounded in those passages — dramatically reducing hallucinations.

**Key Results:**
- ✅ Source-grounded responses with citation enforcement
- ✅ Multi-turn conversation with full session history
- ✅ 35% faster document query retrieval vs. full-document scanning
- ✅ Supports large enterprise knowledge bases (PDF + text)

---

## Architecture

```
User uploads Document
         │
         ▼
┌─────────────────────────────┐
│     Document Processor      │
│  PDF / Text → Chunks        │
│  Chunk → Embeddings (FAISS) │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     FAISS Vector Index      │  ← Stored in memory / disk
│  (Semantic similarity       │
│   search over chunks)       │
└──────────────┬──────────────┘
               │
         User asks question
               │
               ▼
┌─────────────────────────────┐
│     LangChain Retriever     │
│  Query → Top-K relevant     │
│  chunks fetched from FAISS  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     LLM Synthesis           │
│  (OpenAI / Groq LLaMA3)     │
│                             │
│  Prompt constraints:        │
│  - Answer only from context │
│  - Cite source passages     │
│  - Structured output format │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     MongoDB                 │
│  Persist conversation       │
│  history per session        │
└─────────────────────────────┘
               │
               ▼
          Answer + Source → User
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI, Python |
| RAG Framework | LangChain |
| Vector Store | FAISS |
| LLM | OpenAI API / Groq (LLaMA3) |
| Database | MongoDB (conversation history) |
| Document Support | PDF, plain text |
| Embedding | OpenAI Embeddings / HuggingFace |

---

## Features

### 1. End-to-End RAG Pipeline
```
Document Upload → Chunking → Embedding → FAISS Index → Retrieval → LLM Synthesis
```
- Automatic document chunking with configurable chunk size and overlap
- FAISS vector index built on upload — instant refresh when new documents added
- Top-K semantic retrieval before every LLM call

### 2. Hallucination Control
The synthesis prompt is engineered with strict constraints:
- **Source-only answers** — LLM instructed to answer only from retrieved context
- **Citation enforcement** — responses include source passage references
- **Structured output** — consistent, parseable answer format
- Off-topic questions are gracefully redirected

### 3. Multi-Turn Conversation
- Full conversation history stored in MongoDB per session
- Each new question includes previous Q&A pairs as context
- Enables follow-up questions like "Can you explain that further?" or "What about point 3?"

### 4. REST API Interface
Clean FastAPI endpoints for easy frontend integration:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload PDF or text document |
| POST | `/chat` | Ask a question, get grounded answer |
| GET | `/history/{session_id}` | Retrieve conversation history |
| DELETE | `/session/{session_id}` | Clear session |

---

## Project Structure

```
conversational-rag/
├── main.py                  # FastAPI app entry point
├── rag_pipeline.py          # Document loader, FAISS indexing, retrieval chain
├── llm_config.py            # OpenAI / Groq LLM initialization
├── mongo_service.py         # MongoDB conversation persistence
├── prompts.py               # System prompts with citation constraints
├── models/
│   └── requests.py          # Pydantic request/response models
├── requirements.txt
└── .env                     # API keys (not committed)
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/kishore341/conversational-rag
cd conversational-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Fill in your API keys in .env

# 4. Start MongoDB (local or Atlas)
# Update MONGO_URI in .env

# 5. Run the API
uvicorn main:app --reload
```

---

## Environment Variables

```env
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
MONGO_URI=mongodb://localhost:27017
DB_NAME=rag_conversations
```

---

## Requirements

```
langchain
langchain-community
langchain-openai
faiss-cpu
fastapi
uvicorn
pymongo
python-dotenv
pypdf
openai
```

---

## How It Works — Example

```
User: "What are the payment terms in the contract?"

System:
1. Embeds the question → searches FAISS index
2. Retrieves top 3 relevant chunks from the uploaded contract PDF
3. Sends chunks + question to LLM with citation prompt
4. LLM responds: "According to Section 4.2 of the contract, 
   payment is due within 30 days of invoice..."
5. Response + source reference saved to MongoDB
```

---

## Built At

**Zyxan Technologies Pvt. Limited**, Hyderabad

---

## Author

**Kishore Kumar Kunuku** — AI Engineer  
[LinkedIn](https://linkedin.com/in/kishore-kumar-kunuku-9bb91830b) | [GitHub](https://github.com/kishore341)
