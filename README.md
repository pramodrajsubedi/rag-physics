# Space Physics RAG Assistant

Production RAG system for scientific literature Q&A — retrieves semantically relevant chunks from space physics research papers and generates grounded, cited answers via Gemini. Built end-to-end: ingestion pipeline, vector store, prompt engineering layer, REST API, and CI/CD deployment.

**[API →](https://rag-physics-api.onrender.com/docs)** 

---

## Architecture

```
arXiv API → PDF extraction → LangChain chunking → sentence-transformers
                                                          │
                                                    ChromaDB (disk)
                                                          │
                        User query → embed → similarity search → top-5 chunks
                                                          │
                                              Gemini prompt (system + context)
                                                          │
                                              Cited answer + sources → FastAPI
                                                          │
                                                       Render
```

**Design decisions:**
- Sentence-transformers for embeddings 
- ChromaDB with persistent storage — ingestion runs once, vector store survives restarts
- Lazy model loading — avoids cold-start failures 
- Gemini only handles generation — keeps LLM calls minimal and fast

---

## API

```bash
curl -X POST https://rag-physics-api.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What ML methods are used for geomagnetic storm prediction?", "top_k": 5}'
```

```json
{
  "question": "What ML methods are used for geomagnetic storm prediction?",
  "answer": "Statistical models, artificial neural networks, and nonlinear auto-regression techniques have been applied to predict Dst and Kp indices (Lundstedt et al. 2002; Temerin & Li 2002; Shprits et al. 2019). Traditional models using in-situ solar wind parameters yield 1–2 hour lead times...",
  "sources": [
    "Geomagnetic storm forecasting from solar coronal holes",
    "The importance of ensemble techniques for operational space weather forecasting"
  ]
}
```

---

## Prompt Engineering

Three-layer prompt design:

**Layer 1 — System instruction:** Constrains the model to domain-expert behavior, enforces citation requirements, and defines output structure (Summary / Key Details / Sources).

**Layer 2 — Retrieved context:** Top-5 chunks injected with source metadata (title, authors, URL) — model only answers from this window, not from training data.

**Layer 3 — Output constraints:** Explicit instruction to flag insufficient context rather than hallucinate — critical for scientific Q&A where false citations carry real cost.

```python
SYSTEM_PROMPT = """You are an expert space physics research assistant...
Answer strictly from provided excerpts. Cite sources by title.
If context is insufficient, say so explicitly.
Structure: Summary → Key Details → Sources."""
```

---

## Knowledge Base

| Property | Value |
|---|---|
| Source | arXiv (physics.space-ph, astro-ph.SR) |
| Coverage | Geomagnetic storms, space weather forecasting, solar flares, magnetosphere |
| Total chunks | 1,662 |
| Chunk size | 2,000 tokens / 200 overlap |
| Embedding model | all-MiniLM-L6-v2 (384-dim, local) |
| Vector store | ChromaDB — cosine similarity, persistent on disk |

---

## Ingestion Pipeline

```bash
python src/ingest.py
```

Fetches arXiv papers → downloads PDFs → extracts text via PyMuPDF → chunks with LangChain `RecursiveCharacterTextSplitter` → embeds locally → stores in ChromaDB. Resumes from checkpoint — already-embedded chunks are skipped on re-run.

---

## MLOps

| Component | Implementation |
|---|---|
| API | FastAPI with Pydantic validation |
| Containerization | Docker — ChromaDB baked into image at build time |
| CI/CD | GitHub Actions — test + Docker build on every push to main |
| Deployment | Render |

---

## Local Setup

```bash
git clone https://github.com/pramodrajsubedi/rag-physics-assistant
cd rag-physics-assistant
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
python src/ingest.py          # one-time ingestion
uvicorn api.main:app --reload # start API at localhost:8000
```

---

## Stack

`Python` · `LangChain` · `ChromaDB` · `sentence-transformers` · `Gemini (google-genai)` · `FastAPI` · `PyMuPDF` · `Docker` · `GitHub Actions` · `Render`

---

**Pramod Raj Subedi** · [LinkedIn](https://www.linkedin.com/in/pramodrajsubedi/)
