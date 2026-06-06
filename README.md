# resume-screening-system
AI-powered resume ranking engine · TF-IDF + Sentence Transformers · Top-5 precision 0.62→0.79 · FastAPI · PostgreSQL · Docker

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue?style=flat-square&logo=postgresql)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)](https://docker.com)

---

## Results

| Metric | Before | After |
|---|---|---|
| Top-5 Precision | 0.62 | **0.79** |
| API Latency | 420ms | **250ms** (~40% reduction) |
| Dataset Size | — | ~8,500 resumes |

---

## Overview

A production-grade resume screening API that automatically ranks candidates against a job description. Built to replace manual HR shortlisting for high-volume hiring pipelines.

The ranking engine combines two complementary techniques:
- **TF-IDF** captures keyword and skill overlap between resume and job description
- **Sentence Transformers** (all-MiniLM-L6-v2) capture semantic similarity beyond exact keyword matches

The two scores are combined using a weighted ensemble, and results are served via an async FastAPI endpoint with PostgreSQL-backed candidate storage.

---

## Architecture

```
Job Description (text)
        │
        ▼
┌───────────────────┐
│   Text Parser     │  ← PyMuPDF / pdfplumber
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────┐
│        Embedding Engine        │
│  ┌─────────────┐  ┌─────────┐ │
│  │   TF-IDF    │  │SBERT    │ │
│  │ Vectorizer  │  │Encoder  │ │
│  └──────┬──────┘  └────┬────┘ │
│         └──────┬───────┘      │
│           Weighted Ensemble   │
└───────────────┬───────────────┘
                │ Cosine Similarity Score
                ▼
┌───────────────────────────────┐
│     PostgreSQL (Indexed)      │  ← Candidate + score storage
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│    FastAPI (Async Endpoints)  │  ← /rank, /upload, /results
└───────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI (async) |
| ML / Embeddings | Scikit-learn, Sentence Transformers (all-MiniLM-L6-v2) |
| PDF Parsing | PyMuPDF, pdfplumber |
| Database | PostgreSQL 15 (indexed queries) |
| Containerisation | Docker, Docker Compose |
| Language | Python 3.10+ |

---

## Key Features

- **Dual-model ranking** — TF-IDF for keyword matching + Sentence-BERT for semantic understanding
- **Async API** — non-blocking FastAPI endpoints handle concurrent resume uploads
- **Indexed DB queries** — PostgreSQL index on embedding vectors reduces query time
- **Batch upload** — accepts ZIP of resumes or individual PDFs
- **Explainability** — response includes top matching keywords per candidate
- **Containerised** — single `docker-compose up` deploys the full stack

---

## Project Structure

```
resume-screening-system/
├── app/
│   ├── main.py              # FastAPI app entrypoint
│   ├── routers/
│   │   ├── upload.py        # Resume upload endpoints
│   │   └── rank.py          # Ranking endpoints
│   ├── services/
│   │   ├── parser.py        # PDF text extraction
│   │   ├── embedder.py      # TF-IDF + SBERT embedding
│   │   └── ranker.py        # Cosine similarity scoring
│   ├── models/
│   │   └── candidate.py     # SQLAlchemy ORM model
│   └── db.py                # PostgreSQL connection pool
├── tests/
│   ├── test_ranker.py
│   └── test_api.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (recommended)
- PostgreSQL 15 (if running locally)

### Run with Docker (recommended)

```bash
git clone https://github.com/kanishasharma/resume-screening-system.git
cd resume-screening-system

docker-compose up --build
```

API will be available at `http://localhost:8000`

### Run locally

```bash
git clone https://github.com/kanishasharma/resume-screening-system.git
cd resume-screening-system

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Set up PostgreSQL connection
export DATABASE_URL="postgresql://user:password@localhost:5432/resume_db"

uvicorn app.main:app --reload
```

---

## API Reference

### `POST /upload`
Upload one or more resumes (PDF).

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.pdf"
```

**Response:**
```json
{
  "uploaded": 2,
  "candidate_ids": ["c1a2b3", "d4e5f6"]
}
```

### `POST /rank`
Rank all uploaded candidates against a job description.

```bash
curl -X POST "http://localhost:8000/rank" \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "We are looking for a Python backend engineer with FastAPI and PostgreSQL experience...",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "results": [
    {
      "candidate_id": "c1a2b3",
      "name": "Jane Doe",
      "score": 0.87,
      "top_keywords": ["FastAPI", "PostgreSQL", "Python", "REST API"]
    }
  ],
  "latency_ms": 248
}
```

### `GET /results/{job_id}`
Retrieve previously computed rankings.

---

## How the Ranking Works

```python
# Simplified core ranking logic

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes(job_description: str, resumes: list[str]) -> list[float]:
    # TF-IDF score
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([job_description] + resumes)
    tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Semantic score (Sentence-BERT)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([job_description] + resumes)
    sbert_scores = cosine_similarity([embeddings[0]], embeddings[1:]).flatten()

    # Weighted ensemble
    final_scores = 0.4 * tfidf_scores + 0.6 * sbert_scores
    return final_scores.tolist()
```

---

## Performance Optimisations

- **Async endpoints** — `async def` route handlers avoid blocking the event loop during DB I/O
- **PostgreSQL indexing** — B-tree index on `candidate_id` and `job_id` columns
- **Connection pooling** — `asyncpg` connection pool reused across requests (not created per-request)
- **Batch embedding** — resumes embedded in batches of 32 (not one by one) to maximise GPU/CPU throughput

---

## License

MIT License — see [LICENSE](LICENSE) for details.
