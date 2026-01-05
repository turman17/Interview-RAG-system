# Interview API

Small FastAPI service that answers questions by routing between:
- a local RAG pipeline over `data/source.txt`
- an albums API (with local JSON fallback)

## Quickstart
1) Create a virtual environment and install deps:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2) Run the API:

```bash
uvicorn src.app:app --reload
```

3) Try it:

```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the singularity?"}'
```

## Configuration
Environment variables (all optional):
- `LLM_BASE_URL` (default: `http://127.0.0.1:1234`)
- `LLM_MODEL` (default: `qwen2.5-7b-instruct-1m`)
- `LLM_API_KEY` (default: `lm-studio`)
- `EMBEDDING_PROVIDER` (default: `local`)
- `EMBEDDING_MODEL` (default: `LLM_MODEL`)
- `ALBUMS_API_URL` (default: `https://jsonplaceholder.typicode.com/albums`)
- `RAG_SOURCE_PATH` (default: `data/source.txt`)
- `RAG_BACKEND` (default: `faiss`)
- `RAG_TOP_K` (default: `3`)
- `RAG_MIN_RELEVANCE` (default: `0.6`)
- `RAG_CHUNK_SIZE` (default: `800`)
- `RAG_CHUNK_OVERLAP` (default: `100`)
- `RAG_PERSIST_DIR` (default: `data/index`)

## Structure

```bash
src/
├─ app.py              # FastAPI app + routes
├─ agent.py            # routing + orchestration
├─ llm.py              # LLM wrapper
├─ rag.py              # RAG pipeline
├─ api_tool.py         # albums API client
└─ config.py           # env-driven settings
```
