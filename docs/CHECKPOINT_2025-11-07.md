# Project Checkpoint – 2025-11-07

This checkpoint captures the current architecture, environment assumptions, and rollback guidance for the IISc CDS Chatbot (RAG pipeline). Use this file to understand the state at commit `2d61010` on branch `dev_hs`.

---
## Git Snapshot
- Branch: `dev_hs`
- Commit (short): `2d61010`
- Uncommitted changes (porcelain):
  - M .gitignore
  - ?? scripts/crawl_iisc_full.py
  - ?? scripts/remote_poll.sh
  - ?? scripts/requirements_crawler.txt
  - ?? scripts/run_application.sh
  - ?? scripts/stop_application.sh

If you need an immutable snapshot, commit or stash these files before proceeding.

---
## High-Level Architecture (Current)
Pipeline (offline):
```
crawl → preprocess + dedup → link graph → tree chunking → embeddings (FastEmbed/hash) → index build (NumPy / optional FAISS)
```
Serving (online):
- FastAPI (`api/main.py`) endpoints: `/health`, `/search`, `/query`
- Streamlit UI (`ui/streamlit_app.py`) for chat + retrieval + optional Gemini generation
- Vector store: primary NumPy in-memory index (`data/index/hash_numpy` or fastembed variant)

Artifacts live under `data/index/`:
- `hash_numpy/`
- `fastembed_bge_small/`
- `faiss_flat_hash_sample/`
- `fastembed_bge_small_sample/`

---
## Key Components
| Layer | File(s) | Notes |
|-------|---------|-------|
| Crawler | `scripts/crawl_iisc_full.py` | BFS, politeness, depth control (newly untracked) |
| Preprocess | `scripts/preprocess_and_chunk.py` | Cleans text, generates dedup reports |
| Link Graph | `scripts/build_link_graph.py` | Produces `link_graph*.json` files |
| Chunking | `scripts/chunk_pages_tree.py`, `src/data_processing/*` | Hierarchical + referential context |
| Embeddings | `src/embeddings/fastembed_embedder.py`, `src/embeddings/hash_embedder.py` | FastEmbed or hashing fallback |
| Index | `scripts/build_index.py`, `src/vector_store/numpy_store.py` | Generates `vectors.npz` + metadata |
| Retrieval API | `api/main.py` | Rerank boosts + exact phrase fallback |
| UI | `ui/streamlit_app.py` | Index discovery, Gemini calls, chat session state |
| LLM Wrapper | `src/llm/gemini.py` | Thin abstraction over Gemini API |
| App Control | `scripts/run_application.sh`, `scripts/stop_application.sh` | Start/stop with port cleanup |

---
## Environment Variables (from `.env.example` at checkpoint)
```
GEMINI_API_KEY=
# INDEX_DIR= (optional override)
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
LOG_LEVEL=INFO
DB_HOST=localhost
DB_NAME=research_rag_db
DB_USER=your_username
DB_PASSWORD=your_password
DB_PORT=5432
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```
Notes:
- Database + Ollama vars are legacy/unused presently.
- `INDEX_DIR` can point to one of the directories listed under `data/index/`. If unset, defaults inside code apply.

---
## Current Limitations / Gaps (Pre-Deployment)
1. No centralized Pydantic settings loader.
2. No Dockerfile or container strategy checked in.
3. Single-process FastAPI startup (no explicit workers/uvicorn tuning).
4. No rate limiting or auth for `/search` / `/query`.
5. Index versioning metadata lacks manifest file (only `index_meta.json` per directory).
6. CI/CD workflows absent.
7. Observability limited to log files (no metrics/tracing).
8. Mixed requirements (crawler + runtime) in one `requirements.txt`.

---
## Rollback Strategy
If deployment-oriented changes become unstable, revert to this checkpoint:
1. Ensure you have the commit hash (tag it optionally: `git tag checkpoint-2025-11-07 2d61010`).
2. Use `git reset --hard 2d61010` to return working tree to this state (WARNING: loses local changes).
3. If new index artifacts were added but cause issues, point `INDEX_DIR` back to a known working folder (`data/index/hash_numpy`).
4. If configuration refactors fail, restore `.env.example` from this file’s block above.
5. For any added Docker/CI files causing pipeline failure, `git checkout 2d61010 <filepath>` per file.

---
## Suggested Next Actions (Forward Plan)
1. Add `docs/ROADMAP.md` summarizing phases (MVP → scaling → advanced retrieval).
2. Introduce `runtime_settings.py` using Pydantic for env parsing.
3. Split dependencies: `requirements-runtime.txt`, `requirements-build.txt`.
4. Create multi-stage `Dockerfile` (builder for index vs runtime).
5. Implement `/ready` endpoint & extend `/health` with index version info.
6. Generate an `index_manifest.json` during `scripts/build_index.py` with reproducibility metadata.
7. Add simple rate limiting middleware (e.g., token bucket) + query length cap.
8. Set up GitHub Actions workflow (lint → test → build image → optional deploy).
9. Add Prometheus instrumentation or structured JSON logging.

---
## Invariants To Preserve While Evolving
- FastEmbed/hash dual encoder support; avoid hard-coding one path.
- Simple inspectable index format (NumPy + JSONL).
- Tree-smart chunking logic (path context) untouched until dedicated tests added.
- Rerank logic in `api/main.py` boosting exact phrases; ensure test coverage before refactor.

---
## Minimal Test Targets To Add Soon
- Unit test for hashing embedder dimension consistency.
- Regression test: search endpoint returns exact-phrase boosted result first when phrase present.
- UI retrieval dimension mismatch guard (raise on wrong encoder) using sample index fixture.

---
## Manual Verification Commands (At Checkpoint)
```bash
# Start application
bash scripts/run_application.sh
curl -s http://localhost:8000/health | jq
# Sample search
curl -X POST http://localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query":"data science","top_k":3}' | jq
```

---
## Contact / Ownership
- Repository: `IISC-RAG-based-chatbot`
- Owner (GitHub org/user): `T2AH`
- Working branch: `dev_hs`

---
## How To Use This Checkpoint
Treat this file as a stable anchor before introducing deployment-centric changes. Update with a new dated file when major phases complete (e.g., `CHECKPOINT_2025-11-21.md`). Avoid editing this file retroactively—create a new one instead.

---
*End of checkpoint.*
