

<div align="center">
	<h1>IISc Chatbot</h1>
	<br><br>
	<strong>Web Application URL:</strong><br>
	<a href="http://65.0.29.172:8501" style="font-size:1.2em; font-weight:bold;">http://65.0.29.172</a>
	<br><br>
	<em>This is the main public URL for users.</em>
</div>

---

## Quick Start

### 1. SSH into your server
```bash
ssh -i /path/to/iisc-chatbot.pem ubuntu@65.0.29.172
```
Replace `/path/to/your-key.pem` with your .pem key path.
```
cd /opt/iisc-chatbot
```

After successful SSH login, navigate to the application directory:

### 2. Build & Run the Application
```bash
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```
Or use the shell script:
```bash
bash scripts/run_application.sh
```

### 3. (Optional) Access Container Shell
```bash
docker exec -it iisc-chatbot-api-1 /bin/bash
```

---

## Features
- FastEmbed-powered semantic search
- NumPy vector store for blazing-fast retrieval
- Streamlit UI for chat and source citations
- Gemini LLM integration (optional)
- Modular pipeline: crawl, clean, chunk, embed, serve

---

# IISc Chatbot (RAG Pipeline: FastEmbed + NumPy)


## Example API Usage
<details>
<summary>Show API Examples</summary>

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Search Query:**
```bash
curl -X POST http://localhost:8000/search \
	-H "Content-Type: application/json" \
	-d '{"query": "quantum materials", "top_k": 5}'
```

**Python Example:**
```python
import requests
resp = requests.post("http://localhost:8000/search", json={"query": "quantum materials", "top_k": 5})
print(resp.json())
```
</details>

## Troubleshooting & Tips
- If you see `Connection refused`, check that the container is running: `docker ps`
- For cloud servers, ensure ports 8000/8501 are open in your security group
- Check logs for errors: `docker logs iisc-chatbot-api-1`
- Edit `.env` to set API keys and index directory if needed

---

A minimal retrieval‑augmented chatbot for IISc content. The pipeline:

```
crawl → clean & dedup → link graph → tree‑smart chunking → FastEmbed embeddings → NumPy index → API + UI (Gemini optional)
```


## Architecture Overview
We have pruned FAISS, SBERT, Chromadb, Ollama, and legacy CDS‑specific code. The remaining stack is small and CPU‑friendly.


| Component      | File(s) | Purpose |
|-------------------|-----------|-----------|
| Crawler           | `scripts/crawl_iisc_full.py` | BFS crawl of `iisc.ac.in` (+ optional github.io external depth) with dedup, robots respect, depth control |
| Preprocess + Dedup| `scripts/preprocess_and_chunk.py` | Clean text, dedup by URL & content, write multiple JSON reports |
| Link Graph        | `scripts/build_link_graph.py` | Build incoming/outgoing link mappings, collapse duplicates |
| Tree Chunking     | `scripts/chunk_pages_tree.py`; `src/data_processing/tree_chunker.py`; `src/utils/url_hierarchy.py` | Content + URL + referential paths → hierarchical chunks with effective context |
| Index Builder     | `scripts/build_index.py`; `src/embeddings/fastembed_embedder.py`; `src/vector_store/numpy_store.py` | Batch embed FastEmbed chunks, store vectors.npz + metadata.jsonl + index_meta.json |
| Retrieval (CLI)   | `scripts/retrieve.py` | Query NumPy index with rerank heuristics |
| API               | `api/main.py` | FastAPI endpoints (`/health`, `/search`, `/query`) loading NumPy index |
| UI                | `ui/streamlit_app.py` | Streamlit chat interface, source citations, optional Gemini generation |
| LLM Wrapper       | `src/llm/gemini.py` | Simple Gemini text generation wrapper |
| Text Cleaning     | `src/data_processing/text_clean.py` | Normalize whitespace, remove boilerplate |
| Helpers           | `src/utils/helpers.py`, `src/utils/url_hierarchy.py` | Utility functions and URL→heading path construction |
| App Control       | `scripts/app_manager.py`, `scripts/run_application.sh`, `scripts/stop_application.sh` | Start/stop API + UI; port cleanup |
| Remote Monitoring | `scripts/remote_poll.sh` | Poll remote index build logs |

## Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:
* `GEMINI_API_KEY` (optional — if absent, UI shows sources only)
* `INDEX_DIR` or `VECTOR_INDEX_DIR` (override automatic discovery)
* `GEMINI_MODEL` (optional; default `models/gemini-1.5-flash`)

## Crawl IISc
```bash
python scripts/crawl_iisc_full.py \
	--url https://iisc.ac.in \
	--depth 8 --same-domain --include-subdomains \
	--external-depth 2 --delay 0.4 \
	--output data/raw/iisc_full.jsonl --format jsonl --verbose
```
Depth counts link hops; `--include-subdomains` lets pages like `cds.iisc.ac.in` be included. Only `github.io` external links are crawled (up to `--external-depth`).

## Preprocess & Deduplicate
```bash
python scripts/preprocess_and_chunk.py --in data/raw/iisc_full.jsonl --input-format jsonl --suffix _iisc_2025nov07
```
Generated (with suffix):
* `cleaned_pages_iisc_2025nov07.json`
* `duplicates_by_url_iisc_2025nov07.json`
* `cleaned_pages_dedup_iisc_2025nov07.json`
* `duplicates_by_content_iisc_2025nov07.json`
* `cleaned_pages_dedup_by_content_iisc_2025nov07.json`
* `duplicate_url_to_canonical_iisc_2025nov07.json`

## Link Graph
```bash
python scripts/build_link_graph.py --suffix _iisc_2025nov07
```
Produces `link_graph_iisc_2025nov07.json` (incoming/outgoing edges, optionally collapsed via canonical mapping).

## Tree‑Smart Chunking
```bash
python scripts/chunk_pages_tree.py --suffix _iisc_2025nov07
```
Output: `chunks_tree_iisc_2025nov07.jsonl` (each line: chunk text, URL, hierarchical paths, referential contexts).

## Build FastEmbed NumPy Index
```bash
python scripts/build_index.py --suffix _iisc_2025nov07 --model BAAI/bge-small-en-v1.5
```
Output dir: `data/index/fastembed_bge_small_iisc_2025nov07/` containing:
* `vectors.npz`
* `metadata.jsonl`
* `index_meta.json` (encoder/model/dim/count)

## CLI Retrieval
```bash
python scripts/retrieve.py --index data/index/fastembed_bge_small_iisc_2025nov07 --q "quantum materials" --k 5
```
Rerank heuristics (additive boosts): exact phrase (+1.0), token overlap (≤ +0.15), path matches (+0.05/+0.03).

## Run Application
Option 1 (shell script):
```bash
bash scripts/run_application.sh
```
Option 2 (manager):
```bash
python scripts/app_manager.py restart
```
Stop:
```bash
bash scripts/stop_application.sh
# or
python scripts/app_manager.py stop
```
Endpoints:
* API: `http://localhost:8000/health`
* UI:  `http://localhost:8501`

## Remote Heavy Index Build (Example)
```bash
# On remote host
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
nohup python scripts/build_index.py --suffix _iisc_2025nov07 > logs/index_build_iisc_2025nov07.log 2>&1 &
tail -f logs/index_build_iisc_2025nov07.log

# After completion, copy back
rsync -av remote_host:/path/to/data/index/fastembed_bge_small_iisc_2025nov07 data/index/
```
Set `.env` `INDEX_DIR` to new directory and restart application.

## Environment Variables Summary
| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Enable Gemini generation in UI |
| `GEMINI_MODEL` | Select Gemini model (default flash) |
| `INDEX_DIR` / `VECTOR_INDEX_DIR` | Override index auto-discovery |
| `API_HOST`, `API_PORT` | Optional API binding overrides |

## Data & Git Hygiene
The `data/` directory is ignored to prevent large binary artifacts in Git. Keep local copies; publish indices via external storage if needed (use Git LFS or release assets for >100MB files).

## Design Choices
* FastEmbed chosen for CPU throughput & small footprint.
* NumPy matrix + metadata JSONL keeps format simple & inspectable.
* Tree‑smart chunking merges URL hierarchy + referential incoming link context.
* Lightweight lexical boosting complements embeddings without extra models.

## Extensibility Ideas
* Parallel crawler (multi-process/thread) respecting politeness & robots.
* Index compression (quantization or float16) for memory reduction.
* Advanced reranking (BM25 fallback or small cross‑encoder) if size constraints allow.

## Contributing
Please open Issues / PRs with clear descriptions. Style: keep scripts single‑purpose, avoid heavy dependencies, prefer pure Python + NumPy.

## License
MIT (add explicit `LICENSE` file if missing before public release).

## Disclaimer
Gemini answers only use provided context; if insufficient, it should respond that more information is needed.
