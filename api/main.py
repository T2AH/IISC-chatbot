"""FastAPI backend for IISc CDS chatbot: search/query over prebuilt NumPy index.

This version loads the lightweight NumPy vector store created by
`scripts/build_index.py` and exposes endpoints used by the Streamlit UI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv

# Local imports
import sys
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.vector_store.numpy_store import NumpyVectorStore
from src.embeddings.hash_embedder import HashingEmbedder
try:
    from src.embeddings.fastembed_embedder import FastEmbedEmbedder  # optional
except Exception:  # pragma: no cover
    FastEmbedEmbedder = None  # type: ignore


# Environment and logging
load_dotenv()
os.makedirs("logs", exist_ok=True)
logger.add("logs/api.log", rotation="10 MB")


class QueryRequest(BaseModel):
    text: str
    top_k: int = 5
    encoder: str = "hash"  # "hash" or "fastembed"
    model: Optional[str] = None  # fastembed model name if encoder==fastembed


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    encoder: str = "hash"
    model: Optional[str] = None
    # Optional: internal candidate pool size before rerank; if not provided, a heuristic is used
    candidates: Optional[int] = None


class SearchHit(BaseModel):
    url: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    score: float
    path: Optional[List[str]] = None


app = FastAPI(title="IISc Chatbot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global index state (prefer IISc 2025nov07 index by default)
INDEX_DIR = os.environ.get("INDEX_DIR", "data/index/fastembed_bge_small_iisc_2025nov07")
_store: Optional[NumpyVectorStore] = None
_hash_embedder: Optional[HashingEmbedder] = None
_fastembed: Any = None


def _get_embedder(encoder: str, model: Optional[str]):
    if encoder == "fastembed":
        if FastEmbedEmbedder is None:
            raise HTTPException(status_code=400, detail="fastembed not available. Use encoder=hash or install fastembed.")
        name = model or "BAAI/bge-small-en-v1.5"
        return FastEmbedEmbedder(model=name)
    # default
    return _hash_embedder or HashingEmbedder(dim=_store.dim if _store else 2048)


def _flatten_path_text(path: Optional[List[dict]]) -> str:
    try:
        return " ".join([seg.get("text", "") for seg in (path or [])]).lower()
    except Exception:
        return ""


def _rerank_boost(hits: List, query: str) -> List:
    """Apply lightweight lexical boosts so proper-name and exact-phrase queries surface.

    Boosting strategy (additive on cosine score):
    - +1.00 if exact lowercase query substring is in the chunk text (names/phrases)
    - +0.05 per token present in chunk text (up to +0.15)
    - +0.05 if any token present in effective_path
    - +0.03 if any token present in heading_path
    """
    ql = (query or "").strip().lower()
    if not ql:
        return hits
    toks = [t for t in ql.split() if t]
    boosted = []
    for score, meta in hits:
        text = (meta.get("text") or "").lower()
        ep_text = _flatten_path_text(meta.get("effective_path"))
        hp_text = _flatten_path_text(meta.get("heading_path"))
        boost = 0.0
        # Exact phrase in text
        if ql and ql in text:
            boost += 1.00
        # Token matches in body text
        if toks:
            present = sum(1 for t in toks if t in text)
            boost += min(present, 3) * 0.05
        # Path/headline hints
        if toks and any(t in ep_text for t in toks):
            boost += 0.05
        if toks and any(t in hp_text for t in toks):
            boost += 0.03
        boosted.append((float(score) + boost, meta))
    boosted.sort(key=lambda x: x[0], reverse=True)
    return boosted


def _exact_phrase_in_hits(hits: List, query: str) -> bool:
    ql = (query or "").strip().lower()
    if not ql:
        return True
    for _s, meta in hits:
        text = (meta.get("text") or "").lower()
        if ql in text:
            return True
    return False


def _lexical_fallback_exact(store: NumpyVectorStore, query: str, limit: int = 5) -> List:
    ql = (query or "").strip().lower()
    if not ql:
        return []
    out: List = []
    meta_list = getattr(store, "_meta", []) or []
    for meta in meta_list:
        text = (meta.get("text") or "").lower()
        if ql in text:
            # synthetic high score to place above semantic-only hits
            out.append((2.0, meta))
            if len(out) >= limit:
                break
    return out


@app.on_event("startup")
def _startup():
    global _store, _hash_embedder
    logger.info(f"Loading NumPy index from {INDEX_DIR} ...")
    _store = NumpyVectorStore(dim=2048)
    _store.load(INDEX_DIR)
    # If vectors exist, set dim based on loaded matrix to avoid mismatches
    if _store._vectors is not None and _store._vectors.size > 0:
        _store.dim = int(_store._vectors.shape[1])
    _hash_embedder = HashingEmbedder(dim=_store.dim)
    logger.info("Index loaded: vectors=%s dim=%s",
                int(_store._vectors.shape[0]) if _store and _store._vectors is not None else 0,
                int(_store.dim) if _store else None)


@app.get("/")
def root():
    return {"message": "IISc Chatbot API", "version": "1.0.0", "model_alias": "IISc Chatbot"}


@app.get("/health")
def health():
    loaded = _store is not None and _store._vectors is not None and _store._vectors.shape[0] > 0
    return {
        "status": "healthy" if loaded else "empty",
        "index_dir": INDEX_DIR,
        "num_vectors": int(_store._vectors.shape[0]) if _store and _store._vectors is not None else 0,
        "dim": int(_store.dim) if _store else None,
    }


@app.post("/search")
def search(req: SearchRequest) -> Dict[str, Any]:
    if _store is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    emb = _get_embedder(req.encoder, req.model)
    q = emb.encode(req.query)
    # Fetch a larger candidate pool to allow lexical reranker to surface precise matches
    pool_k = req.candidates or max(50, req.top_k * 10)
    hits = _store.search(q, k=pool_k)
    hits = _rerank_boost(hits, req.query)
    # If none of the hits contain the exact phrase, prepend lexical exact matches
    if not _exact_phrase_in_hits(hits, req.query):
        lex = _lexical_fallback_exact(_store, req.query, limit=min(5, req.top_k))
        # De-duplicate by URL to avoid repeats
        seen = set([m.get("url") for _, m in lex])
        rest = [(s, m) for (s, m) in hits if m.get("url") not in seen]
        hits = (lex + rest)[: req.top_k]
    else:
        hits = hits[: req.top_k]
    out = []
    for score, meta in hits:
        out.append(
            SearchHit(
                url=meta.get("url"),
                title=meta.get("title"),
                snippet=(meta.get("text") or "")[:300],
                score=float(score),
                path=[seg.get("text") for seg in (meta.get("effective_path") or [])],
            ).model_dump()
        )
    return {"results": out}


@app.post("/query/")
def query(req: QueryRequest) -> Dict[str, Any]:
    """Simple QA placeholder that returns top snippets; aligns with Streamlit UI."""
    if _store is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    emb = _get_embedder(req.encoder, req.model)
    q = emb.encode(req.text)
    pool_k = max(50, req.top_k * 10)
    hits = _store.search(q, k=pool_k)
    hits = _rerank_boost(hits, req.text)
    if not _exact_phrase_in_hits(hits, req.text):
        lex = _lexical_fallback_exact(_store, req.text, limit=min(5, req.top_k))
        seen = set([m.get("url") for _, m in lex])
        rest = [(s, m) for (s, m) in hits if m.get("url") not in seen]
        hits = (lex + rest)[: req.top_k]
    else:
        hits = hits[: req.top_k]
    if not hits:
        return {"response": "No relevant results found in CDS dataset."}
    # Concatenate top snippets as a basic answer (no LLM here)
    snippets = []
    for score, meta in hits[:3]:
        title = meta.get("title") or meta.get("url") or ""
        snippets.append(f"[{title}] {meta.get('text','')[:200]}...")
    answer = " \n\n".join(snippets)
    return {"response": answer}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
