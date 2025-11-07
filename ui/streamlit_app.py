import os
import sys

# Ensure project root is on sys.path so `src.*` imports work when launched via Streamlit
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import json
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
from dotenv import load_dotenv
import requests

# Local imports
from src.llm.gemini import Gemini
from src.embeddings.hash_embedder import HashingEmbedder
try:
    from src.embeddings.fastembed_embedder import FastEmbedEmbedder
except Exception:
    FastEmbedEmbedder = None  # type: ignore

# Vector stores/loaders
from src.vector_store.numpy_store import NumpyVectorStore

# FAISS support removed for minimal NumPy-only runtime.


st.set_page_config(page_title="IISc Chatbot", page_icon="💬", layout="wide")

# Load environment variables from a .env file if present
load_dotenv()


def discover_default_index() -> str:
    """Return default NumPy index directory based on existing folders."""
    # Prefer the dated IISc index first (2025nov07), then general fastembed, then hash fallback.
    candidates = [
        "data/index/fastembed_bge_small_iisc_2025nov07",
        "data/index/fastembed_bge_small",
        "data/index/hash_numpy",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "data/index/hash_numpy"


def discover_numpy_indexes() -> List[str]:
    """Scan data/index for NumPy index layouts (vectors.npz)."""
    base = "data/index"
    out: List[str] = []
    if not os.path.isdir(base):
        return out
    try:
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name)
            if not os.path.isdir(p):
                continue
            if os.path.exists(os.path.join(p, "vectors.npz")):
                out.append(p)
    except Exception:
        pass
    return out


def read_index_meta(index_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(index_dir, "index_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@st.cache_resource(show_spinner=False)
def load_numpy_store(index_dir: str) -> NumpyVectorStore:
    store = NumpyVectorStore(dim=2048)
    store.load(index_dir)
    # If vectors exist, adjust dim
    if store._vectors is not None and store._vectors.size > 0:
        store.dim = int(store._vectors.shape[1])
    return store




@st.cache_resource(show_spinner=False)
def get_embedder(encoder: str, model: Optional[str], dim_hint: int) -> Any:
    if encoder == "fastembed":
        if FastEmbedEmbedder is None:
            raise RuntimeError("fastembed not installed. Install fastembed or use encoder=hash.")
        name = model or "BAAI/bge-small-en-v1.5"
        return FastEmbedEmbedder(model=name)
    return HashingEmbedder(dim=dim_hint or 2048)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    # Retained for compatibility; NumPy backend already normalizes embeddings.
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v.astype(np.float32)


def build_prompt(user_query: str, contexts: List[Dict[str, Any]]) -> str:
    header = (
        "You are a helpful assistant for the IISc Computational & Data Sciences (CDS) department.\n"
        "Answer the user's question using ONLY the context provided. If not enough information, say so.\n"
        "Cite sources inline like [1], [2] corresponding to the sources list.\n\n"
        "Context:\n"
    )
    ctx_lines = []
    for i, c in enumerate(contexts, 1):
        title = c.get("title") or c.get("url") or f"Source {i}"
        url = c.get("url") or ""
        text = (c.get("text") or c.get("snippet") or "")[:1000]
        ctx_lines.append(f"[{i}] {title}\nURL: {url}\n{text}\n")
    prompt = header + "\n".join(ctx_lines)
    prompt += f"\nUser Question: {user_query}\n\nAnswer:"
    return prompt


def retrieve_numpy(index_dir: str, encoder: str, model: Optional[str], query: str, k: int) -> List[Dict[str, Any]]:
    store = load_numpy_store(index_dir)
    emb = get_embedder(encoder, model, store.dim)
    # Validate embedding dimension vs index
    try:
        probe = emb.encode("dimension check")
        if probe.shape[-1] != store.dim:
            raise ValueError(
                f"Encoder/model dim {probe.shape[-1]} != index dim {store.dim}. "
                "Select the correct encoder for this index (fastembed for BGE, hash for hash index)."
            )
    except Exception as e:
        raise RuntimeError(f"Embedding initialization failed: {e}")
    q = emb.encode(query)
    hits = store.search(q, k=k)
    out = []
    for score, meta in hits:
        out.append({
            "score": float(score),
            "url": meta.get("url"),
            "title": meta.get("title"),
            "text": meta.get("text"),
            "path": [seg.get("text") for seg in (meta.get("effective_path") or [])],
        })
    return out




def render_sources(sources: List[Dict[str, Any]]):
    if not sources:
        return
    st.markdown("### Sources")
    for i, s in enumerate(sources, 1):
        title = s.get("title") or s.get("url") or f"Source {i}"
        url = s.get("url") or ""
        st.markdown(f"[{i}] [{title}]({url})")


def main():
    st.title("IISc Chatbot 💬")
    st.caption("Semantic search over IISc content with optional Gemini reasoning (branded as 'IISc Chatbot')")

    # Sidebar: config
    with st.sidebar:
        st.header("Settings")
        default_index = discover_default_index()
        discovered = discover_numpy_indexes()
        choices = ["<custom path>"] + discovered
        pick = st.selectbox("Index directory", choices, index=(choices.index(default_index) if default_index in choices else 0))
        if pick == "<custom path>":
            index_dir = st.text_input("Custom NumPy index path", value=default_index)
        else:
            index_dir = pick
        meta = read_index_meta(index_dir)
        # Try to auto-set encoder/model from meta
        detected_encoder = meta.get("encoder") or ("fastembed" if "bge" in index_dir else "hash")
        encoder = st.selectbox("Encoder", ["fastembed", "hash"], index=(0 if detected_encoder == "fastembed" else 1))
        model = st.text_input("Model (fastembed)", value=(meta.get("model") or "BAAI/bge-small-en-v1.5"))
        # Show index metadata hints
        idx_dim = None
        try:
            s = load_numpy_store(index_dir)
            idx_dim = s.dim
        except Exception:
            pass
        if meta:
            st.caption(f"Index meta: encoder={meta.get('encoder','?')} model={meta.get('model','?')} count={meta.get('count','?')}")
        if idx_dim:
            st.caption(f"Index dim: {idx_dim}")
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)
        # Gemini fixed configuration (no UI inputs): always use gemini-2.5-flash, key from environment only.
        gemini_model = "models/gemini-2.5-flash"
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            st.caption("Gemini backend: Gemini 2.5 Flash (key hidden)")
        else:
            st.caption("Gemini backend unavailable (no GEMINI_API_KEY set). Showing sources only.")

        # API connectivity check
        st.divider()
        st.subheader("API Server")
        api_base = st.text_input("API Base URL", value=os.environ.get("API_BASE_URL", "http://localhost:8000"))
        health_url = api_base.rstrip("/") + "/health"
        api_status = "unknown"
        try:
            r = requests.get(health_url, timeout=3)
            if r.ok:
                data = r.json()
                api_status = data.get("status", "ok")
                st.success(f"Connected: {api_status}")
            else:
                st.warning(f"Unhealthy: HTTP {r.status_code}")
        except Exception as e:
            st.error(f"Not reachable: {e}")

    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_query = st.chat_input("Ask about IISc CDS…")
    if not user_query:
        return

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieval
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer…"):
            try:
                contexts = retrieve_numpy(index_dir, encoder, model, user_query, top_k)
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                return

            # Build prompt and call Gemini
            if api_key:
                try:
                    llm = Gemini(api_key=api_key, model_name=gemini_model)
                    prompt = build_prompt(user_query, contexts)
                    answer = llm.generate(prompt)
                    st.caption("Model: Gemini 2.5 Flash (IISc Chatbot)")
                except Exception as e:
                    st.error(f"Gemini error: {e}")
                    answer = "(Failed to generate with Gemini. Showing sources instead.)\n\n" + "\n\n".join([f"[{i+1}] {(c.get('title') or c.get('url') or 'Source')} — {(c.get('text') or '')[:200]}…" for i, c in enumerate(contexts[:3])])
            else:
                answer = "\n\n".join([f"[{i+1}] {(c.get('title') or c.get('url') or 'Source')} — {(c.get('text') or '')[:200]}…" for i, c in enumerate(contexts[:3])])

            st.markdown(answer)
            render_sources(contexts[:5])
            st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
