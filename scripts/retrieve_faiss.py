import os
import sys
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import faiss

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.embeddings.hash_embedder import HashingEmbedder
try:
    from src.embeddings.fastembed_embedder import FastEmbedEmbedder
except Exception:
    FastEmbedEmbedder = None
try:
    from src.embeddings.sbert_embedder import SbertEmbedder
except Exception:
    SbertEmbedder = None


DEFAULT_INDEX_DIR = "data/index/faiss_flat"


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Query a FAISS IP index with cosine-normalized vectors")
    p.add_argument("--index", type=str, default=DEFAULT_INDEX_DIR)
    p.add_argument("--q", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--encoder", type=str, default="hash", choices=["hash", "fastembed", "sbert"])
    p.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args()

    # Load index and metadata
    index_path = os.path.join(args.index, "faiss.index")
    meta_path = os.path.join(args.index, "metadata.jsonl")
    if not os.path.exists(index_path):
        print(f"Index not found: {index_path}")
        sys.exit(1)

    index = faiss.read_index(index_path)

    # Load embedder
    if args.encoder == "fastembed":
        if FastEmbedEmbedder is None:
            print("FastEmbed not available. Install fastembed or use --encoder hash")
            sys.exit(2)
        emb = FastEmbedEmbedder(model=args.model or "BAAI/bge-small-en-v1.5")
    elif args.encoder == "sbert":
        if SbertEmbedder is None:
            print("SBERT not available. Install sentence-transformers or use --encoder hash")
            sys.exit(2)
        emb = SbertEmbedder(model_name=args.model)
    else:
        emb = HashingEmbedder(dim=2048)

    q = l2_normalize(emb.encode(args.q))
    D, I = index.search(np.expand_dims(q, 0), args.k)

    # Load metadata lines
    metas: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                metas.append(json.loads(line))
            except Exception:
                metas.append({})

    out = []
    for rank, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(metas):
            continue
        md = metas[idx]
        out.append({
            "score": float(D[0][rank]),
            "url": md.get("url"),
            "title": md.get("title"),
            "snippet": (md.get("text") or "")[:300],
            "path": [seg.get("text") for seg in (md.get("effective_path") or [])],
        })

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
