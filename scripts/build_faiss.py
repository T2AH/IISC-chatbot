import os
import sys
import json
import argparse
from typing import List, Dict

import numpy as np
import faiss

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.embeddings.hash_embedder import HashingEmbedder
from src.embeddings.fastembed_embedder import FastEmbedEmbedder  # optional
try:
    from src.embeddings.sbert_embedder import SbertEmbedder  # optional
except Exception:
    SbertEmbedder = None


DEFAULT_CHUNKS = "data/processed/chunks_tree.jsonl"
DEFAULT_OUT_DIR = "data/index/faiss_flat"


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norms


def main():
    p = argparse.ArgumentParser(description="Build a FAISS index from chunks JSONL")
    p.add_argument("--chunks", type=str, default=DEFAULT_CHUNKS)
    p.add_argument("--out", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--encoder", type=str, default="hash", choices=["hash", "fastembed", "sbert"])
    p.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--limit", type=int, default=0, help="Limit number of chunks (0=all)")
    p.add_argument("--batch", type=int, default=int(os.environ.get("EMBED_BATCH", "256")))
    args = p.parse_args()

    # Select embedder
    if args.encoder == "fastembed":
        emb = FastEmbedEmbedder(model=args.model or "BAAI/bge-small-en-v1.5")
        dim = emb.dim
    elif args.encoder == "sbert":
        if SbertEmbedder is None:
            print("SBERT not available. Install sentence-transformers or use --encoder fastembed/hash.")
            sys.exit(2)
        emb = SbertEmbedder(model_name=args.model)
        dim = emb.dim
    else:
        emb = HashingEmbedder(dim=2048)
        dim = emb.dim

    os.makedirs(args.out, exist_ok=True)
    chunks_path = os.path.join(args.out, "chunks.txt")
    meta_path = os.path.join(args.out, "metadata.jsonl")
    index_path = os.path.join(args.out, "faiss.index")
    meta_info_path = os.path.join(args.out, "index_meta.json")

    vectors: List[np.ndarray] = []
    metas: List[Dict] = []
    texts: List[str] = []
    seen = 0

    print(f"[faiss-build] Encoder={args.encoder} dim={dim} batch={args.batch}")
    print(f"[faiss-build] Reading chunks: {args.chunks}")

    def flush_batch(batch_texts: List[str], batch_meta: List[Dict]):
        nonlocal vectors
        if not batch_texts:
            return
        if hasattr(emb, "batch_encode"):
            try:
                embs = emb.batch_encode(batch_texts, batch_size=args.batch)
                if not isinstance(embs, list):
                    embs = list(embs)
            except TypeError:
                embs = list(emb.batch_encode(batch_texts))
        else:
            embs = [emb.encode(t) for t in batch_texts]
        for v in embs:
            vectors.append(np.asarray(v, dtype=np.float32))
        metas.extend(batch_meta)

    # Read
    with open(args.chunks, "r", encoding="utf-8") as f:
        buf_t: List[str] = []
        buf_m: List[Dict] = []
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            seen += 1
            if args.limit and seen > args.limit:
                break
            md = {
                "url": rec.get("url"),
                "title": rec.get("title"),
                "effective_path": rec.get("effective_path"),
                "heading_path": rec.get("heading_path"),
                "chunk_index": rec.get("chunk_index"),
                "page_index": rec.get("page_index"),
                "text": text[:1200],
            }
            buf_t.append(text)
            buf_m.append(md)
            texts.append(text)
            if len(buf_t) >= args.batch:
                flush_batch(buf_t, buf_m)
                buf_t, buf_m = [], []
        flush_batch(buf_t, buf_m)

    if not vectors:
        print("[faiss-build] No vectors to index.")
        sys.exit(1)

    mat = np.vstack(vectors).astype(np.float32)
    # normalize for cosine (use IP index)
    mat = l2_normalize(mat)

    print(f"[faiss-build] Building IndexFlatIP with {mat.shape[0]} vectors, dim={dim}")
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    # Save index
    faiss.write_index(index, index_path)
    with open(chunks_path, "w", encoding="utf-8") as cf:
        cf.write("\n".join(texts))
    with open(meta_path, "w", encoding="utf-8") as mf:
        for m in metas:
            mf.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(meta_info_path, "w", encoding="utf-8") as jf:
        json.dump({
            "encoder": args.encoder,
            "model": args.model,
            "dim": int(dim),
            "count": int(mat.shape[0])
        }, jf)

    print(f"[faiss-build] DONE -> {args.out} | count={mat.shape[0]} dim={dim}")


if __name__ == "__main__":
    main()
