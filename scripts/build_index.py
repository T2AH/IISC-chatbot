import os
import sys
import json
import argparse

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.embeddings.hash_embedder import HashingEmbedder
from src.vector_store.numpy_store import NumpyVectorStore
try:
    from src.embeddings.sbert_embedder import SbertEmbedder  # optional
except Exception:
    SbertEmbedder = None
try:
    from src.embeddings.fastembed_embedder import FastEmbedEmbedder  # optional
except Exception:
    FastEmbedEmbedder = None


DEFAULT_CHUNKS = "data/processed/chunks_tree.jsonl"
DEFAULT_INDEX_DIR = "data/index/hash_numpy"


def main():
    parser = argparse.ArgumentParser(description="Build a NumPy-based index over chunks JSONL")
    parser.add_argument("--chunks", type=str, default=DEFAULT_CHUNKS, help="Path to chunks JSONL")
    parser.add_argument("--out", type=str, default=DEFAULT_INDEX_DIR, help="Output index directory")
    parser.add_argument("--dim", type=int, default=2048, help="Embedding dimension (hash mode)")
    parser.add_argument("--encoder", type=str, default="hash", choices=["hash", "sbert", "fastembed"], help="Embedding encoder")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Model name (sbert=HF model, fastembed=bge/e5 model)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of chunks to index (0 = no limit)")
    args = parser.parse_args()

    if not os.path.exists(args.chunks):
        print(f"Chunks file not found: {args.chunks}")
        sys.exit(1)

    print(f"[builder] Encoder: {args.encoder}")
    if args.encoder == "sbert":
        if SbertEmbedder is None:
            print("SBERT encoder requested but sentence-transformers is not available. Install it or use --encoder hash.")
            sys.exit(2)
        print(f"[builder] Loading SBERT model: {args.model} (this may take a while on first run)...", flush=True)
        embedder = SbertEmbedder(model_name=args.model)
        store = NumpyVectorStore(dim=embedder.dim)
    elif args.encoder == "fastembed":
        if FastEmbedEmbedder is None:
            print("FastEmbed encoder requested but fastembed is not available. Install it or use --encoder hash.")
            sys.exit(2)
        name = args.model or "BAAI/bge-small-en-v1.5"
        print(f"[builder] Loading FastEmbed model: {name} (first run downloads ONNX, please wait)...", flush=True)
        embedder = FastEmbedEmbedder(model=name)
        store = NumpyVectorStore(dim=embedder.dim)
    else:
        embedder = HashingEmbedder(dim=args.dim)
        store = NumpyVectorStore(dim=args.dim)
    
    vectors: list = []
    meta: list = []
    total = 0
    BATCH = int(os.environ.get("EMBED_BATCH", "128"))
    print(f"[builder] Embedding dim: {store.dim} | Batch size: {BATCH}")

    def flush_batch(text_batch, rec_batch):
        nonlocal vectors, meta, total
        if not text_batch:
            return
        # Prefer batch_encode if available
        if hasattr(embedder, "batch_encode"):
            try:
                embs = embedder.batch_encode(text_batch, batch_size=BATCH)  # SBERT signature
                # fastembed returns generator if used directly, our wrapper returns whatever model returns
                if not isinstance(embs, list):
                    embs = list(embs)
            except TypeError:
                # fastembed wrapper batch_encode without batch_size
                embs = list(embedder.batch_encode(text_batch))
        else:
            embs = [embedder.encode(t) for t in text_batch]

        for v, rec in zip(embs, rec_batch):
            vectors.append(v)
            meta.append(rec)
            total += 1
        if total % 1000 == 0:
            print(f"[builder] Processed {total} chunks...", flush=True)

    text_buf = []
    rec_buf = []
    seen = 0
    print(f"[builder] Reading chunks from {args.chunks} ...")
    with open(args.chunks, "r", encoding="utf-8") as f:
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
            # Prepare metadata record
            md = {
                "url": rec.get("url"),
                "title": rec.get("title"),
                "effective_path": rec.get("effective_path"),
                "heading_path": rec.get("heading_path"),
                "chunk_index": rec.get("chunk_index"),
                "page_index": rec.get("page_index"),
                "text": text[:1200],
            }
            text_buf.append(text)
            rec_buf.append(md)
            if len(text_buf) >= BATCH:
                flush_batch(text_buf, rec_buf)
                text_buf, rec_buf = [], []

    # Flush remaining
    flush_batch(text_buf, rec_buf)

    store.add(vectors, meta)
    store.save(args.out)

    # Write simple index metadata
    try:
        meta_path = os.path.join(args.out, "index_meta.json")
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump({
                "encoder": args.encoder,
                "model": args.model,
                "dim": int(store.dim),
                "count": int(total),
            }, mf)
    except Exception:
        pass

    print(f"[builder] DONE: {total} chunks -> {args.out} (encoder={args.encoder}, dim={store.dim})")


if __name__ == "__main__":
    main()
