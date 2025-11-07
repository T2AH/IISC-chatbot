import os
import sys
import json
import argparse
from typing import List

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from src.embeddings.hash_embedder import HashingEmbedder
from src.vector_store.numpy_store import NumpyVectorStore
try:
	from src.embeddings.sbert_embedder import SbertEmbedder
except Exception:
	SbertEmbedder = None
try:
	from src.embeddings.fastembed_embedder import FastEmbedEmbedder
except Exception:
	FastEmbedEmbedder = None


DEFAULT_INDEX_DIR = "data/index/hash_numpy"


def flatten_path_text(path: List[dict]) -> str:
	try:
		return " ".join([seg.get("text", "") for seg in (path or [])]).lower()
	except Exception:
		return ""


def rerank(results, query: str) -> list:
	ql = (query or "").strip().lower()
	toks = [t for t in ql.split() if t]
	boosted = []
	for score, meta in results:
		text = (meta.get("text") or "").lower()
		ep_text = flatten_path_text(meta.get("effective_path") or [])
		hp_text = flatten_path_text(meta.get("heading_path") or [])
		boost = 0.0
		if ql and ql in text:
			boost += 1.00
		if toks:
			present = sum(1 for t in toks if t in text)
			boost += min(present, 3) * 0.05
		if toks and any(t in ep_text for t in toks):
			boost += 0.05
		if toks and any(t in hp_text for t in toks):
			boost += 0.03
		boosted.append((float(score) + boost, meta))
	boosted.sort(key=lambda x: x[0], reverse=True)
	return boosted


def exact_phrase_in_results(results, query: str) -> bool:
	ql = (query or "").strip().lower()
	if not ql:
		return True
	for _s, meta in results:
		text = (meta.get("text") or "").lower()
		if ql in text:
			return True
	return False


def main():
	parser = argparse.ArgumentParser(description="Query NumPy-based index")
	parser.add_argument("--index", type=str, default=DEFAULT_INDEX_DIR, help="Index directory")
	parser.add_argument("--q", type=str, required=True, help="Query text")
	parser.add_argument("--k", type=int, default=5, help="Top K results")
	parser.add_argument("--encoder", type=str, default="hash", choices=["hash", "sbert", "fastembed"], help="Encoder used to build the index")
	parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Model name (sbert HF, fastembed bge/e5)")
	parser.add_argument("--candidates", type=int, default=None, help="Candidate pool size before reranking (defaults to max(50, 10*k))")
	args = parser.parse_args()

	# Load index
	# For search, we don't know dim from disk directly; default to 2048 for hash, or infer from SBERT
	if args.encoder == "sbert":
		if SbertEmbedder is None:
			print("SBERT encoder requested but not available. Install sentence-transformers or use --encoder hash.")
			sys.exit(2)
		emb = SbertEmbedder(model_name=args.model)
		store = NumpyVectorStore(dim=emb.dim)
	elif args.encoder == "fastembed":
		if FastEmbedEmbedder is None:
			print("FastEmbed encoder requested but not available. Install fastembed or use --encoder hash.")
			sys.exit(2)
		emb = FastEmbedEmbedder(model=args.model)
		store = NumpyVectorStore(dim=emb.dim)
	else:
		emb = HashingEmbedder(dim=2048)
		store = NumpyVectorStore(dim=2048)
	store.load(args.index)
	qvec = emb.encode(args.q)
	pool_k = args.candidates or max(50, args.k * 10)
	results = store.search(qvec, k=pool_k)
	results = rerank(results, args.q)
	results = results[: args.k]

	out = []
	for sc, meta in results:
		out.append({
			"score": round(float(sc), 4),
			"url": meta.get("url"),
			"title": meta.get("title"),
			"snippet": meta.get("text"),
			"path": [seg.get("text") for seg in (meta.get("effective_path") or [])],
		})
	print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()

