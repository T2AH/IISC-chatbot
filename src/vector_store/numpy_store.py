import json
import os
from typing import List, Dict, Any, Tuple

import numpy as np


class NumpyVectorStore:
	"""
	Minimal vector store using NumPy arrays and cosine similarity.
	Persists to NPZ (vectors) and JSONL (metadata records).
	"""

	def __init__(self, dim: int):
		self.dim = dim
		self._vectors: np.ndarray | None = None  # shape (n, dim)
		self._meta: List[Dict[str, Any]] = []

	@staticmethod
	def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
		denom = (np.linalg.norm(a) * np.linalg.norm(b))
		if denom == 0:
			return 0.0
		return float(np.dot(a, b) / denom)

	def add(self, vectors: List[np.ndarray], meta: List[Dict[str, Any]]):
		if not vectors:
			return
		mat = np.vstack(vectors).astype(np.float32)
		if self._vectors is None:
			self._vectors = mat
		else:
			self._vectors = np.vstack([self._vectors, mat])
		self._meta.extend(meta)

	def save(self, dirpath: str):
		os.makedirs(dirpath, exist_ok=True)
		vec_path = os.path.join(dirpath, "vectors.npz")
		meta_path = os.path.join(dirpath, "metadata.jsonl")
		if self._vectors is None:
			np.savez_compressed(vec_path, vectors=np.zeros((0, self.dim), dtype=np.float32))
		else:
			np.savez_compressed(vec_path, vectors=self._vectors)
		with open(meta_path, "w", encoding="utf-8") as f:
			for rec in self._meta:
				f.write(json.dumps(rec, ensure_ascii=False) + "\n")

	def load(self, dirpath: str):
		vec_path = os.path.join(dirpath, "vectors.npz")
		meta_path = os.path.join(dirpath, "metadata.jsonl")
		if os.path.exists(vec_path):
			data = np.load(vec_path)
			self._vectors = data["vectors"].astype(np.float32)
		else:
			self._vectors = np.zeros((0, self.dim), dtype=np.float32)
		self._meta = []
		if os.path.exists(meta_path):
			with open(meta_path, "r", encoding="utf-8") as f:
				for line in f:
					try:
						self._meta.append(json.loads(line))
					except Exception:
						continue

	def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
		if self._vectors is None or self._vectors.shape[0] == 0:
			return []
		# compute cosine similarities efficiently
		A = self._vectors
		# Normalize rows once for faster cosine
		norms = np.linalg.norm(A, axis=1)
		qn = np.linalg.norm(query)
		if qn == 0:
			sims = np.zeros(A.shape[0], dtype=np.float32)
		else:
			dots = A @ query
			sims = dots / (norms * qn + 1e-8)
		# top-k
		k = min(k, sims.shape[0])
		idx = np.argpartition(-sims, k - 1)[:k]
		# sort by score desc
		idx = idx[np.argsort(-sims[idx])]
		return [(float(sims[i]), self._meta[i]) for i in idx]
