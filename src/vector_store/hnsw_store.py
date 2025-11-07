import json
import os
from typing import List, Dict, Tuple

from annoy import AnnoyIndex
import numpy as np


class HnswStore:
    """Annoy-backed store (angular distance) as a Windows-friendly alternative to FAISS/HNSWLIB."""

    def __init__(self, dimension: int, metric: str = "angular", n_trees: int = 50):
        self.dimension = dimension
        self.metric = metric
        self.index = AnnoyIndex(dimension, metric)
        self.next_id = 0
        self.built = False
        self.n_trees = n_trees
        self.chunks: List[str] = []
        self.metadatas: List[Dict] = []

    def add(self, vectors: np.ndarray, chunks: List[str], metadatas: List[Dict]):
        if len(chunks) != len(metadatas) or len(chunks) != len(vectors):
            raise ValueError("vectors, chunks, and metadatas must be same length")
        for vec, ch, md in zip(vectors, chunks, metadatas):
            self.index.add_item(self.next_id, vec.tolist())
            self.chunks.append(ch)
            self.metadatas.append(md)
            self.next_id += 1
        # Rebuild index when new items are added
        self.index.build(self.n_trees)
        self.built = True

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, Dict, float]]:
        if not self.built:
            self.index.build(self.n_trees)
            self.built = True
        ids, distances = self.index.get_nns_by_vector(query_vector.tolist(), k, include_distances=True)
        results: List[Tuple[str, Dict, float]] = []
        for idx, dist in zip(ids, distances):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], self.metadatas[idx], float(dist)))
        return results

    def save(self, index_path: str, chunks_path: str, metadata_path: str):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.index.save(index_path)
        with open(chunks_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.chunks))
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for md in self.metadatas:
                f.write(json.dumps(md, ensure_ascii=False) + '\n')

    def load(self, index_path: str, chunks_path: str, metadata_path: str):
        self.index = AnnoyIndex(self.dimension, self.metric)
        self.index.load(index_path)
        self.built = True
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = f.read().splitlines()
        self.metadatas = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.metadatas.append(json.loads(line))
