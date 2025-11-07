# c:\Users\harsh\Documents\chat application\src\vector_store\faiss_store.py
import faiss
import json
import numpy as np
from typing import List, Dict, Tuple


class FaissStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: List[str] = []
        self.metadatas: List[Dict] = []

    def add(self, vectors: np.ndarray, chunks: List[str], metadatas: List[Dict]):
        """Adds vectors and corresponding chunks/metadata to the store."""
        if len(chunks) != len(metadatas) or len(chunks) != len(vectors):
            raise ValueError("vectors, chunks, and metadatas must be same length")
        self.index.add(np.array(vectors, dtype=np.float32))
        self.chunks.extend(chunks)
        self.metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Return top-k (chunk, metadata, distance) tuples."""
        distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
        results: List[Tuple[str, Dict, float]] = []
        for rank, i in enumerate(indices[0]):
            if i < 0 or i >= len(self.chunks):
                continue
            results.append((self.chunks[i], self.metadatas[i], float(distances[0][rank])))
        return results

    def save(self, index_path: str, chunks_path: str, metadata_path: str):
        """Saves the index, chunks, and metadatas to disk."""
        faiss.write_index(self.index, index_path)
        with open(chunks_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.chunks))
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for md in self.metadatas:
                f.write(json.dumps(md, ensure_ascii=False) + '\n')

    def load(self, index_path: str, chunks_path: str, metadata_path: str):
        """Loads the index, chunks, and metadatas from disk."""
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = f.read().splitlines()
        self.metadatas = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.metadatas.append(json.loads(line))
