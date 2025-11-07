import re
import numpy as np
from typing import List


class HashingEmbedder:
    """
    Lightweight embedding using the hashing trick (feature hashing) over tokens.
    - No external deps beyond numpy
    - Deterministic given dim and seed
    - Normalizes vectors to unit length
    """

    def __init__(self, dim: int = 2048, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self._token_re = re.compile(r"[A-Za-z0-9_]+")

    def _tokens(self, text: str) -> List[str]:
        if not text:
            return []
        return [t.lower() for t in self._token_re.findall(text)]

    def _hash(self, token: str) -> int:
        # Stable Python hash across runs by incorporating seed explicitly
        # Use a simple FNV-1a like mix for cross-run stability
        h = 2166136261 + self.seed
        for ch in token:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = self._tokens(text)
        if not tokens:
            return vec
        for tok in tokens:
            if len(tok) < 2:
                continue
            idx = self._hash(tok) % self.dim
            vec[idx] += 1.0
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
