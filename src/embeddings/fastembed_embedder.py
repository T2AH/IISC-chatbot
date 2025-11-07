from typing import List

import numpy as np


class FastEmbedEmbedder:
    """
    FastEmbed wrapper (CPU-only, no Torch). Downloads model on first use.
    Default model: BAAI/bge-small-en-v1.5 (good balance of quality/speed).
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except Exception as e:
            raise RuntimeError(
                "fastembed is not installed. Please `pip install fastembed` or run requirements install."
            ) from e
        self.TextEmbedding = TextEmbedding
        self.model_name = model
        # Initialize model once; fastembed caches models under user cache dir
        self._model = self.TextEmbedding(model_name=self.model_name)
        # Get dimension by encoding a small sample
        v = self.encode("test")
        self.dim = int(v.shape[-1])

    def encode(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        # TextEmbedding.embed returns a generator of np arrays
        arr = next(self._model.embed([text]))
        # L2 normalize
        arr = np.asarray(arr, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def batch_encode(self, texts: List[str]):
        # Returns a generator; convert to numpy array if needed by caller
        return self._model.embed(texts)
