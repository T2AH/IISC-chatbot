from typing import List

import numpy as np


class SbertEmbedder:
    """
    Sentence-Transformers embedder wrapper.
    Requires the 'sentence-transformers' package.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            msg = str(e)
            # Common Windows case: Torch DLL load failure (WinError 126)
            if "WinError 126" in msg or "torch_python.dll" in msg or "import torch" in msg:
                raise RuntimeError(
                    "PyTorch backend failed to load (likely missing Microsoft Visual C++ Redistributable). "
                    "Install the x64 VC++ 2015-2022 redistributable and reinstall CPU-only torch, or use --encoder hash."
                ) from e
            raise RuntimeError(
                "Failed to import sentence-transformers. Ensure it is installed and dependencies are satisfied."
            ) from e

        self.model = SentenceTransformer(model_name, device=device)
        try:
            self.dim = int(self.model.get_sentence_embedding_dimension())
        except Exception:
            # Fallback for older versions
            self.dim = int(self.model.encode(["test"]).shape[-1])

    def encode(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, normalize_embeddings=True)
        # ensure float32 numpy vector
        v = np.asarray(vec, dtype=np.float32)
        # If model returns batch shape when given a single string on some versions
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        return v

    def batch_encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
