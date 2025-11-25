"""
Embedding generation using Sentence Transformers
"""

from typing import List, Dict, Any, Union
import numpy as np
from loguru import logger


class EmbeddingGenerator:
    """Generate embeddings for text using Sentence Transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Sentence Transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded Sentence Transformer model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model: {e}")
            self.model = None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to generate embedding for
        
        Returns:
            Embedding vector as numpy array
        """
        if not text or not self.model:
            return np.array([])
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.array([])
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 256, 
                          show_progress: bool = False) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Batch size for processing (optimized for 16GB RAM)
            show_progress: Show progress bar
        
        Returns:
            List of embedding vectors
        """
        if not texts or not self.model:
            return []
        
        try:
            # Optimized batch size for 16GB RAM - 256 for faster processing
            # Truncate very long texts to prevent memory issues
            truncated_texts = [text[:8000] if len(text) > 8000 else text for text in texts]
            
            embeddings = self.model.encode(
                truncated_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Pre-normalize for cosine similarity
                convert_to_tensor=False  # Stay in numpy for faster processing
            )
            
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    def generate_chunk_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks and add to chunk dictionaries
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
        
        Returns:
            List of chunk dictionaries with added 'embedding' field
        """
        if not chunks or not self.model:
            return chunks
        
        try:
            # Extract texts from chunks
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings with larger batch size for speed
            # Disable progress bar in parallel mode to avoid output clutter
            embeddings = self.generate_embeddings(texts, batch_size=128, show_progress=False)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding
            
            logger.debug(f"Generated embeddings for {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            logger.error(f"Error generating chunk embeddings: {e}")
            return chunks
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Reshape for sklearn
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def get_embedding_dimension(self) -> int:
        """
        Get dimension of embeddings produced by the model
        
        Returns:
            Embedding dimension
        """
        if not self.model:
            return 0
        
        return self.model.get_sentence_embedding_dimension()
