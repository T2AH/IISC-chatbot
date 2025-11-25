"""
Query embedding generator using OpenAI text-embedding-3-large
"""
from typing import List
from openai import OpenAI
from loguru import logger
import os

class QueryEmbedder:
    """Generate query embeddings using OpenAI"""
    
    def __init__(self, api_key: str = None):
        """Initialize OpenAI client"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-large"
        self.dimensions = 3072
        
        logger.info(f"QueryEmbedder initialized with {self.model} ({self.dimensions}D)")
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single query
        
        Args:
            text: Query text
        
        Returns:
            Embedding vector (3072 dimensions)
        """
        try:
            # Truncate if too long
            if len(text) > 6000:
                text = text[:6000]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=self.dimensions
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for query (len={len(text)} chars)")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.dimensions
