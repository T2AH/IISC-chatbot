"""
Text processing utilities for cleaning and chunking text
"""

import re
from typing import List, Dict, Any
from loguru import logger


class TextProcessor:
    """Utility class for text processing operations"""
    
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50, min_chunk_size: int = 50):
        """
        Initialize text processor
        
        Args:
            chunk_size: Target number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
            min_chunk_size: Minimum words for a valid chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and special characters
        
        Args:
            text: Raw text to clean
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Remove repeated punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with each chunk
        
        Returns:
            List of chunk dictionaries with text, chunk_id, and metadata
        """
        if not text:
            return []
        
        # Split into words
        words = text.split()
        
        if len(words) < self.min_chunk_size:
            logger.warning(f"Text too short to chunk ({len(words)} words)")
            return []
        
        chunks = []
        chunk_id = 1
        start_idx = 0
        
        while start_idx < len(words):
            # Get chunk of words
            end_idx = start_idx + self.chunk_size
            chunk_words = words[start_idx:end_idx]
            
            # Skip if chunk is too small (unless it's the last chunk)
            if len(chunk_words) < self.min_chunk_size and start_idx + self.chunk_size < len(words):
                start_idx += self.chunk_size - self.chunk_overlap
                continue
            
            # Create chunk text
            chunk_text = ' '.join(chunk_words)
            
            # Create chunk dictionary
            chunk = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'word_count': len(chunk_words),
                'start_word': start_idx,
                'end_word': start_idx + len(chunk_words)
            }
            
            # Add metadata if provided
            if metadata:
                chunk.update(metadata)
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            chunk_id += 1
            start_idx += self.chunk_size - self.chunk_overlap
            
            # Break if we've processed all words
            if end_idx >= len(words):
                break
        
        logger.debug(f"Created {len(chunks)} chunks from {len(words)} words")
        return chunks
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text
        
        Args:
            text: Text to extract sentences from
        
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Simple sentence splitting (can be improved with spaCy)
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about text
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0.0,
                'avg_sentence_length': 0.0
            }
        
        words = text.split()
        sentences = self.extract_sentences(text)
        
        stats = {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0.0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0.0
        }
        
        return stats
