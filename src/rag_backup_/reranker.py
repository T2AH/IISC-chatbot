"""
3-Stage Reranking Pipeline for RAG
Optimized for speed and accuracy
"""

from typing import List, Dict, Any
import numpy as np
from loguru import logger


class Reranker:
    """3-stage reranking pipeline: Cross-Encoder -> MMR -> Metadata Boosting"""
    
    def __init__(self, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker with cross-encoder model
        
        Args:
            cross_encoder_model: HuggingFace cross-encoder model name
        """
        self.cross_encoder = None
        self.model_name = cross_encoder_model
        self._load_cross_encoder()
    
    def _load_cross_encoder(self):
        """Load cross-encoder model for reranking"""
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            logger.warning("Reranking will use simplified scoring")
    
    def rerank(self, query: str, results: List[Dict[str, Any]], 
               top_k: int = 50, use_mmr: bool = True, 
               metadata_boost: bool = True, matched_depts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Apply 3-stage reranking pipeline
        
        Args:
            query: User query
            results: List of retrieval results
            top_k: Number of top results to return
            use_mmr: Apply MMR diversification
            metadata_boost: Apply metadata-based boosting
        
        Returns:
            Reranked list of results
        """
        if not results:
            return []
        
        # Stage 1: Cross-Encoder Reranking
        results = self._cross_encoder_rerank(query, results, top_k=min(50, len(results)))
        
        # Stage 2: MMR Diversification (optional)
        if use_mmr and len(results) > top_k:
            results = self._mmr_diversification(query, results, top_k=min(50, len(results)))
        
        # Stage 3: Metadata Boosting (optional)
        if metadata_boost:
            results = self._metadata_boost(results, matched_depts=matched_depts)
        
        # Return top_k results
        return results[:top_k]
    
    def _cross_encoder_rerank(self, query: str, results: List[Dict[str, Any]], 
                             top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Stage 1: Cross-encoder reranking for accuracy
        
        Args:
            query: User query
            results: Retrieval results
            top_k: Number of top results to keep
        
        Returns:
            Reranked results
        """
        if not self.cross_encoder:
            # Fallback: use existing scores
            return sorted(results, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
        
        try:
            # Prepare query-document pairs
            pairs = []
            for result in results:
                doc_text = result.get('text', result.get('document', ''))
                pairs.append([query, doc_text])
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Update results with new scores
            for i, result in enumerate(results):
                result['cross_encoder_score'] = float(scores[i])
                result['original_score'] = result.get('score', 0)
            
            # Sort by cross-encoder score
            reranked = sorted(results, key=lambda x: x['cross_encoder_score'], reverse=True)
            
            logger.debug(f"Cross-encoder reranked {len(results)} -> {top_k} results")
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return results[:top_k]
    
    def _mmr_diversification(self, query: str, results: List[Dict[str, Any]], 
                            top_k: int = 50, lambda_param: float = 0.7) -> List[Dict[str, Any]]:
        """
        Stage 2: Maximal Marginal Relevance (MMR) for diversity
        
        Args:
            query: User query
            results: Reranked results from stage 1
            top_k: Number of diverse results to select
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        
        Returns:
            Diversified results
        """
        if len(results) <= top_k:
            return results
        
        try:
            # Extract embeddings (if available)
            embeddings = []
            for result in results:
                emb = result.get('embedding')
                if emb is None or len(emb) == 0:
                    # No embeddings available, skip MMR
                    logger.warning("No embeddings available for MMR, skipping diversification")
                    return results[:top_k]
                embeddings.append(emb)
            
            embeddings = np.array(embeddings)
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
            
            # Initialize
            selected_indices = []
            remaining_indices = list(range(len(results)))
            
            # Select first document (highest score)
            first_idx = 0
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
            
            # Iteratively select diverse documents
            while len(selected_indices) < top_k and remaining_indices:
                selected_embeddings = embeddings[selected_indices]
                remaining_embeddings = embeddings[remaining_indices]
                
                # Compute relevance scores (use cross-encoder scores)
                relevance_scores = np.array([
                    results[idx].get('cross_encoder_score', results[idx].get('score', 0))
                    for idx in remaining_indices
                ])
                
                # Normalize relevance scores
                if relevance_scores.max() > relevance_scores.min():
                    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())
                
                # Compute max similarity to selected documents
                similarities = np.matmul(remaining_embeddings, selected_embeddings.T)
                max_similarities = similarities.max(axis=1)
                
                # MMR score: λ * relevance - (1-λ) * max_similarity
                mmr_scores = lambda_param * relevance_scores - (1 - lambda_param) * max_similarities
                
                # Select document with highest MMR score
                best_idx = remaining_indices[mmr_scores.argmax()]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            # Reorder results
            diversified = [results[idx] for idx in selected_indices]
            
            logger.debug(f"MMR diversified {len(results)} -> {len(diversified)} results")
            return diversified
            
        except Exception as e:
            logger.error(f"MMR diversification failed: {e}")
            return results[:top_k]
    
    def _metadata_boost(self, results: List[Dict[str, Any]], matched_depts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Stage 3: Metadata-based score boosting
        
        Args:
            results: Diversified results from stage 2
        
        Returns:
            Results with metadata-boosted scores
        """
        try:
            for result in results:
                metadata = result.get('metadata', {})
                page_type = (metadata.get('page_type', '') or result.get('page_type', '')).lower()

                # Get current score
                score = result.get('cross_encoder_score', result.get('score', 0))

                # Apply dynamic, non-hardcoded boosts
                boost_factor = 1.0

                # If retriever flagged this as matching the detected department, boost
                if result.get('_meta_matches_dept'):
                    boost_factor += 0.25

                # If matched_depts present, boost documents whose domain/source contains any dept token
                src = (result.get('source') or '').lower()
                domain = (metadata.get('domain') or '').lower()
                if matched_depts:
                    for d in matched_depts:
                        if d in src or d in domain:
                            boost_factor += 0.20
                            break

                # Boost for page types
                if 'faculty' in page_type:
                    boost_factor += 0.30
                if 'department' in page_type or 'faculty-list' in page_type:
                    boost_factor += 0.30
                if 'research' in page_type:
                    boost_factor += 0.15
                if 'lab' in page_type or 'project' in page_type:
                    boost_factor += 0.15

                # Update score with boost (multiplicative mix)
                result['boosted_score'] = score * (1.0 + (boost_factor - 1.0))
                result['boost_factor'] = boost_factor
            
            # Re-sort by boosted score
            results = sorted(results, key=lambda x: x.get('boosted_score', 0), reverse=True)
            
            logger.debug(f"Applied metadata boosting to {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Metadata boosting failed: {e}")
            return results
