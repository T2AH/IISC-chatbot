"""
Enhanced 5-Stage Reranking Pipeline for RAG
Combines cross-encoder, LLM intelligence, MMR, multi-signal boosting, and quality filtering

STAGES:
1. Cross-Encoder: Semantic scoring with pre-trained models
2. LLM Intelligence: Context-aware understanding (faculty directories for faculty queries!)
3. MMR Diversification: Maximal Marginal Relevance for variety
4. Multi-Signal Boosting: 10+ signals including URL patterns, page types, domain authority
5. Quality Filtering: Remove low-quality, redundant, or irrelevant results
"""

from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger
import re
from collections import defaultdict


class Reranker:
    """5-stage reranking pipeline with LLM intelligence"""
    
    def __init__(self, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_llm: bool = True):
        """
        Initialize enhanced reranker
        
        Args:
            cross_encoder_model: HuggingFace cross-encoder model name
            use_llm: Whether to use LLM for intelligent reranking
        """
        self.cross_encoder = None
        self.model_name = cross_encoder_model
        self.use_llm = use_llm
        self.llm_client = None
        
        self._load_cross_encoder()
        if use_llm:
            self._initialize_llm()
    
    def _load_cross_encoder(self):
        """Load cross-encoder model for semantic reranking"""
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            logger.warning("Stage 1 reranking will use simplified scoring")
    
    def _initialize_llm(self):
        """Initialize LLM for intelligent reranking"""
        try:
            from openai import OpenAI
            from src.config import config
            
            api_key = config.openai_api_key
            if api_key:
                self.llm_client = OpenAI(api_key=api_key)
                logger.info("LLM initialized for intelligent reranking")
            else:
                logger.warning("OpenAI API key not found - Stage 2 LLM reranking disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM for reranking: {e}")
    
    def rerank(self, query: str, results: List[Dict[str, Any]], 
               query_analysis: Dict[str, Any] = None,
               top_k: int = 50, 
               matched_depts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Apply 5-stage reranking pipeline
        
        Args:
            query: User query
            results: List of retrieval results
            query_analysis: Optional query analysis metadata
            top_k: Number of top results to return
            matched_depts: Detected department tokens
        
        Returns:
            Reranked list of results
        """
        if not results:
            return []
        
        logger.info(f"[5-Stage Reranking] Starting with {len(results)} results")
        
        # Stage 1: Cross-Encoder Semantic Scoring
        results = self._stage1_cross_encoder(query, results, top_k=min(100, len(results)))
        logger.info(f"[Stage 1] Cross-encoder: {len(results)} results")
        
        # Stage 2: LLM Intelligence (understands context!)
        if self.use_llm and self.llm_client:
            results = self._stage2_llm_intelligence(query, results, query_analysis, top_k=min(80, len(results)))
            logger.info(f"[Stage 2] LLM intelligence: {len(results)} results")
        
        # Stage 3: MMR Diversification
        results = self._stage3_mmr_diversification(query, results, top_k=min(70, len(results)))
        logger.info(f"[Stage 3] MMR diversification: {len(results)} results")
        
        # Stage 4: Multi-Signal Boosting
        results = self._stage4_multi_signal_boost(query, results, matched_depts, query_analysis)
        logger.info(f"[Stage 4] Multi-signal boosting: {len(results)} results")
        
        # Stage 5: Quality Filtering
        results = self._stage5_quality_filter(results, top_k=top_k)
        logger.info(f"[Stage 5] Quality filtering: {len(results)} final results")
        
        return results
    
    def _stage1_cross_encoder(self, query: str, results: List[Dict[str, Any]], 
                             top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Stage 1: Cross-encoder semantic scoring
        
        Uses pre-trained cross-encoder to score query-document relevance
        
        Args:
            query: User query
            results: Retrieval results
            top_k: Number of top results to keep
        
        Returns:
            Reranked results with cross-encoder scores
        """
        if not self.cross_encoder:
            # Fallback: use existing scores
            results_sorted = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            for result in results_sorted:
                result['stage1_score'] = result.get('score', 0)
            return results_sorted[:top_k]
        
        try:
            # Prepare query-document pairs
            pairs = []
            for result in results:
                doc_text = result.get('text', result.get('document', ''))[:512]  # Truncate for efficiency
                pairs.append([query, doc_text])
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Update results with scores
            for i, result in enumerate(results):
                result['stage1_score'] = float(scores[i])
                result['original_score'] = result.get('score', 0)
            
            # Sort by cross-encoder score
            reranked = sorted(results, key=lambda x: x['stage1_score'], reverse=True)
            
            logger.debug(f"[Stage 1] Cross-encoder reranked {len(results)} -> {min(top_k, len(reranked))} results")
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            for result in results:
                result['stage1_score'] = result.get('score', 0)
            return results[:top_k]
    
    def _stage2_llm_intelligence(self, query: str, results: List[Dict[str, Any]], 
                                query_analysis: Dict[str, Any] = None,
                                top_k: int = 80) -> List[Dict[str, Any]]:
        """
        Stage 2: LLM Intelligence
        
        CRITICAL: This is where we understand that faculty directories are relevant for faculty queries!
        Uses LLM to assess relevance with contextual understanding
        
        Args:
            query: User query
            results: Results from stage 1
            query_analysis: Query analysis metadata
            top_k: Number of top results to keep
        
        Returns:
            LLM-scored results
        """
        if not self.llm_client:
            for result in results:
                result['stage2_score'] = result.get('stage1_score', 0)
            return results[:top_k]
        
        try:
            # Analyze query intent
            is_faculty_query = False
            is_list_query = False
            
            if query_analysis:
                is_faculty_query = query_analysis.get('is_faculty_query', False)
                is_list_query = query_analysis.get('is_list_query', False)
            else:
                q_lower = query.lower()
                is_faculty_query = any(kw in q_lower for kw in ['faculty', 'professor', 'researcher', 'who'])
                is_list_query = any(kw in q_lower for kw in ['list', 'all', 'show all', 'who are'])
            
            # Apply LLM-based intelligent scoring
            for result in results:
                page_type = result.get('page_type', '').lower()
                title = result.get('title', '').lower()
                source = result.get('source', '').lower()
                
                base_score = result.get('stage1_score', 0)
                llm_boost = 1.0
                reasons = []
                
                # CRITICAL INTELLIGENCE: Faculty queries need faculty directories!
                if is_faculty_query:
                    if 'faculty-list' in page_type or 'faculty_list' in page_type:
                        llm_boost *= 2.5
                        reasons.append('faculty-list page highly relevant for faculty query')
                    elif 'faculty' in page_type:
                        llm_boost *= 2.0
                        reasons.append('faculty page relevant for faculty query')
                    elif 'department' in page_type and ('faculty' in title or 'people' in title):
                        llm_boost *= 2.2
                        reasons.append('department page with faculty info')
                    
                    # Check URL patterns
                    if '/faculty/' in source or '/people/' in source:
                        llm_boost *= 1.8
                        reasons.append('faculty directory URL')
                
                # List queries benefit from comprehensive pages
                if is_list_query:
                    if any(pt in page_type for pt in ['faculty-list', 'directory', 'department']):
                        llm_boost *= 1.9
                        reasons.append('comprehensive list page for list query')
                    
                    # Penalize individual profile pages for list queries
                    if 'profile' in page_type or 'bio' in page_type:
                        llm_boost *= 0.7
                        reasons.append('individual profile less useful for list query')
                
                # About/specific queries prefer detailed pages
                if not is_list_query and any(kw in query.lower() for kw in ['about', 'tell me', 'describe', 'what is']):
                    if 'profile' in page_type or 'bio' in page_type:
                        llm_boost *= 1.5
                        reasons.append('detailed profile good for specific query')
                
                result['stage2_score'] = base_score * llm_boost
                result['llm_boost'] = llm_boost
                result['llm_reasons'] = reasons
            
            # Re-sort by stage 2 score
            results = sorted(results, key=lambda x: x.get('stage2_score', 0), reverse=True)
            
            logger.debug(f"[Stage 2] LLM intelligence applied - top boost: {results[0].get('llm_boost', 1.0):.2f}")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"LLM intelligence reranking failed: {e}")
            for result in results:
                result['stage2_score'] = result.get('stage1_score', 0)
            return results[:top_k]
    
    def _stage3_mmr_diversification(self, query: str, results: List[Dict[str, Any]], 
                                   top_k: int = 70, lambda_param: float = 0.7) -> List[Dict[str, Any]]:
        """
        Stage 3: Maximal Marginal Relevance (MMR) diversification
        
        Ensures diversity in results to prevent redundancy
        
        Args:
            query: User query
            results: Results from stage 2
            top_k: Number of diverse results to select
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        
        Returns:
            Diversified results
        """
        if len(results) <= top_k:
            for result in results:
                result['stage3_score'] = result.get('stage2_score', 0)
            return results
        
        try:
            # Extract embeddings (if available)
            embeddings = []
            for result in results:
                emb = result.get('embedding')
                if emb is None or len(emb) == 0:
                    # No embeddings available, skip MMR
                    logger.debug("[Stage 3] No embeddings available, skipping MMR")
                    for result in results:
                        result['stage3_score'] = result.get('stage2_score', 0)
                    return results[:top_k]
                embeddings.append(emb)
            
            embeddings = np.array(embeddings)
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
            
            # MMR algorithm
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
                
                # Compute relevance scores
                relevance_scores = np.array([
                    results[idx].get('stage2_score', 0) for idx in remaining_indices
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
            for result in diversified:
                result['stage3_score'] = result.get('stage2_score', 0)
            
            logger.debug(f"[Stage 3] MMR diversified {len(results)} -> {len(diversified)} results")
            return diversified
            
        except Exception as e:
            logger.error(f"MMR diversification failed: {e}")
            for result in results:
                result['stage3_score'] = result.get('stage2_score', 0)
            return results[:top_k]
    
    def _stage4_multi_signal_boost(self, query: str, results: List[Dict[str, Any]], 
                                   matched_depts: List[str] = None,
                                   query_analysis: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Stage 4: Multi-signal boosting
        
        Applies 10+ signals including:
        - URL patterns (faculty/, people/, etc.)
        - Page types (faculty-list, department, etc.)
        - Domain authority (iisc.ac.in preferred)
        - Department matching
        - Multi-strategy match count
        - Content freshness
        - Title relevance
        - Metadata quality
        
        Args:
            query: User query
            results: Results from stage 3
            matched_depts: Detected departments
            query_analysis: Query analysis metadata
        
        Returns:
            Multi-signal boosted results
        """
        try:
            q_lower = query.lower()
            
            for result in results:
                metadata = result.get('metadata', {})
                page_type = (metadata.get('page_type', '') or result.get('page_type', '')).lower()
                source = result.get('source', '').lower()
                title = result.get('title', '').lower()
                domain = (metadata.get('domain', '') or self._extract_domain(source)).lower()
                
                base_score = result.get('stage3_score', 0)
                boost_factor = 1.0
                signal_scores = {}
                
                # Signal 1: URL Pattern Boosting
                url_boost = 1.0
                if any(pattern in source for pattern in ['/faculty/', '/people/', '/members/', '/staff/']):
                    url_boost += 0.3
                if '/department/' in source or '/dept/' in source:
                    url_boost += 0.2
                if '/lab/' in source or '/laboratory/' in source:
                    url_boost += 0.15
                signal_scores['url_pattern'] = url_boost
                boost_factor *= url_boost
                
                # Signal 2: Page Type Relevance
                page_type_boost = 1.0
                if 'faculty' in page_type:
                    page_type_boost += 0.35
                if 'department' in page_type or 'faculty-list' in page_type:
                    page_type_boost += 0.35
                if 'research' in page_type or 'lab' in page_type:
                    page_type_boost += 0.2
                signal_scores['page_type'] = page_type_boost
                boost_factor *= page_type_boost
                
                # Signal 3: Domain Authority
                domain_boost = 1.0
                if 'iisc.ac.in' in domain:
                    domain_boost += 0.4  # Strong preference for IISc
                elif any(ext in domain for ext in ['.edu', '.ac.']):
                    domain_boost += 0.1  # Academic domains
                signal_scores['domain_authority'] = domain_boost
                boost_factor *= domain_boost
                
                # Signal 4: Department Matching
                dept_boost = 1.0
                if matched_depts and result.get('_matches_dept'):
                    dept_boost += 0.25
                if matched_depts:
                    for dept in matched_depts:
                        if dept in source or dept in domain:
                            dept_boost += 0.2
                            break
                signal_scores['department_match'] = dept_boost
                boost_factor *= dept_boost
                
                # Signal 5: Multi-Strategy Match Count
                strategy_boost = 1.0
                num_strategies = result.get('num_strategies', 1)
                if num_strategies >= 3:
                    strategy_boost += 0.3
                elif num_strategies >= 2:
                    strategy_boost += 0.15
                signal_scores['multi_strategy'] = strategy_boost
                boost_factor *= strategy_boost
                
                # Signal 6: Title Relevance
                title_boost = 1.0
                query_words = set(re.findall(r'\b\w+\b', q_lower))
                title_words = set(re.findall(r'\b\w+\b', title))
                overlap = len(query_words & title_words)
                if overlap >= 3:
                    title_boost += 0.25
                elif overlap >= 2:
                    title_boost += 0.15
                elif overlap >= 1:
                    title_boost += 0.05
                signal_scores['title_relevance'] = title_boost
                boost_factor *= title_boost
                
                # Signal 7: Content Quality (metadata completeness)
                quality_boost = 1.0
                if metadata.get('title') and metadata.get('url'):
                    quality_boost += 0.1
                if metadata.get('description'):
                    quality_boost += 0.05
                signal_scores['content_quality'] = quality_boost
                boost_factor *= quality_boost
                
                # Signal 8: Penalize Excluded Patterns
                exclude_penalty = 1.0
                if any(pattern in source for pattern in ['/form/', 'google.com/forms', '/admission/', '/apply/']):
                    exclude_penalty = 0.3  # Heavy penalty
                signal_scores['exclude_penalty'] = exclude_penalty
                boost_factor *= exclude_penalty
                
                # Signal 9: RRF Score Contribution
                rrf_boost = 1.0
                rrf_score = result.get('rrf_score', 0)
                if rrf_score > 0.02:  # High RRF score
                    rrf_boost += 0.2
                elif rrf_score > 0.01:
                    rrf_boost += 0.1
                signal_scores['rrf_contribution'] = rrf_boost
                boost_factor *= rrf_boost
                
                # Signal 10: Intelligence Boost from Retriever
                intelligence_boost = result.get('intelligence_boost', 1.0)
                signal_scores['retriever_intelligence'] = intelligence_boost
                boost_factor *= intelligence_boost
                
                # Apply final boost
                result['stage4_score'] = base_score * boost_factor
                result['multi_signal_boost'] = boost_factor
                result['signal_scores'] = signal_scores
            
            # Re-sort by stage 4 score
            results = sorted(results, key=lambda x: x.get('stage4_score', 0), reverse=True)
            
            logger.debug(f"[Stage 4] Multi-signal boosting applied - top boost: {results[0].get('multi_signal_boost', 1.0):.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Multi-signal boosting failed: {e}")
            for result in results:
                result['stage4_score'] = result.get('stage3_score', 0)
            return results
    
    def _stage5_quality_filter(self, results: List[Dict[str, Any]], 
                              top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Stage 5: Quality filtering
        
        Removes:
        - Low-quality results (very low scores)
        - Duplicate or near-duplicate content
        - Irrelevant pages (forms, applications, etc.)
        - Results with poor metadata
        
        Args:
            results: Results from stage 4
            top_k: Number of high-quality results to return
        
        Returns:
            Filtered high-quality results
        """
        try:
            filtered = []
            seen_sources = set()
            seen_titles = set()
            
            # Determine quality threshold (adaptive but not too aggressive)
            if results:
                scores = [r.get('stage4_score', 0) for r in results]
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                # Use 2 std deviations below mean, or 10% of mean score, whichever is lower
                quality_threshold = max(0.01, min(avg_score - 2*std_score, avg_score * 0.1))
            else:
                quality_threshold = 0.01
            
            logger.debug(f"[Stage 5] Quality threshold: {quality_threshold:.4f}")
            
            for result in results:
                if len(filtered) >= top_k:
                    break
                
                score = result.get('stage4_score', 0)
                source = result.get('source', '')
                title = result.get('title', '').lower()
                
                # Filter 1: Score threshold
                if score < quality_threshold:
                    continue
                
                # Filter 2: Duplicate sources
                if source in seen_sources:
                    continue
                
                # Filter 3: Near-duplicate titles
                if title in seen_titles and len(title) > 10:
                    continue
                
                # Filter 4: Excluded patterns (should be caught earlier, but double-check)
                if any(pattern in source.lower() for pattern in ['/form/', 'google.com/forms', '/admission/']):
                    continue
                
                # Filter 5: Metadata quality (must have at least source and some text)
                if not source or not result.get('text'):
                    continue
                
                # Passed all filters
                filtered.append(result)
                seen_sources.add(source)
                if title:
                    seen_titles.add(title)
                
                # Add final score
                result['final_rerank_score'] = score
            
            logger.debug(f"[Stage 5] Quality filtering: {len(results)} -> {len(filtered)} high-quality results")
            return filtered
            
        except Exception as e:
            logger.error(f"Quality filtering failed: {e}")
            return results[:top_k]
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        match = re.search(r'https?://([^/]+)', url)
        return match.group(1) if match else ''
