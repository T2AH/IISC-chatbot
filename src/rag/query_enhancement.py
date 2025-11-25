"""
Advanced Query Enhancement System
Implements industry-standard techniques: HyDE, Multi-Query, Query Rewriting
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os


class QueryEnhancer:
    """
    Enhances queries using multiple industry-standard techniques:
    1. HyDE (Hypothetical Document Embeddings) - Generate what answer WOULD look like
    2. Multi-Query - Generate multiple variations of the query
    3. Query Rewriting - Rephrase for better retrieval
    """
    
    def __init__(self, llm=None):
        """
        Initialize query enhancer
        
        Args:
            llm: LangChain LLM instance (optional, will create if not provided)
        """
        self.llm = llm
        if not self.llm:
            try:
                from langchain_openai import ChatOpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.llm = ChatOpenAI(
                        api_key=api_key,
                        model="gpt-4o-mini",  # Cheaper model for query processing
                        temperature=0.3,
                        max_tokens=400
                    )
                    logger.info("QueryEnhancer initialized with GPT-4o-mini")
                else:
                    logger.warning("No OpenAI API key found, advanced query enhancement disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM for query enhancement: {e}")
    
    def enhance_query(self, query: str, methods: List[str] = None) -> Dict[str, Any]:
        """
        Enhance query using multiple methods
        
        Args:
            query: Original user query
            methods: List of methods to use ['hyde', 'multi_query', 'rewrite']
                    If None, uses all available methods
        
        Returns:
            {
                'original': original query,
                'hyde_document': hypothetical answer,
                'multi_queries': list of query variations,
                'rewritten_queries': list of rewritten queries,
                'all_search_texts': combined list for searching
            }
        """
        if not self.llm:
            return {
                'original': query,
                'hyde_document': None,
                'multi_queries': [query],
                'rewritten_queries': [query],
                'all_search_texts': [query]
            }
        
        methods = methods or ['hyde', 'multi_query', 'rewrite']
        
        result = {
            'original': query,
            'hyde_document': None,
            'multi_queries': [],
            'rewritten_queries': [],
            'all_search_texts': [query]  # Always include original
        }
        
        try:
            # Method 1: HyDE - Generate hypothetical answer
            if 'hyde' in methods:
                hyde_doc = self._generate_hyde_document(query)
                if hyde_doc:
                    result['hyde_document'] = hyde_doc
                    result['all_search_texts'].append(hyde_doc)
                    logger.debug(f"HyDE generated: {hyde_doc[:100]}...")
            
            # Method 2: Multi-Query - Generate query variations
            if 'multi_query' in methods:
                multi_queries = self._generate_multi_queries(query)
                result['multi_queries'] = multi_queries
                result['all_search_texts'].extend(multi_queries)
                logger.debug(f"Multi-query generated {len(multi_queries)} variations")
            
            # Method 3: Query Rewriting - Rephrase for better matching
            if 'rewrite' in methods:
                rewritten = self._rewrite_query(query)
                result['rewritten_queries'] = rewritten
                result['all_search_texts'].extend(rewritten)
                logger.debug(f"Query rewriting generated {len(rewritten)} versions")
            
            # Remove duplicates
            result['all_search_texts'] = list(set(result['all_search_texts']))
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
        
        return result
    
    def _generate_hyde_document(self, query: str) -> Optional[str]:
        """
        HyDE: Generate a hypothetical document that would answer this query
        This helps match with actual documents better than the question itself
        """
        if not self.llm:
            return None
        
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            system_prompt = """You are an expert at generating hypothetical answers.
Given a question, write a realistic answer as it would appear in an academic webpage or directory.

For faculty queries: Write a faculty directory entry with names, emails, research areas.
For lab queries: Write a lab description with members, projects, focus areas.
For research queries: Write about research topics, publications, collaborations.

Be specific but generic enough to match real content. Include typical keywords and phrases."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Question: {query}\n\nGenerate a hypothetical answer (2-3 sentences):")
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return None
    
    def _generate_multi_queries(self, query: str, n: int = 3) -> List[str]:
        """
        Generate multiple variations of the query for diverse retrieval
        """
        if not self.llm:
            return []
        
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            system_prompt = """Generate different variations of the given query to improve information retrieval.
Create variations that:
1. Use synonyms and alternative phrasings
2. Break down complex queries into simpler parts  
3. Expand abbreviations or use common alternatives
4. Focus on different aspects of the question

Return ONLY the queries, one per line, no numbering or explanations."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Original query: {query}\n\nGenerate {n} different variations:")
            ]
            
            response = self.llm.invoke(messages)
            queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            # Remove numbering if present
            queries = [q.lstrip('0123456789.-) ') for q in queries]
            return queries[:n]
            
        except Exception as e:
            logger.error(f"Multi-query generation failed: {e}")
            return []
    
    def _rewrite_query(self, query: str) -> List[str]:
        """
        Rewrite query to match how information appears in documents
        """
        if not self.llm:
            return []
        
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            system_prompt = """Rewrite the query to match how information appears in academic documents.

Examples:
- "list all faculty in X" → "X department faculty members staff researchers"
- "who is faculty in Y" → "Y faculty directory professors researchers"
- "tell me about Z lab" → "Z laboratory research group team members projects"

Focus on KEYWORDS that would appear in the actual page content.
Return 2-3 rewritten versions, one per line."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Query: {query}\n\nRewrite for document matching:")
            ]
            
            response = self.llm.invoke(messages)
            rewrites = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            return rewrites[:3]
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return []


class FusionRetriever:
    """
    Retrieval fusion - combines results from multiple query variations
    Implements Reciprocal Rank Fusion (RRF) algorithm
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize fusion retriever
        
        Args:
            k: RRF constant (typically 60)
        """
        self.k = k
    
    def fuse_results(self, query_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Fuse results from multiple queries using Reciprocal Rank Fusion
        
        Args:
            query_results: Dict mapping query -> list of results
        
        Returns:
            Fused and reranked results
        """
        # Track document scores by ID (URL)
        doc_scores = {}
        doc_data = {}
        
        for query, results in query_results.items():
            for rank, doc in enumerate(results):
                doc_id = doc.get('source', f"doc_{rank}")
                
                # RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank + 1)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_data[doc_id] = doc
                
                doc_scores[doc_id] += rrf_score
        
        # Sort by fused score
        sorted_docs = sorted(
            [(doc_id, score) for doc_id, score in doc_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build final results with fusion scores
        fused_results = []
        for doc_id, fusion_score in sorted_docs:
            doc = doc_data[doc_id].copy()
            doc['fusion_score'] = fusion_score
            doc['score'] = fusion_score  # Override original score
            fused_results.append(doc)
        
        logger.info(f"Fused {len(query_results)} query results into {len(fused_results)} unique documents")
        return fused_results
