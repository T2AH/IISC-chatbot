"""
LLM-Powered Intelligent Reranker
Uses LLM to understand document relevance beyond semantic similarity
"""

from typing import List, Dict, Any
from loguru import logger
import os


class LLMReranker:
    """
    Uses LLM to rerank results based on intelligent understanding
    Can recognize that a faculty directory is relevant for faculty queries
    even if semantic similarity is low
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        if not self.llm:
            try:
                from langchain_openai import ChatOpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.llm = ChatOpenAI(
                        api_key=api_key,
                        model="gpt-4o-mini",
                        temperature=0,
                        max_tokens=300
                    )
                    logger.info("LLMReranker initialized")
            except Exception as e:
                logger.warning(f"LLMReranker init failed: {e}")
    
    def intelligent_rerank(self, query: str, results: List[Dict[str, Any]], 
                          top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Rerank using LLM's understanding of relevance
        
        Args:
            query: User query
            results: Retrieved results
            top_k: Rerank top K results (expensive operation)
        
        Returns:
            Reranked results
        """
        if not self.llm or len(results) == 0:
            return results
        
        try:
            # Only rerank top candidates (expensive)
            candidates = results[:top_k]
            
            # Build relevance assessment prompt
            docs_info = []
            for i, doc in enumerate(candidates):
                url = doc.get('source', 'Unknown')
                title = doc.get('title', 'Unknown')
                snippet = doc.get('text', '')[:200]
                docs_info.append(f"{i+1}. URL: {url}\n   Title: {title}\n   Snippet: {snippet}...")
            
            docs_text = "\n\n".join(docs_info[:20])  # Limit to prevent token overflow
            
            from langchain_core.messages import SystemMessage, HumanMessage
            
            system_prompt = """You are an expert at assessing document relevance for academic queries.

Analyze URLs and titles to identify HIGH-VALUE pages:
- Faculty/People directories (*/faculty/, */people/, */staff/) are HIGHLY relevant for faculty queries
- Individual faculty profile pages (*/faculty/name/) are relevant for specific person queries
- Lab pages (*/lab/, */research/) are relevant for lab/research queries
- Department overview pages are relevant for department queries

IGNORE semantic similarity - focus on PAGE TYPE relevance.

Return ONLY the numbers of the top 10 most relevant documents, comma-separated.
Example: 3,7,1,15,2,9,4,11,6,8"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Query: {query}\n\nDocuments:\n{docs_text}\n\nTop 10 most relevant (numbers only):")
            ]
            
            response = self.llm.invoke(messages)
            rankings = response.content.strip()
            
            # Parse rankings
            try:
                ranked_ids = [int(x.strip()) - 1 for x in rankings.split(',') if x.strip().isdigit()]
                
                # Reorder results based on LLM ranking
                reranked = []
                seen = set()
                
                # Add LLM-ranked items first
                for idx in ranked_ids:
                    if 0 <= idx < len(candidates) and idx not in seen:
                        reranked.append(candidates[idx])
                        seen.add(idx)
                
                # Add remaining items
                for idx, doc in enumerate(candidates):
                    if idx not in seen:
                        reranked.append(doc)
                
                # Add rest of results beyond top_k
                reranked.extend(results[top_k:])
                
                logger.info(f"LLM reranking moved {len(ranked_ids)} documents")
                return reranked
                
            except Exception as e:
                logger.error(f"Failed to parse LLM rankings: {e}")
                return results
                
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return results
