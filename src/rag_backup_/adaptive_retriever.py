"""
Adaptive Retrieval System - Self-correcting with iterative refinement
Inspired by MCP (Model Context Protocol) - tries multiple strategies until satisfied
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from src.rag.query_analyzer import QueryAnalyzer
from src.rag.query_enhancement import QueryEnhancer, FusionRetriever
from src.rag.llm_reranker import LLMReranker
from src.database.chromadb_client import ChromaDBClient
from src.database.neo4j_client import Neo4jClient


class AdaptiveRetriever:
    """
    Self-correcting retrieval system that adapts search strategy based on results quality
    Uses iterative refinement like MCP tools
    """
    
    def __init__(self, chromadb_client: ChromaDBClient, neo4j_client: Neo4jClient, llm=None):
        self.chromadb = chromadb_client
        self.neo4j = neo4j_client
        self.analyzer = QueryAnalyzer()
        self.query_enhancer = QueryEnhancer(llm=llm)
        self.fusion_retriever = FusionRetriever()
        self.llm_reranker = LLMReranker(llm=llm)
        
        # Adaptive parameters
        self.initial_k = 35
        self.max_k = 80
        self.confidence_threshold = 0.6
    
    def retrieve(self, query: str, max_iterations: int = 3, use_enhancement: bool = True) -> Dict[str, Any]:
        """
        Adaptive retrieval with iterative refinement and query enhancement
        
        Args:
            query: User query
            max_iterations: Maximum refinement iterations
            use_enhancement: Use HyDE + Multi-Query + Fusion
        
        Returns:
            Best results after adaptive search
        """
        # Analyze query to understand intent
        analysis = self.analyzer.analyze(query)
        logger.info(f"Query type: {analysis['query_type']}, Intents: {analysis['intents']}")
        
        # ENHANCEMENT: Use HyDE + Multi-Query for better semantic matching
        if use_enhancement and self.query_enhancer.llm:
            logger.info("Applying query enhancement (HyDE + Multi-Query + Fusion)")
            enhanced = self.query_enhancer.enhance_query(query)
            
            # Execute search for each query variation
            query_results = {}
            for search_text in enhanced['all_search_texts'][:5]:  # Limit to top 5 variations
                results = self._execute_search(search_text, analysis, self.initial_k)
                # FILTER: Remove very low scoring results before fusion
                filtered = [r for r in results if r.get('score', 0) > 0.3]
                query_results[search_text] = filtered[:25]  # Top 25 per query
            
            # Fuse results using RRF
            fused_results = self.fusion_retriever.fuse_results(query_results)
            
            # POST-FUSION BOOST: Apply URL boosting to fused results
            for doc in fused_results:
                url_boost = self._calculate_dynamic_boost(doc.get('source', ''), analysis)
                doc['score'] = doc.get('fusion_score', 0) * url_boost
                doc['url_boost'] = url_boost
            
            # Re-sort by boosted scores
            fused_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # FINAL STEP: LLM intelligent reranking (understands page types)
            logger.info("Applying LLM intelligent reranking")
            fused_results = self.llm_reranker.intelligent_rerank(query, fused_results, top_k=50)
            
            # Evaluate fused results
            confidence = self._evaluate_confidence(query, fused_results, analysis)
            logger.info(f"Enhanced retrieval confidence: {confidence:.3f}")
            
            return {
                'results': fused_results,
                'confidence': confidence,
                'iterations': 1,
                'analysis': analysis,
                'enhanced': True,
                'query_variations': len(enhanced['all_search_texts'])
            }
        
        # Fallback to standard adaptive retrieval
        best_results = None
        best_confidence = 0.0
        current_k = self.initial_k
        
        for iteration in range(max_iterations):
            logger.info(f"Retrieval iteration {iteration + 1}/{max_iterations}, k={current_k}")
            
            # Execute search with current strategy
            results = self._execute_search(query, analysis, current_k)
            
            # Evaluate results quality
            confidence = self._evaluate_confidence(query, results, analysis)
            logger.info(f"Iteration {iteration + 1} confidence: {confidence:.3f}")
            
            # Keep best results
            if confidence > best_confidence:
                best_results = results
                best_confidence = confidence
            
            # Stop if confident enough
            if confidence >= self.confidence_threshold:
                logger.info(f"Confident results found at iteration {iteration + 1}")
                break
            
            # Adapt strategy for next iteration
            current_k = min(current_k + 20, self.max_k)
            analysis = self._adapt_strategy(analysis, results, confidence)
        
        logger.info(f"Final confidence: {best_confidence:.3f}")
        return {
            'results': best_results,
            'confidence': best_confidence,
            'iterations': iteration + 1,
            'analysis': analysis,
            'enhanced': False
        }
    
    def _execute_search(self, query: str, analysis: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """Execute search based on analyzed strategy"""
        strategy = analysis['search_strategy']
        
        all_results = []
        
        # Vector search with dynamic boosting
        vector_results = self._smart_vector_search(
            query, 
            analysis, 
            int(k * strategy['vector_weight'])
        )
        all_results.extend(vector_results)
        
        # Graph search for entity aggregation
        if strategy['use_entity_aggregation']:
            graph_results = self._entity_aggregation_search(query, analysis)
            all_results.extend(graph_results)
        else:
            graph_results = self._standard_graph_search(query, analysis, int(k * 0.3))
            all_results.extend(graph_results)
        
        return all_results
    
    def _smart_vector_search(self, query: str, analysis: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """Vector search with intelligent query expansion and URL boosting"""
        # Build expanded query
        expansion_terms = ' '.join(analysis['expansion_terms'])
        expanded_query = f"{query} {expansion_terms}"
        
        # Execute vector search
        try:
            results = self.chromadb.query(query_text=expanded_query, n_results=k)
            
            documents = []
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                url = metadata.get('url', 'Unknown')
                
                # Calculate base similarity
                distance = results['distances'][0][i]
                base_similarity = 1 / (1 + distance)
                
                # Apply dynamic URL boosting based on analysis
                url_boost = self._calculate_dynamic_boost(url, analysis)
                final_score = base_similarity * url_boost
                
                doc = {
                    'text': results['documents'][0][i],
                    'metadata': metadata,
                    'source': url,
                    'title': metadata.get('title', 'Unknown'),
                    'domain': metadata.get('domain', 'Unknown'),
                    'page_type': metadata.get('page_type', 'general'),
                    'base_score': base_similarity,
                    'url_boost': url_boost,
                    'score': final_score,
                    'source_type': 'vector'
                }
                documents.append(doc)
            
            # Sort by final score
            documents.sort(key=lambda x: x['score'], reverse=True)
            return documents
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def _calculate_dynamic_boost(self, url: str, analysis: Dict[str, Any]) -> float:
        """Calculate URL boost based on query analysis (no hardcoding)"""
        boost = 1.0
        url_lower = url.lower()
        strategy = analysis['search_strategy']
        
        # CRITICAL: Always heavily penalize external institutions
        external_patterns = ['iitm.ac.in', 'iitb.ac.in', 'mit.edu', 'stanford.edu', 
                           'ibm.com', 'uq.edu.au', '.edu/', '.ac.uk']
        for pattern in external_patterns:
            if pattern in url_lower:
                return 0.2  # Heavy penalty for external sites
        
        # Strongly boost primary domain (iisc.ac.in)
        if 'iisc.ac.in' in url_lower:
            boost *= 2.0
        
        # SUPER BOOST: Exact faculty directory pages (high-value pages)
        if url_lower.endswith('/faculty/') or url_lower.endswith('/people/') or url_lower.endswith('/staff/'):
            boost *= 3.0  # Triple boost for directory pages
        
        # Penalize forms and applications
        if any(pattern in url_lower for pattern in ['/form/', 'google.com/forms', '/admission/', '/apply/']):
            return 0.3
        
        # Boost based on recommended URL patterns from analysis
        for pattern in strategy.get('boost_url_patterns', []):
            if pattern.lower() in url_lower:
                boost *= 1.8
                break
        
        # Boost if URL contains extracted acronyms (domain matching)
        if 'domain_hints' in analysis.get('filters', {}):
            for acronym in analysis['filters']['domain_hints']:
                if acronym.lower() in url_lower:
                    boost *= 1.5
                    break
        
        return boost
    
    def _entity_aggregation_search(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Aggregate entities from graph for 'list all' type queries
        Dynamically detects entity type and context
        """
        try:
            if not self.neo4j.driver:
                logger.warning("Neo4j not available for entity aggregation")
                return []
            
            results = []
            
            # Determine entity type from keywords
            keywords = analysis['entities']['keywords']
            entity_type = None
            
            if any(kw in keywords for kw in ['faculty', 'professor', 'researcher', 'staff']):
                entity_type = 'PERSON'
            elif any(kw in keywords for kw in ['lab', 'laboratory', 'group']):
                entity_type = 'ORG'  # Labs are often ORG entities
            
            if not entity_type:
                return []
            
            # Extract domain context from acronyms
            domain_terms = analysis['entities'].get('acronyms', [])
            if not domain_terms:
                # Use any significant keywords as domain hints
                domain_terms = [kw for kw in keywords if len(kw) > 3][:3]
            
            with self.neo4j.driver.session() as session:
                # Dynamic Cypher query
                cypher = """
                MATCH (p:Page)-[r:MENTIONS]->(e:Entity {type: $entity_type})
                WHERE any(term IN $domain_terms 
                      WHERE toLower(p.url) CONTAINS toLower(term) 
                         OR toLower(p.title) CONTAINS toLower(term)
                         OR toLower(e.name) CONTAINS toLower(term))
                WITH e, count(DISTINCT p) as page_count, sum(r.count) as total_mentions,
                     collect(DISTINCT {url: p.url, title: p.title})[..3] as pages
                WHERE total_mentions > 2
                ORDER BY total_mentions DESC
                LIMIT 20
                RETURN e.name as entity_name, e.type as entity_type,
                       total_mentions, page_count, pages
                """
                
                query_results = session.run(cypher, 
                                           entity_type=entity_type,
                                           domain_terms=domain_terms)
                
                for record in query_results:
                    pages = record['pages']
                    if pages:
                        page = pages[0]
                        results.append({
                            'text': f"{record['entity_name']} ({record['entity_type']}) - Mentioned {record['total_mentions']} times",
                            'entity_name': record['entity_name'],
                            'entity_type': record['entity_type'],
                            'mentions': record['total_mentions'],
                            'source': page.get('url', 'Unknown'),
                            'title': page.get('title', 'Unknown'),
                            'score': min(record['total_mentions'] / 50.0, 1.0),
                            'source_type': 'graph_entity'
                        })
                
                logger.info(f"Entity aggregation found {len(results)} {entity_type} entities")
                return results
                
        except Exception as e:
            logger.error(f"Entity aggregation error: {e}")
            return []
    
    def _standard_graph_search(self, query: str, analysis: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """Standard graph entity search"""
        try:
            if not self.neo4j.driver:
                logger.warning("Neo4j not available for standard graph search")
                return []
            
            results = []
            search_terms = analysis['entities']['keywords']
            
            if not search_terms:
                return []
            
            with self.neo4j.driver.session() as session:
                cypher = """
                MATCH (p:Page)-[r:MENTIONS]->(e:Entity)
                WHERE any(term IN $search_terms 
                      WHERE toLower(e.name) CONTAINS toLower(term))
                WITH e, count(DISTINCT p) as page_count, sum(r.count) as total_mentions,
                     collect(DISTINCT {url: p.url, title: p.title})[..3] as pages
                ORDER BY total_mentions DESC
                LIMIT $limit
                RETURN e.name as entity_name, e.type as entity_type,
                       total_mentions, page_count, pages
                """
                
                query_results = session.run(cypher, search_terms=search_terms, limit=k)
                
                for record in query_results:
                    pages = record['pages']
                    if pages:
                        page = pages[0]
                        results.append({
                            'text': f"Entity: {record['entity_name']} ({record['entity_type']})",
                            'entity_name': record['entity_name'],
                            'entity_type': record['entity_type'],
                            'source': page.get('url', 'Unknown'),
                            'title': page.get('title', 'Unknown'),
                            'score': min(record['total_mentions'] / 100.0, 1.0),
                            'source_type': 'graph'
                        })
                
                return results
                
        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return []
    
    def _evaluate_confidence(self, query: str, results: List[Dict[str, Any]], 
                            analysis: Dict[str, Any]) -> float:
        """
        Evaluate confidence in results quality
        Returns score 0.0 to 1.0
        """
        if not results:
            return 0.0
        
        confidence = 0.5  # baseline
        
        # Factor 1: Top result score
        if results:
            top_score = results[0].get('score', 0)
            confidence += min(top_score * 0.2, 0.2)
        
        # Factor 2: Score diversity (not all same)
        scores = [r.get('score', 0) for r in results[:10]]
        if len(set(scores)) > 5:  # Good diversity
            confidence += 0.1
        
        # Factor 3: Source diversity (multiple domains/types)
        sources = set([r.get('source', '') for r in results[:10]])
        if len(sources) > 5:
            confidence += 0.1
        
        # Factor 4: Presence of recommended URL patterns
        strategy = analysis['search_strategy']
        boost_patterns = strategy.get('boost_url_patterns', [])
        if boost_patterns:
            matching = sum(1 for r in results[:10] 
                          if any(p in r.get('source', '').lower() for p in boost_patterns))
            confidence += min(matching / 10.0, 0.1)
        
        return min(confidence, 1.0)
    
    def _adapt_strategy(self, analysis: Dict[str, Any], results: List[Dict[str, Any]], 
                       confidence: float) -> Dict[str, Any]:
        """Adapt search strategy based on previous results"""
        # If confidence is low, try different approach
        if confidence < 0.4:
            # Switch to more graph-heavy approach
            analysis['search_strategy']['graph_weight'] *= 1.5
            analysis['search_strategy']['use_entity_aggregation'] = True
            logger.info("Low confidence - increasing graph weight")
        
        return analysis
