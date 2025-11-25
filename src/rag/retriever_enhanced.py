"""
Enhanced Multi-Strategy Retriever with Reciprocal Rank Fusion
Combines dense, lexical, graph, and targeted retrieval methods

FEATURES:
- Multi-strategy retrieval: 4 different methods (dense, lexical, graph, targeted)
- Reciprocal Rank Fusion: Intelligently merges all results
- Smart query analysis: Auto-detects intents, entities, departments
- Intelligence boosting: Context-aware scoring based on page types
- Result diversification: Prevents redundancy
"""

from typing import List, Dict, Any, Tuple, Set
from loguru import logger
import re
from collections import defaultdict
import numpy as np

from src.database.chromadb_client import ChromaDBClient
from src.database.neo4j_client import Neo4jClient
from src.rag.query_embedder import QueryEmbedder
from src.rag.query_analyzer import QueryAnalyzer
from src.config import config


class EnhancedHybridRetriever:
    """Enhanced hybrid retriever with 4 retrieval strategies and RRF fusion"""
    
    def __init__(self, chromadb_client: ChromaDBClient = None, neo4j_client: Neo4jClient = None):
        """
        Initialize enhanced hybrid retriever
        
        Args:
            chromadb_client: ChromaDB client instance
            neo4j_client: Neo4j client instance
        """
        self.chromadb = chromadb_client or ChromaDBClient()
        self.neo4j = neo4j_client or Neo4jClient()
        
        # Initialize query embedder
        try:
            self.query_embedder = QueryEmbedder()
        except Exception as e:
            logger.error(f"Failed to initialize query embedder: {e}")
            self.query_embedder = None
        
        # Initialize query analyzer for smart detection
        try:
            self.query_analyzer = QueryAnalyzer()
        except Exception as e:
            logger.warning(f"Query analyzer not available: {e}")
            self.query_analyzer = None
        
        # Get configuration
        self.top_k_vectors = config.get('rag', 'retrieval', 'top_k_vectors', default=100)
        self.top_k_graph = config.get('rag', 'retrieval', 'top_k_graph', default=20)
        self.similarity_threshold = config.get('rag', 'retrieval', 'similarity_threshold', default=0.25)
        self.merge_same_source = config.get('rag', 'retrieval', 'merge_same_source', default=True)
        self.max_chunks_per_source = config.get('rag', 'retrieval', 'max_chunks_per_source', default=10)
        
        # Department aliases for better matching
        self.dept_aliases = {
            'biochem': ['department of biochemistry', 'biochemistry department', 'biochemistry'],
            'caf': ['central animal facility'],
            'ces': ['centre for ecological sciences', 'ecological sciences centre', 'ecological sciences'],
            'cidr': ['centre for infectious disease research', 'infectious disease research centre'],
            'cns': ['centre for neuroscience', 'neuroscience centre', 'neuroscience'],
            'mcb': ['department of microbiology and cell biology', 'microbiology and cell biology'],
            'mbu': ['molecular biophysics unit', 'molecular biophysics'],
            'dbg': ['department of developmental biology and genetics', 'developmental biology and genetics'],
            'ipc': ['department of inorganic and physical chemistry', 'inorganic and physical chemistry'],
            'mrc': ['materials research centre', 'materials research'],
            'orgchem': ['department of organic chemistry', 'organic chemistry department'],
            'sscu': ['solid state and structural chemistry unit', 'solid state chemistry'],
            'csa': ['computer science and automation', 'csa department'],
            'ece': ['electrical communication engineering', 'ece department'],
            'dese': ['department of electronic systems engineering', 'electronic systems engineering'],
            'ee': ['electrical engineering', 'electrical engineering department'],
            'cistup': ['centre for infrastructure, sustainable transportation and urban planning'],
            'be': ['bioengineering', 'bioengineering department'],
            'csp': ['centre for sustainable technologies', 'sustainable technologies centre'],
            'cense': ['centre for nanoscience and engineering', 'nanoscience and engineering'],
            'cds': ['computational and data sciences', 'computational data sciences', 'cds department'],
            'mgmt': ['management studies', 'management department'],
            'icer': ['interdisciplinary centre for energy research', 'energy research centre'],
            'icwar': ['interdisciplinary centre for water research', 'water research centre'],
            'cps': ['centre for contemporary studies', 'contemporary studies centre'],
            'msci': ['department of materials science', 'materials science department'],
            'serc': ['supercomputer education and research centre', 'supercomputer research centre'],
            'iqti': ['international centre for quantum technology initiatives'],
            'abcmc': ['atomic, biomolecular and chemical sciences centre'],
            'longevity': ['centre for longevity research', 'longevity research centre'],
            'aero': ['aerospace engineering', 'aerospace engineering department'],
            'caos': ['centre for atmospheric and oceanic sciences', 'atmospheric and oceanic sciences'],
            'ceas': ['centre for earth sciences', 'earth sciences centre'],
            'camm': ['centre for advanced manufacturing and materials'],
            'dm': ['department of mathematics', 'mathematics department', 'mathematics'],
            'cst': ['centre for scientific teaching', 'scientific teaching centre'],
            'chemeng': ['chemical engineering', 'chemical engineering department'],
            'civil': ['civil engineering', 'civil engineering department'],
            'dccc': ['digital campus and cloud computing centre', 'cloud computing centre'],
            'materials': ['materials engineering', 'materials engineering department'],
            'mecheng': ['mechanical engineering', 'mechanical engineering department'],
            'physics': ['department of physics', 'physics department', 'physics'],
            'cbr': ['centre for brain research', 'brain research centre'],
            'math': ['mathematics department', 'mathematics'],
            'iap': ['instrumentation and applied physics', 'applied physics']
        }
        
        # URL patterns for intelligence boosting
        self.url_patterns = {
            'faculty': ['/faculty/', '/people/', '/members/', '/staff/', '/researchers/', '/team/'],
            'lab': ['/lab/', '/laboratory/', '/research-group/', '/group/', '/center/', '/centre/'],
            'department': ['/department/', '/dept/', '/division/', '/faculty-list/'],
            'exclude': ['/form/', 'google.com/forms', '/admission/', '/apply/', '/application/']
        }
        
        # RRF parameter (k=60 is standard for Reciprocal Rank Fusion)
        self.rrf_k = 60
        
        logger.info("Enhanced Multi-Strategy Hybrid Retriever initialized")

    def _detect_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Smart query analysis to detect intents, entities, and departments
        
        Args:
            query: User query
            
        Returns:
            Dictionary with detected information
        """
        if self.query_analyzer:
            analysis = self.query_analyzer.analyze(query)
        else:
            # Fallback simple detection
            analysis = {
                'intents': [],
                'query_type': 'general',
                'entities': {},
                'filters': {}
            }
        
        # Detect departments using alias map
        matched_depts = self._detect_departments(query)
        analysis['departments'] = matched_depts
        
        # Detect if list query
        q_lower = query.lower()
        analysis['is_list_query'] = any(kw in q_lower for kw in [
            'list', 'all', 'show all', 'who are', 'faculty members', 'researchers'
        ])
        
        # Detect if faculty query
        analysis['is_faculty_query'] = any(kw in q_lower for kw in [
            'faculty', 'professor', 'researcher', 'staff', 'who is', 'people'
        ])
        
        return analysis

    def _detect_departments(self, query: str) -> List[str]:
        """
        Detect department tokens from the query using the alias map
        
        Args:
            query: User query
            
        Returns:
            List of matched department keys
        """
        q = (query or '').lower()
        matched = []
        try:
            for key, aliases in self.dept_aliases.items():
                for alias in aliases:
                    if re.search(rf"\b{re.escape(alias)}\b", q):
                        matched.append(key)
                        break
        except Exception:
            return []
        return matched

    def retrieve(self, query: str, filters: Dict[str, Any] = None, top_k: int = 50) -> Dict[str, Any]:
        """
        Enhanced multi-strategy retrieval with RRF fusion
        
        Strategies:
        1. Dense retrieval (vector/semantic search)
        2. Lexical retrieval (keyword/BM25-like)
        3. Graph retrieval (knowledge graph entities)
        4. Targeted retrieval (page type + URL patterns)
        
        Args:
            query: User query
            filters: Optional metadata filters
            top_k: Number of final results after fusion
        
        Returns:
            Dictionary with fused results and metadata
        """
        logger.info(f"[Multi-Strategy Retrieval] Query: {query[:100]}...")
        
        # Step 1: Smart query analysis
        query_analysis = self._detect_query_intent(query)
        matched_depts = query_analysis.get('departments', [])
        is_faculty_query = query_analysis.get('is_faculty_query', False)
        
        if matched_depts:
            logger.info(f"Detected departments: {matched_depts}")
        if is_faculty_query:
            logger.info("Faculty query detected - will boost faculty pages")
        
        # Step 2: Execute all 4 retrieval strategies
        strategy_results = {}
        
        # Strategy 1: Dense (semantic/vector) retrieval
        dense_results = self._dense_retrieval(query, filters, matched_depts)
        strategy_results['dense'] = dense_results
        logger.info(f"Dense retrieval: {len(dense_results)} results")
        
        # Strategy 2: Lexical retrieval (keyword-based)
        lexical_results = self._lexical_retrieval(query, filters, matched_depts)
        strategy_results['lexical'] = lexical_results
        logger.info(f"Lexical retrieval: {len(lexical_results)} results")
        
        # Strategy 3: Graph retrieval
        graph_results = self._graph_retrieval(query, matched_depts)
        strategy_results['graph'] = graph_results
        logger.info(f"Graph retrieval: {len(graph_results)} results")
        
        # Strategy 4: Targeted retrieval (context-aware)
        targeted_results = self._targeted_retrieval(query, query_analysis, filters)
        strategy_results['targeted'] = targeted_results
        logger.info(f"Targeted retrieval: {len(targeted_results)} results")
        
        # Step 3: Apply Reciprocal Rank Fusion (RRF)
        fused_results = self._reciprocal_rank_fusion(strategy_results, top_k=top_k*2)  # Get more for post-processing
        logger.info(f"RRF fusion: {len(fused_results)} fused results")
        
        # Step 4: Apply intelligence boosting based on context
        boosted_results = self._intelligence_boosting(fused_results, query_analysis)
        
        # Step 5: Diversification to prevent redundancy
        final_results = self._diversify_results(boosted_results, top_k=top_k)
        
        logger.info(f"[Final] Returning {len(final_results)} diversified results")
        
        return {
            'query': query,
            'query_analysis': query_analysis,
            'strategy_results': {k: len(v) for k, v in strategy_results.items()},
            'vector_results': final_results,  # Keep compatibility with existing code
            'graph_results': graph_results[:10],  # Keep some graph results separate
            'matched_depts': matched_depts,
            'num_results': len(final_results)
        }

    def _dense_retrieval(self, query: str, filters: Dict[str, Any] = None, 
                        matched_depts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Strategy 1: Dense semantic retrieval using embeddings
        
        Args:
            query: Search query
            filters: Metadata filters
            matched_depts: Detected departments
            
        Returns:
            List of semantically relevant documents
        """
        try:
            if not self.chromadb.collection or not self.query_embedder:
                return []
            
            # Query expansion for better semantic matching
            expanded_query = self._expand_query_semantic(query, matched_depts)
            
            # Generate embedding
            query_embedding = self.query_embedder.embed(expanded_query)
            
            # Query ChromaDB
            results = self.chromadb.query(
                query_embedding=query_embedding,
                n_results=self.top_k_vectors,
                where=filters
            )
            
            # Format results
            documents = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                
                # Convert distance to similarity
                similarity = 1 / (1 + distance)
                
                doc = {
                    'text': results['documents'][0][i],
                    'metadata': metadata,
                    'score': similarity,
                    'distance': distance,
                    'source': metadata.get('url', 'Unknown'),
                    'title': metadata.get('title', 'Unknown'),
                    'page_type': metadata.get('page_type', 'general'),
                    'retrieval_strategy': 'dense',
                    '_matches_dept': self._check_dept_match(metadata, matched_depts)
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Dense retrieval error: {e}")
            return []

    def _lexical_retrieval(self, query: str, filters: Dict[str, Any] = None,
                          matched_depts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Strategy 2: Lexical/keyword-based retrieval
        
        Uses query terms directly for matching, good for exact keywords
        
        Args:
            query: Search query
            filters: Metadata filters
            matched_depts: Detected departments
            
        Returns:
            List of keyword-matched documents
        """
        try:
            if not self.chromadb.collection or not self.query_embedder:
                return []
            
            # Extract keywords (remove stopwords)
            keywords = self._extract_keywords(query)
            keyword_query = ' '.join(keywords)
            
            # Add department keywords
            if matched_depts:
                keyword_query += ' ' + ' '.join(matched_depts)
            
            # Generate embedding for keyword query
            query_embedding = self.query_embedder.embed(keyword_query)
            
            results = self.chromadb.query(
                query_embedding=query_embedding,
                n_results=min(50, self.top_k_vectors),
                where=filters
            )
            
            documents = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                similarity = 1 / (1 + distance)
                
                doc = {
                    'text': results['documents'][0][i],
                    'metadata': metadata,
                    'score': similarity,
                    'source': metadata.get('url', 'Unknown'),
                    'title': metadata.get('title', 'Unknown'),
                    'page_type': metadata.get('page_type', 'general'),
                    'retrieval_strategy': 'lexical',
                    '_matches_dept': self._check_dept_match(metadata, matched_depts)
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Lexical retrieval error: {e}")
            return []

    def _graph_retrieval(self, query: str, matched_depts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Strategy 3: Knowledge graph retrieval
        
        Leverages entity relationships and mentions
        
        Args:
            query: Search query
            matched_depts: Detected departments
            
        Returns:
            List of graph-based results
        """
        try:
            if not self.neo4j.driver:
                return []
            
            results = []
            
            with self.neo4j.driver.session() as session:
                # Try exact match first
                exact_cypher = """
                MATCH (e:Entity)
                WHERE toLower(e.name) = toLower($q) OR toLower(e.alias) = toLower($q)
                RETURN e.name AS entity_name, e.type AS entity_type, e AS entity
                LIMIT $limit
                """
                exact_results = session.run(exact_cypher, q=query.strip(), limit=self.top_k_graph)
                
                for record in exact_results:
                    results.append({
                        'entity_name': record['entity_name'],
                        'entity_type': record['entity_type'],
                        'text': f"{record['entity_name']} ({record['entity_type']})",
                        'source': 'knowledge_graph',
                        'score': 1.0,  # Exact match gets highest score
                        'retrieval_strategy': 'graph',
                        'page_type': 'entity'
                    })
                
                if results:
                    return results
                
                # Fuzzy entity search with mentions
                query_terms = self._extract_keywords(query)
                search_text = ' '.join(query_terms)
                
                fuzzy_cypher = """
                MATCH (p:Page)-[r:MENTIONS]->(e:Entity)
                WHERE any(term IN split(toLower($query_text), ' ') 
                      WHERE toLower(e.name) CONTAINS term OR 
                            toLower(e.type) CONTAINS term)
                WITH e, count(DISTINCT p) as page_count, sum(r.count) as total_mentions,
                     collect(DISTINCT {url: p.url, title: p.title})[..3] as sample_pages
                ORDER BY total_mentions DESC, page_count DESC
                LIMIT $limit
                RETURN e.name as entity_name, e.type as entity_type, 
                       page_count, total_mentions, 
                       [page IN sample_pages | page.url] as sample_urls,
                       [page IN sample_pages | page.title] as sample_titles
                """
                
                fuzzy_results = session.run(fuzzy_cypher, query_text=search_text, limit=self.top_k_graph)
                
                for record in fuzzy_results:
                    mention_score = min(1.0, (record['total_mentions'] or 0) / 100.0)
                    
                    results.append({
                        'entity_name': record['entity_name'],
                        'entity_type': record['entity_type'],
                        'text': f"{record['entity_name']} ({record['entity_type']}) - {record['total_mentions']} mentions",
                        'page_count': record['page_count'],
                        'total_mentions': record['total_mentions'],
                        'sample_urls': record.get('sample_urls', []),
                        'sample_titles': record.get('sample_titles', []),
                        'source': 'knowledge_graph',
                        'score': mention_score,
                        'retrieval_strategy': 'graph',
                        'page_type': 'entity'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Graph retrieval error: {e}")
            return []

    def _targeted_retrieval(self, query: str, query_analysis: Dict[str, Any], 
                           filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Strategy 4: Targeted retrieval based on query understanding
        
        Context-aware retrieval that targets specific page types and patterns
        
        Args:
            query: Search query
            query_analysis: Query analysis results
            filters: Metadata filters
            
        Returns:
            List of targeted documents
        """
        try:
            if not self.chromadb.collection or not self.query_embedder:
                return []
            
            # Determine target page types
            target_page_types = self._determine_target_page_types(query_analysis)
            
            # Build targeted filter
            targeted_filter = filters.copy() if filters else {}
            if target_page_types:
                if 'page_type' not in targeted_filter:
                    targeted_filter['page_type'] = {'$in': target_page_types}
            
            # Generate embedding
            query_embedding = self.query_embedder.embed(query)
            
            # Query with targeted filter
            results = self.chromadb.query(
                query_embedding=query_embedding,
                n_results=min(30, self.top_k_vectors),
                where=targeted_filter if targeted_filter else None
            )
            
            documents = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                url = metadata.get('url', '')
                
                # Calculate targeted score with URL boost
                base_similarity = 1 / (1 + distance)
                url_boost = self._calculate_url_boost(url, query)
                targeted_score = base_similarity * url_boost
                
                doc = {
                    'text': results['documents'][0][i],
                    'metadata': metadata,
                    'score': targeted_score,
                    'base_score': base_similarity,
                    'url_boost': url_boost,
                    'source': url,
                    'title': metadata.get('title', 'Unknown'),
                    'page_type': metadata.get('page_type', 'general'),
                    'retrieval_strategy': 'targeted',
                    '_matches_dept': self._check_dept_match(metadata, query_analysis.get('departments', []))
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Targeted retrieval error: {e}")
            return []

    def _reciprocal_rank_fusion(self, strategy_results: Dict[str, List[Dict[str, Any]]], 
                               top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion (RRF) to merge all retrieval strategies
        
        RRF formula: score(d) = sum over all strategies of 1/(k + rank(d))
        where k=60 is standard, rank(d) is the rank of document d in that strategy
        
        Args:
            strategy_results: Dict mapping strategy name to ranked results
            top_k: Number of top results to return
            
        Returns:
            Fused and ranked result list
        """
        logger.debug(f"[RRF] Fusing {len(strategy_results)} strategies...")
        
        # Track documents by unique identifier (URL + text hash)
        doc_scores = defaultdict(lambda: {'rrf_score': 0.0, 'doc': None, 'strategies': []})
        
        for strategy_name, results in strategy_results.items():
            for rank, doc in enumerate(results, start=1):
                # Create unique key (use source URL primarily)
                doc_key = doc.get('source', '') or doc.get('entity_name', '') or doc.get('text', '')[:100]
                
                # Calculate RRF contribution: 1 / (k + rank)
                rrf_contribution = 1.0 / (self.rrf_k + rank)
                
                # Accumulate RRF score
                doc_scores[doc_key]['rrf_score'] += rrf_contribution
                doc_scores[doc_key]['strategies'].append(strategy_name)
                
                # Store document (first occurrence)
                if doc_scores[doc_key]['doc'] is None:
                    doc_scores[doc_key]['doc'] = doc
        
        # Convert to list and add RRF metadata
        fused_documents = []
        for doc_key, data in doc_scores.items():
            doc = data['doc']
            doc['rrf_score'] = data['rrf_score']
            doc['matched_strategies'] = data['strategies']
            doc['num_strategies'] = len(data['strategies'])
            fused_documents.append(doc)
        
        # Sort by RRF score (highest first)
        fused_documents.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        logger.debug(f"[RRF] Fused {len(fused_documents)} unique documents")
        if fused_documents:
            logger.debug(f"[RRF] Top result: {fused_documents[0].get('source', 'Unknown')[:60]} "
                        f"(rrf_score={fused_documents[0]['rrf_score']:.4f}, strategies={fused_documents[0]['num_strategies']})")
        
        return fused_documents[:top_k]

    def _intelligence_boosting(self, results: List[Dict[str, Any]], 
                              query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply context-aware score boosting based on page types and query intent
        
        CRITICAL: This is where we understand that faculty directories are relevant for faculty queries!
        
        Args:
            results: Fused results from RRF
            query_analysis: Query analysis metadata
            
        Returns:
            Results with intelligence-boosted scores
        """
        is_faculty_query = query_analysis.get('is_faculty_query', False)
        is_list_query = query_analysis.get('is_list_query', False)
        matched_depts = query_analysis.get('departments', [])
        
        for doc in results:
            page_type = doc.get('page_type', '').lower()
            source = doc.get('source', '').lower()
            base_score = doc.get('rrf_score', 0.0)
            
            boost_factor = 1.0
            boost_reasons = []
            
            # CRITICAL: Faculty queries should boost faculty-list and faculty pages!
            if is_faculty_query:
                if 'faculty-list' in page_type or 'faculty_list' in page_type:
                    boost_factor *= 2.5
                    boost_reasons.append('faculty-list page for faculty query')
                elif 'faculty' in page_type:
                    boost_factor *= 2.0
                    boost_reasons.append('faculty page for faculty query')
                elif 'department' in page_type:
                    boost_factor *= 1.8
                    boost_reasons.append('department page for faculty query')
                
                # Boost URLs with /faculty/ or /people/ paths
                if any(pattern in source for pattern in ['/faculty/', '/people/', '/members/']):
                    boost_factor *= 1.5
                    boost_reasons.append('faculty URL pattern')
            
            # List queries benefit from comprehensive pages
            if is_list_query:
                if any(pt in page_type for pt in ['faculty-list', 'department', 'directory']):
                    boost_factor *= 1.8
                    boost_reasons.append('list-type page for list query')
            
            # Department match bonus
            if doc.get('_matches_dept') or matched_depts:
                domain_match = any(dept in source for dept in matched_depts)
                if domain_match:
                    boost_factor *= 1.6
                    boost_reasons.append('department domain match')
            
            # Boost for multiple strategy matches (high confidence)
            num_strategies = doc.get('num_strategies', 1)
            if num_strategies >= 3:
                boost_factor *= 1.4
                boost_reasons.append(f'matched {num_strategies} strategies')
            elif num_strategies >= 2:
                boost_factor *= 1.2
                boost_reasons.append(f'matched {num_strategies} strategies')
            
            # Apply boost
            doc['intelligence_boost'] = boost_factor
            doc['boost_reasons'] = boost_reasons
            doc['final_score'] = base_score * boost_factor
        
        # Re-sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        logger.debug(f"[Intelligence Boost] Applied to {len(results)} results")
        if results:
            top = results[0]
            logger.debug(f"[Intelligence Boost] Top result: {top.get('source', 'Unknown')[:60]} "
                        f"(boost={top.get('intelligence_boost', 1.0):.2f}, reasons={top.get('boost_reasons', [])})")
        
        return results

    def _diversify_results(self, results: List[Dict[str, Any]], top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Apply result diversification to prevent redundancy
        
        Ensures variety in sources, page types, and content
        
        Args:
            results: Intelligence-boosted results
            top_k: Number of diverse results to return
            
        Returns:
            Diversified result list
        """
        if len(results) <= top_k:
            return results
        
        diversified = []
        seen_sources = set()
        seen_domains = set()
        page_type_counts = defaultdict(int)
        
        # First pass: Take top results with diversity constraints
        for doc in results:
            if len(diversified) >= top_k:
                break
            
            source = doc.get('source', '')
            domain = doc.get('metadata', {}).get('domain', '') or self._extract_domain(source)
            page_type = doc.get('page_type', 'general')
            
            # Diversity constraints (can be relaxed for very high scores)
            high_score = doc.get('final_score', 0) > 0.05
            
            # Check domain diversity (allow max 5 from same domain unless high score)
            domain_count = sum(1 for d in diversified if self._extract_domain(d.get('source', '')) == domain)
            if domain_count >= 5 and not high_score:
                continue
            
            # Check page type diversity (allow max 10 of same type unless high score)
            if page_type_counts[page_type] >= 10 and not high_score:
                continue
            
            # Add to diversified results
            diversified.append(doc)
            seen_sources.add(source)
            seen_domains.add(domain)
            page_type_counts[page_type] += 1
        
        # If we didn't get enough, add remaining high-scoring results
        if len(diversified) < top_k:
            for doc in results:
                if len(diversified) >= top_k:
                    break
                if doc not in diversified:
                    diversified.append(doc)
        
        logger.debug(f"[Diversification] Selected {len(diversified)} diverse results from {len(results)}")
        
        return diversified

    # ========== Helper Methods ==========

    def _expand_query_semantic(self, query: str, matched_depts: List[str] = None) -> str:
        """Expand query for better semantic retrieval"""
        expanded = query
        
        # Add department names if detected
        if matched_depts:
            for dept in matched_depts:
                dept_names = self.dept_aliases.get(dept, [])
                if dept_names:
                    expanded += ' ' + dept_names[0]
        
        # Generic semantic expansion
        q_lower = query.lower()
        if 'faculty' in q_lower or 'professor' in q_lower:
            expanded += ' researcher scientist staff instructor teacher'
        if 'lab' in q_lower:
            expanded += ' laboratory research group'
        if 'research' in q_lower:
            expanded += ' study investigation project'
        
        return expanded

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
                    'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when',
                    'where', 'how', 'why', 'all', 'about', 'tell', 'me'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return keywords

    def _check_dept_match(self, metadata: Dict[str, Any], matched_depts: List[str]) -> bool:
        """Check if document metadata matches detected departments"""
        if not matched_depts:
            return False
        
        domain = (metadata.get('domain', '') or '').lower()
        url = (metadata.get('url', '') or '').lower()
        
        for dept in matched_depts:
            if dept in domain or dept in url:
                return True
        
        return False

    def _determine_target_page_types(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Determine which page types to target based on query analysis"""
        target_types = []
        
        is_faculty_query = query_analysis.get('is_faculty_query', False)
        query_type = query_analysis.get('query_type', 'general')
        
        if is_faculty_query:
            target_types.extend(['faculty', 'faculty-list', 'department', 'lab'])
        
        if 'lab' in query_type:
            target_types.extend(['lab', 'research'])
        
        if 'department' in query_type:
            target_types.extend(['department', 'faculty-list'])
        
        return list(set(target_types))

    def _calculate_url_boost(self, url: str, query: str) -> float:
        """Calculate boost based on URL patterns and query intent"""
        url_lower = url.lower()
        query_lower = query.lower()
        boost = 1.0
        
        # Penalize excluded patterns
        for pattern in self.url_patterns['exclude']:
            if pattern in url_lower:
                return 0.3
        
        # Boost relevant patterns
        if any(kw in query_lower for kw in ['faculty', 'professor', 'researcher', 'staff', 'who is', 'people']):
            for pattern in self.url_patterns['faculty']:
                if pattern in url_lower:
                    boost *= 2.0
                    break
        
        if any(kw in query_lower for kw in ['lab', 'laboratory', 'group']):
            for pattern in self.url_patterns['lab']:
                if pattern in url_lower:
                    boost *= 1.8
                    break
        
        # Prefer IISc domain
        if 'iisc.ac.in' in url_lower:
            boost *= 1.3
        
        return boost

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        match = re.search(r'https?://([^/]+)', url)
        return match.group(1) if match else ''

    def get_context_for_generation(self, retrieval_results: Dict[str, Any], 
                                   max_context_length: int = 4000) -> str:
        """
        Format retrieval results into context string for LLM generation
        
        Args:
            retrieval_results: Results from retrieve()
            max_context_length: Maximum length of context in words
        
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        # Add vector search results (from multi-strategy fusion)
        for i, doc in enumerate(retrieval_results.get('vector_results', []), 1):
            text = doc.get('text', '')
            words = text.split()
            
            if current_length + len(words) > max_context_length:
                remaining = max_context_length - current_length
                text = ' '.join(words[:remaining])
            
            # Include metadata for better context
            title = doc.get('title', 'Untitled')
            page_type = doc.get('page_type', 'general')
            source = doc.get('source', 'Unknown')
            
            context_parts.append(f"[Document {i} - {page_type}] {title}\n{text}\nSource: {source}")
            current_length += len(words)
            
            if current_length >= max_context_length:
                break
        
        # Add knowledge graph results
        for result in retrieval_results.get('graph_results', [])[:5]:
            try:
                name = result.get('entity_name', 'Entity')
                etype = result.get('entity_type', 'Unknown')
                mentions = result.get('total_mentions', 0)
                
                context_parts.append(f"[Knowledge Graph] {name} ({etype}) - {mentions} mentions")
            except Exception:
                pass
        
        return '\n\n'.join(context_parts)
