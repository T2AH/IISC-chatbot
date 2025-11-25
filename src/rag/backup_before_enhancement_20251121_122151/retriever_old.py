"""
Enhanced Multi-Strategy Retriever with Reciprocal Rank Fusion
Combines dense, lexical, graph, and targeted retrieval methods
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


class HybridRetriever:
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
    'biochem': [
        'department of biochemistry',
        'biochemistry department',
        'biochemistry'
    ],
    'caf': [
        'central animal facility'
    ],
    'ces': [
        'centre for ecological sciences',
        'ecological sciences centre',
        'ecological sciences'
    ],
    'cidr': [
        'centre for infectious disease research',
        'infectious disease research centre',
        'infectious disease research'
    ],
    'cns': [
        'centre for neuroscience',
        'neuroscience centre',
        'neuroscience'
    ],
    'mcb': [
        'department of microbiology and cell biology',
        'microbiology and cell biology department',
        'microbiology and cell biology'
    ],
    'mbu': [
        'molecular biophysics unit',
        'molecular biophysics'
    ],
    'dbg': [
        'department of developmental biology and genetics',
        'developmental biology and genetics department',
        'developmental biology and genetics'
    ],
    'ipc': [
        'department of inorganic and physical chemistry',
        'inorganic and physical chemistry department',
        'inorganic and physical chemistry'
    ],
    'mrc': [
        'materials research centre',
        'materials research'
    ],
    'orgchem': [
        'department of organic chemistry',
        'organic chemistry department',
        'organic chemistry'
    ],
    'sscu': [
        'solid state and structural chemistry unit',
        'solid state chemistry',
        'structural chemistry'
    ],
    'csa': [
        'computer science and automation',
        'csa department'
    ],
    'ece': [
        'electrical communication engineering',
        'ece department',
        'electrical communication'
    ],
    'dese': [
        'department of electronic systems engineering',
        'electronic systems engineering department',
        'electronic systems engineering'
    ],
    'ee': [
        'electrical engineering',
        'electrical engineering department'
    ],
    'cistup': [
        'centre for infrastructure, sustainable transportation and urban planning',
        'infrastructure sustainable transportation and urban planning centre'
    ],
    'be': [
        'bioengineering',
        'bioengineering department'
    ],
    'csp': [
        'centre for sustainable technologies',
        'sustainable technologies centre',
        'sustainable technologies'
    ],
    'cense': [
        'centre for nanoscience and engineering',
        'nanoscience and engineering centre',
        'nanoscience and engineering'
    ],
    'cds': [
        'computational and data sciences',
        'computational data sciences',
        'cds department'
    ],
    'mgmt': [
        'management studies',
        'management department'
    ],
    'icer': [
        'interdisciplinary centre for energy research',
        'energy research centre',
        'energy research'
    ],
    'icwar': [
        'interdisciplinary centre for water research',
        'water research centre',
        'water research'
    ],
    'cps': [
        'centre for contemporary studies',
        'contemporary studies centre',
        'contemporary studies'
    ],
    'msci': [
        'department of materials science',
        'materials science department',
        'materials science'
    ],
    'serc': [
        'supercomputer education and research centre',
        'supercomputer research centre'
    ],
    'iqti': [
        'international centre for quantum technology initiatives',
        'quantum technology initiatives centre'
    ],
    'abcmc': [
        'atomic, biomolecular and chemical sciences centre',
        'chemical sciences centre'
    ],
    'longevity': [
        'centre for longevity research',
        'longevity research centre',
        'longevity research'
    ],
    'aero': [
        'aerospace engineering',
        'aerospace engineering department'
    ],
    'caos': [
        'centre for atmospheric and oceanic sciences',
        'atmospheric and oceanic sciences centre',
        'atmospheric and oceanic sciences'
    ],
    'ceas': [
        'centre for earth sciences',
        'earth sciences centre',
        'earth sciences'
    ],
    'camm': [
        'centre for advanced manufacturing and materials',
        'advanced manufacturing and materials centre'
    ],
    'dm': [
        'department of mathematics',
        'mathematics department',
        'mathematics'
    ],
    'cst': [
        'centre for scientific teaching',
        'scientific teaching centre'
    ],
    'chemeng': [
        'chemical engineering',
        'chemical engineering department'
    ],
    'civil': [
        'civil engineering',
        'civil engineering department'
    ],
    'dccc': [
        'digital campus and cloud computing centre',
        'cloud computing centre'
    ],
    'materials': [
        'materials engineering',
        'materials engineering department'
    ],
    'mecheng': [
        'mechanical engineering',
        'mechanical engineering department'
    ],
    'physics_jap': [
        'department of physics (japan group page)',
        'physics japan group'
    ],
    'cct': [
        'centre for catalysis and transition metal chemistry',
        'catalysis and transition metal chemistry centre'
    ],
    'chep': [
        'centre for high energy physics',
        'high energy physics centre'
    ],
    'math': [
        'mathematics department',
        'mathematics'
    ],
    'iap': [
        'instrumentation and applied physics',
        'applied physics',
        'instrumentation'
    ],
    'physics': [
        'department of physics',
        'physics department',
        'physics'
    ],
    'cbr': [
        'centre for brain research',
        'brain research centre',
        'brain research'
    ],
    'fsid': [
        'foundation for science, innovation and development',
        'science innovation and development foundation'
    ],
    'diarcoe': [
        'department of interdisciplinary and applied research in chemical engineering',
        'interdisciplinary and applied chemical engineering research department'
    ]
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

    def _detect_departments(self, query: str) -> List[str]:
        """
        Detect department tokens from the query using the alias map.

        Returns a list of dept keys (from self.dept_aliases) that match whole words in the query.
        """
        q = (query or '').lower()
        matched = []
        try:
            for key, aliases in self.dept_aliases.items():
                for alias in aliases:
                    # match whole words / phrases
                    if re.search(rf"\b{re.escape(alias)}\b", q):
                        matched.append(key)
                        break
        except Exception:
            return []
        return matched
    
    def _calculate_url_boost(self, url: str, query: str) -> float:
        """
        Calculate boost score based on URL patterns and query intent
        
        Args:
            url: Document URL
            query: User query
        
        Returns:
            Boost multiplier (1.0 = no change, >1.0 = boost, <1.0 = penalize)
        """
        url_lower = url.lower()
        query_lower = query.lower()
        boost = 1.0
        
        # Penalize excluded patterns (forms, applications)
        for pattern in self.url_patterns['exclude']:
            if pattern in url_lower:
                boost *= 0.3  # Heavy penalty
                logger.debug(f"Penalizing URL (excluded pattern): {url}")
                return boost
        
        # Boost faculty pages if query is about faculty/people
        if any(kw in query_lower for kw in ['faculty', 'professor', 'researcher', 'staff', 'who is', 'people', 'list']):
            for pattern in self.url_patterns['faculty']:
                if pattern in url_lower:
                    boost *= 2.5  # Stronger boost for faculty pages
                    logger.debug(f"Boosting faculty URL: {url}")
                    break
        
        # Boost lab pages if query is about labs/groups
        if any(kw in query_lower for kw in ['lab', 'laboratory', 'group', 'center', 'centre']):
            for pattern in self.url_patterns['lab']:
                if pattern in url_lower:
                    boost *= 2.2  # Stronger boost
                    logger.debug(f"Boosting lab URL: {url}")
                    break
        
        # Boost department pages if query is about departments
        if 'department' in query_lower or 'dept' in query_lower:
            for pattern in self.url_patterns['department']:
                if pattern in url_lower:
                    boost *= 1.5
                    logger.debug(f"Boosting department URL: {url}")
                    break
        
        # Always boost IISc pages over external institutions
        if 'iisc.ac.in' in url_lower:
            boost *= 1.3  # Prefer IISc pages
            logger.debug(f"Boosting IISc domain: {url}")
        elif any(ext in url_lower for ext in ['iitm.ac.in', 'iitb.ac.in', 'mit.edu', 'stanford.edu']):
            boost *= 0.7  # Penalize external institution pages
            logger.debug(f"Penalizing external institution: {url}")
        
        return boost
    
    def retrieve(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retrieve relevant context using both vector search and knowledge graph
        
        Args:
            query: User query
            filters: Optional metadata filters for vector search
        
        Returns:
            Dictionary with retrieved contexts from both sources
        """
        logger.info(f"Retrieving context for query: {query[:100]}...")

        # Detect department-like tokens from query (non-hardcoded detection)
        matched_depts = self._detect_departments(query)
        if matched_depts:
            logger.debug(f"Detected department tokens: {matched_depts}")

        # 1. Vector search in ChromaDB (pass matched_depts for optional metadata filtering)
        vector_results = self._vector_search(query, filters, matched_depts=matched_depts)
        
        # 2. Aggregate chunks from same source if enabled
        if self.merge_same_source and vector_results:
            vector_results = self._aggregate_same_source(vector_results[:self.max_chunks_per_source])
        
        # 3. Knowledge graph search in Neo4j (exact-match-first if possible)
        graph_results = self._graph_search(query, matched_depts=matched_depts)
        
        # 3. Combine results
        combined_results = {
            'query': query,
            'vector_results': vector_results,
            'graph_results': graph_results,
            'matched_depts': matched_depts,
            'num_vector_results': len(vector_results),
            'num_graph_results': len(graph_results)
        }
        
        logger.info(f"Retrieved {len(vector_results)} vector results and {len(graph_results)} graph results")
        
        return combined_results
    
    def _vector_search(self, query: str, filters: Dict[str, Any] = None, matched_depts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search in ChromaDB
        
        Args:
            query: Search query
            filters: Optional metadata filters
        
        Returns:
            List of relevant documents
        """
        try:
            if not self.chromadb.collection:
                logger.warning("ChromaDB not available")
                return []
            
            # Query expansion for better retrieval
            expanded_query = self._expand_query_with_acronyms(query)
            query_lower = query.lower()
            
            # Generic query expansion based on intent patterns
            faculty_keywords = ["faculty", "instructor", "professor", "lead", "head", "pi", "principal investigator", "director", "researcher", "scientist"]
            if any(keyword in query_lower for keyword in faculty_keywords):
                expanded_query += " faculty professor researcher scientist lead head director staff"
            
            # Expand lab/group queries
            if "lab" in query_lower or "laboratory" in query_lower or "group" in query_lower:
                expanded_query += " laboratory research group center centre team"
            
            # Expand department queries
            if "department" in query_lower:
                expanded_query += " department division school institute faculty staff"
            
            # Expand person name queries (detect proper nouns indicating names)
            words = query.split()
            proper_nouns = [w for w in words if len(w) > 2 and w[0].isupper() and not w.isupper()]
            if len(proper_nouns) >= 2:  # Likely a person's name
                expanded_query += " professor faculty researcher scientist staff member"
            
            # Generate query embedding
            if not self.query_embedder:
                logger.error("Query embedder not available")
                return []
            
            query_embedding = self.query_embedder.embed(expanded_query)
            
            # Build optional metadata filter for Chromadb if departments detected
            chroma_filters = filters
            if matched_depts:
                # Prefer documents whose metadata.domain contains dept token or page_type signals department/faculty-list
                dept_domains = [f"{d}.iisc.ac.in" for d in matched_depts]
                chroma_filters = chroma_filters or {}
                # Some chromadb clients use 'where' dict, others use custom 'filters' parameter - try to be flexible
                chroma_filters['$or'] = chroma_filters.get('$or', []) + [
                    {'domain': {'$in': dept_domains}},
                    {'page_type': {'$in': ['department', 'faculty-list', 'faculty']}}
                ]

            # Query ChromaDB with pre-computed embedding
            results = self.chromadb.query(
                query_embedding=query_embedding,
                n_results=self.top_k_vectors,
                where=chroma_filters
            )
            
            # Format results (distance is L2 distance, lower is better)
            documents = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                url = metadata.get('url', 'Unknown')
                
                # Convert L2 distance to similarity score (approximate)
                base_similarity = 1 / (1 + distance)
                
                # Apply URL-based boosting
                url_boost = self._calculate_url_boost(url, query)
                boosted_score = base_similarity * url_boost
                
                doc = {
                    'text': results['documents'][0][i],
                    'metadata': metadata,
                    'distance': distance,
                    'base_score': base_similarity,
                    'url_boost': url_boost,
                    'score': boosted_score,  # Final score after boosting
                    'source': url,
                    'title': metadata.get('title', 'Unknown'),
                    'domain': metadata.get('domain', 'Unknown'),
                    'page_type': metadata.get('page_type', 'general')
                }
                # Mark whether this document appears to match a detected department (for downstream reranking)
                try:
                    src = (url or '').lower()
                    meta_domain = metadata.get('domain', '').lower()
                    doc['_meta_matches_dept'] = False
                    if matched_depts:
                        for d in matched_depts:
                            if d in src or d in meta_domain:
                                doc['_meta_matches_dept'] = True
                                break
                except Exception:
                    doc['_meta_matches_dept'] = False
                documents.append(doc)
            
            # Re-sort by boosted scores (highest first)
            documents.sort(key=lambda x: x['score'], reverse=True)
            
            logger.debug(f"Applied URL boosting - top result: {documents[0]['source'] if documents else 'None'}")
            
            return documents
        
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def _graph_search(self, query: str, matched_depts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search knowledge graph for relevant entities and relationships
        
        Args:
            query: Search query
        
        Returns:
            List of relevant graph results
        """
        try:
            if not self.neo4j.driver:
                logger.warning("Neo4j not available")
                return []
            
            results = []

            # First try exact-match entity lookup (case-insensitive) to prefer precise graph answers
            with self.neo4j.driver.session() as session:
                try:
                    exact_q = query.strip()
                    exact_cypher = (
                        "MATCH (e:Entity)"
                        " WHERE toLower(e.name) = toLower($q) OR toLower(e.alias) = toLower($q)"
                        " RETURN e.name AS entity_name, e.type AS entity_type, e LIMIT $limit"
                    )
                    exact_res = session.run(exact_cypher, q=exact_q, limit=self.top_k_graph)
                    for rec in exact_res:
                        results.append({
                            'entity_name': rec['entity_name'],
                            'entity_type': rec['entity_type'],
                            'page_count': None,
                            'total_mentions': None,
                            'sample_urls': [],
                            'sample_titles': [],
                            'source': 'knowledge_graph',
                        })
                    if results:
                        return results
                except Exception:
                    # If exact-match query fails for any reason, continue to broader search
                    pass

                # If no exact match, fall back to enhanced mentions-based search
                # Use all significant query terms for graph search
                query_expanded = query.lower()
                
                # Extract meaningful words (3+ chars, not common stopwords)
                stopwords = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'about', 'all', 'list', 'all', 'list'}
                query_words = [w.lower() for w in query.split() if len(w) > 2 and w.lower() not in stopwords]
                
                # Add generic expansion for certain patterns
                if 'lab' in query_expanded or 'laboratory' in query_expanded:
                    query_words.extend(['lab', 'laboratory', 'research', 'group'])
                if 'department' in query_expanded:
                    query_words.extend(['department', 'division', 'school'])
                
                # Remove duplicates
                search_terms = list(set(query_words))
                query_expanded = ' '.join(search_terms)
                
                # Enhanced entity search with better matching
                cypher_query = """
                MATCH (p:Page)-[r:MENTIONS]->(e:Entity)
                WHERE any(term IN split(toLower($query_text), ' ') 
                      WHERE toLower(e.name) CONTAINS term OR 
                            toLower(e.type) CONTAINS term OR
                            any(word IN split(toLower(e.name), ' ') WHERE word CONTAINS term))
                WITH e, count(DISTINCT p) as page_count, sum(r.count) as total_mentions,
                     collect(DISTINCT {url: p.url, title: p.title})[..5] as sample_pages
                ORDER BY total_mentions DESC, page_count DESC
                LIMIT $limit
                RETURN e.name as entity_name, e.type as entity_type, 
                       page_count, total_mentions, 
                       [page IN sample_pages | page.url] as sample_urls,
                       [page IN sample_pages | page.title] as sample_titles
                """
                
                graph_results = session.run(cypher_query, query_text=query_expanded, limit=self.top_k_graph)
                
                for record in graph_results:
                    results.append({
                        'entity_name': record['entity_name'],
                        'entity_type': record['entity_type'],
                        'page_count': record['page_count'],
                        'total_mentions': record['total_mentions'],
                        'sample_urls': record.get('sample_urls', []),
                        'sample_titles': record.get('sample_titles', []),
                        'source': 'knowledge_graph'
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Error in graph search: {e}")
            return []
    
    def retrieve_by_page_type(self, query: str, page_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents filtered by page type
        
        Args:
            query: Search query
            page_type: Page type filter (e.g., 'faculty', 'lab', 'course')
        
        Returns:
            List of relevant documents
        """
        filters = {'page_type': page_type}
        results = self.retrieve(query, filters=filters)
        return results['vector_results']
    
    def retrieve_faculty_info(self, faculty_name: str) -> Dict[str, Any]:
        """
        Retrieve comprehensive information about a faculty member
        
        Args:
            faculty_name: Name of faculty member
        
        Returns:
            Dictionary with faculty information
        """
        try:
            # Query knowledge graph
            faculty_info = self.neo4j.query_faculty(faculty_name)
            
            if faculty_info:
                # Also get relevant documents from vector DB
                vector_results = self._vector_search(faculty_name, filters={'page_type': 'faculty'})
                
                faculty_info['related_documents'] = vector_results
                return faculty_info
            
            return {}
        
        except Exception as e:
            logger.error(f"Error retrieving faculty info: {e}")
            return {}
    
    def _expand_query_with_acronyms(self, query: str) -> str:
        """
        Expand query with full forms of common acronyms
        
        Args:
            query: Original query
        
        Returns:
            Expanded query with acronym expansions
        """
        #IISc department and research area acronyms
        acronym_map = {
            'biochem': 'department of biochemistry',
            'caf': 'central animal facility',
            'ces': 'centre for ecological sciences',
            'cidr': 'centre for infectious disease research',
            'cns': 'centre for neuroscience',
            'mcb': 'department of microbiology and cell biology',
            'mbu': 'molecular biophysics unit',
            'dbg': 'department of developmental biology and genetics',
            'ipc': 'department of inorganic and physical chemistry',
            'mrc': 'materials research centre',
            'orgchem': 'department of organic chemistry',
            'sscu': 'solid state and structural chemistry unit',
            'csa': 'computer science and automation',
            'ece': 'electrical communication engineering',
            'dese': 'department of electronic systems engineering',
            'ee': 'electrical engineering',
            'cistup': 'centre for infrastructure, sustainable transportation and urban planning',
            'be': 'bioengineering',
            'csp': 'centre for sustainable technologies',
            'cense': 'centre for nanoscience and engineering',
            'cds': 'computational and data sciences',
            'mgmt': 'management studies',
            'icer': 'interdisciplinary centre for energy research',
            'icwar': 'interdisciplinary centre for water research',
            'cps': 'centre for contemporary studies',
            'msci': 'department of materials science',
            'serc': 'supercomputer education and research centre',
            'iqti': 'international centre for quantum technology initiatives',
            'abcmc': 'atomic, biomolecular and chemical sciences centre',
            'longevity': 'centre for longevity research',
            'aero': 'aerospace engineering',
            'caos': 'centre for atmospheric and oceanic sciences',
            'ceas': 'centre for earth sciences',
            'camm': 'centre for advanced manufacturing and materials',
            'dm': 'department of mathematics',
            'cst': 'centre for scientific teaching',
            'chemeng': 'chemical engineering',
            'civil': 'civil engineering',
            'dccc': 'digital campus and cloud computing centre',
            'materials': 'materials engineering',
            'mecheng': 'mechanical engineering',
            'physics_jap': 'department of physics (japan group page)',
            'cct': 'centre for catalysis and transition metal chemistry',
            'chep': 'centre for high energy physics',
            'math': 'mathematics department',
            'iap': 'instrumentation and applied physics',
            'physics': 'department of physics',
            'cbr': 'centre for brain research',
            'fsid': 'foundation for science, innovation and development',
            'diarcoe': 'department of interdisciplinary and applied research in chemical engineering',
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'hpc': 'high performance computing',
            'iot': 'internet of things'
        }
        
        expanded = query
        for acronym, full_form in acronym_map.items():
            # Match acronym as whole word (case insensitive)
            if re.search(rf'\b{acronym}\b', query, re.IGNORECASE):
                expanded += f" {full_form}"
                logger.debug(f"Expanded acronym '{acronym}' to '{full_form}'")
        
        # Research area specific expansions
        if any(term in query.lower() for term in ['distributed', 'cloud', 'parallel', 'scalable']):
            expanded += " distributed systems scalable computing parallel processing"
        
        return expanded
    
    def _aggregate_same_source(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge chunks from the same source URL for complete context
        
        Args:
            documents: List of retrieved documents
        
        Returns:
            Aggregated documents with merged text from same sources
        """
        try:
            # Group by source URL
            source_groups = defaultdict(list)
            for doc in documents:
                source_url = doc.get('source', doc.get('metadata', {}).get('url', 'unknown'))
                source_groups[source_url].append(doc)
            
            aggregated = []
            for source_url, docs in source_groups.items():
                if len(docs) == 1:
                    # Single chunk - use as is
                    aggregated.append(docs[0])
                else:
                    # Multiple chunks - merge them
                    # Sort by chunk_id to maintain order
                    try:
                        docs.sort(key=lambda x: int(x.get('metadata', {}).get('chunk_id', 0)))
                    except (ValueError, TypeError):
                        pass  # Keep original order if chunk_id not numeric
                    
                    # Merge text with clear separators
                    merged_text = '\n\n---\n\n'.join([d['text'] for d in docs])
                    
                    # Use best metadata from all chunks
                    merged_doc = {
                        'text': merged_text,
                        'metadata': docs[0].get('metadata', {}),
                        'score': max([d.get('score', 0) for d in docs]),  # Best score
                        'base_score': max([d.get('base_score', 0) for d in docs]),
                        'source': source_url,
                        'title': docs[0].get('title', 'Unknown'),
                        'domain': docs[0].get('domain', 'Unknown'),
                        'page_type': docs[0].get('page_type', 'general'),
                        'chunk_count': len(docs)  # Track how many chunks merged
                    }
                    aggregated.append(merged_doc)
                    logger.debug(f"Merged {len(docs)} chunks from {source_url}")
            
            # Re-sort by score (highest first)
            aggregated.sort(key=lambda x: x.get('score', 0), reverse=True)
            logger.info(f"Aggregated {len(documents)} documents into {len(aggregated)} sources")
            return aggregated
        
        except Exception as e:
            logger.error(f"Error aggregating sources: {e}")
            return documents  # Return original if aggregation fails
    
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
        
        # Add vector search results
        for i, doc in enumerate(retrieval_results.get('vector_results', []), 1):
            text = doc['text']
            words = text.split()
            
            if current_length + len(words) > max_context_length:
                # Truncate to fit
                remaining = max_context_length - current_length
                text = ' '.join(words[:remaining])
            
            context_parts.append(f"[Document {i}] {text}")
            current_length += len(words)
            
            if current_length >= max_context_length:
                break
        
        # Add knowledge graph results (include generically regardless of exact shape)
        for result in retrieval_results.get('graph_results', []):
            try:
                name = result.get('entity_name') or result.get('name') or 'Entity'
                etype = result.get('entity_type') or result.get('type') or 'Unknown'
                page_count = result.get('page_count')
                total_mentions = result.get('total_mentions')
                sample_titles = result.get('sample_titles', []) or []
                sample_urls = result.get('sample_urls', []) or []

                parts = [f"[Knowledge Graph] {name} ({etype})"]
                if page_count is not None:
                    parts.append(f"mentioned_in_pages={page_count}")
                if total_mentions is not None:
                    parts.append(f"total_mentions={total_mentions}")
                if sample_titles:
                    parts.append(f"related_pages: {', '.join(sample_titles[:3])}")
                if sample_urls:
                    parts.append(f"sources: {', '.join(sample_urls[:2])}")

                context_parts.append(' | '.join(parts))
            except Exception:
                # Fallback: stringify the dict
                context_parts.append(f"[Knowledge Graph] {str(result)}")
        
        return '\n\n'.join(context_parts)
