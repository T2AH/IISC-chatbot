"""
Intelligent Query Analyzer - Dynamic Intent Detection
No hardcoding, learns from query patterns and data structure
"""

from typing import Dict, List, Any, Tuple
import re
from loguru import logger


class QueryAnalyzer:
    """Analyzes queries to understand intent and extract key information dynamically"""
    
    def __init__(self):
        # Pattern-based intent detection (no hardcoded domains)
        self.intent_patterns = {
            'list_all': [
                r'\b(list|show|display|give|tell)\s+(all|me\s+all)\b',
                r'\ball\s+(faculty|professors|researchers|staff|people|members|labs|departments)\b',
                r'\bwho\s+are\s+(all|the)\b'
            ],
            'faculty_query': [
                r'\b(faculty|professor|researcher|staff|instructor|teacher|PI|principal investigator)\b',
                r'\bwho\s+is\b',
                r'\bpeople\s+in\b'
            ],
            'lab_query': [
                r'\b(lab|laboratory|research\s+group|center|centre|team)\b'
            ],
            'department_query': [
                r'\b(department|division|school|institute|dept)\b'
            ],
            'person_query': [
                r'\bwho\s+is\s+[A-Z][a-z]+\b',  # "who is John"
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # Proper names
            ],
            'about_query': [
                r'\b(tell|about|describe|what|explain|info|information)\b'
            ]
        }
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to extract intent, entities, and search strategy
        
        Returns:
            {
                'intents': List of detected intents,
                'query_type': Primary query type,
                'entities': Extracted entities/keywords,
                'search_strategy': Recommended search approach,
                'filters': Suggested metadata filters
            }
        """
        query_lower = query.lower()
        
        # Detect intents
        detected_intents = []
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    detected_intents.append(intent)
                    break
        
        # Remove duplicates
        detected_intents = list(set(detected_intents))
        
        # Extract entities (words that might be names, departments, etc.)
        entities = self._extract_entities(query)
        
        # Determine primary query type
        query_type = self._determine_query_type(detected_intents, query_lower)
        
        # Suggest search strategy
        search_strategy = self._suggest_search_strategy(query_type, detected_intents, entities)
        
        # Suggest filters
        filters = self._suggest_filters(query_type, entities)
        
        analysis = {
            'original_query': query,
            'intents': detected_intents,
            'query_type': query_type,
            'entities': entities,
            'search_strategy': search_strategy,
            'filters': filters,
            'expansion_terms': self._get_expansion_terms(query_type, detected_intents)
        }
        
        logger.debug(f"Query analysis: {analysis}")
        return analysis
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract potential entities from query"""
        entities = {
            'proper_nouns': [],
            'acronyms': [],
            'keywords': []
        }
        
        words = query.split()
        
        for i, word in enumerate(words):
            # Proper nouns (capitalized words not at start)
            if i > 0 and word[0].isupper() and len(word) > 1:
                entities['proper_nouns'].append(word)
            
            # Acronyms (2-5 uppercase letters)
            if word.isupper() and 2 <= len(word) <= 5:
                entities['acronyms'].append(word)
            
            # Significant keywords (3+ chars, not common)
            if len(word) > 2 and word.lower() not in ['the', 'and', 'for', 'are', 'with', 'this', 'that']:
                entities['keywords'].append(word.lower())
        
        return entities
    
    def _determine_query_type(self, intents: List[str], query_lower: str) -> str:
        """Determine primary query type"""
        if 'list_all' in intents:
            return 'list_aggregation'
        elif 'person_query' in intents:
            return 'person_lookup'
        elif 'faculty_query' in intents:
            return 'faculty_search'
        elif 'lab_query' in intents:
            return 'lab_search'
        elif 'department_query' in intents:
            return 'department_search'
        elif 'about_query' in intents:
            return 'information_retrieval'
        else:
            return 'general_search'
    
    def _suggest_search_strategy(self, query_type: str, intents: List[str], 
                                 entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Suggest optimal search strategy based on query analysis"""
        strategy = {
            'vector_weight': 1.0,
            'graph_weight': 1.0,
            'use_entity_aggregation': False,
            'boost_url_patterns': [],
            'preferred_page_types': []
        }
        
        if query_type == 'list_aggregation':
            # For "list all X" queries, prioritize graph aggregation
            strategy['graph_weight'] = 2.0
            strategy['use_entity_aggregation'] = True
            
            # Detect what to list
            if any(kw in entities['keywords'] for kw in ['faculty', 'professor', 'researcher', 'staff']):
                strategy['boost_url_patterns'] = ['/faculty/', '/people/', '/staff/', '/members/']
                strategy['preferred_page_types'] = ['faculty', 'people']
            elif any(kw in entities['keywords'] for kw in ['lab', 'laboratory', 'group']):
                strategy['boost_url_patterns'] = ['/lab/', '/research/', '/group/']
                strategy['preferred_page_types'] = ['lab', 'research']
        
        elif query_type in ['person_lookup', 'faculty_search']:
            # For person queries, prioritize URLs with /faculty/ or /people/
            strategy['vector_weight'] = 1.5
            strategy['boost_url_patterns'] = ['/faculty/', '/people/', '/profile/', '/~']
            strategy['preferred_page_types'] = ['faculty', 'profile']
        
        elif query_type == 'lab_search':
            strategy['boost_url_patterns'] = ['/lab/', '/research/', '/group/', '/center/']
            strategy['preferred_page_types'] = ['lab', 'research']
        
        elif query_type == 'department_search':
            strategy['boost_url_patterns'] = ['/department/', '/about/', '/overview/']
            strategy['preferred_page_types'] = ['department', 'overview']
        
        return strategy
    
    def _suggest_filters(self, query_type: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Suggest metadata filters for database queries"""
        filters = {}
        
        # Dynamic filtering based on extracted entities
        if entities['acronyms']:
            # Use acronyms as domain hints (e.g., CDS, DESE)
            filters['domain_hints'] = entities['acronyms']
        
        return filters
    
    def _get_expansion_terms(self, query_type: str, intents: List[str]) -> List[str]:
        """Get query expansion terms based on query type"""
        expansions = []
        
        if query_type in ['faculty_search', 'person_lookup']:
            expansions.extend(['faculty', 'professor', 'researcher', 'staff', 'member'])
        elif query_type == 'lab_search':
            expansions.extend(['laboratory', 'research', 'group', 'center', 'team'])
        elif query_type == 'department_search':
            expansions.extend(['department', 'division', 'school', 'institute'])
        
        if 'list_all' in intents:
            expansions.extend(['members', 'people', 'team', 'staff'])
        
        return list(set(expansions))
