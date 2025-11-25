"""
Keyword extraction using KeyBERT
"""

from typing import List, Dict, Tuple, Any
from loguru import logger


class KeywordExtractor:
    """Extract keywords from text using KeyBERT"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", top_n: int = 10, diversity: float = 0.5):
        """
        Initialize keyword extractor
        
        Args:
            model_name: Name of sentence transformer model to use
            top_n: Number of keywords to extract
            diversity: Diversity of keywords (0-1, higher = more diverse)
        """
        self.model_name = model_name
        self.top_n = top_n
        self.diversity = diversity
        self.kw_model = None
        self._load_model()
    
    def _load_model(self):
        """Load KeyBERT model"""
        try:
            from keybert import KeyBERT
            self.kw_model = KeyBERT(model=self.model_name)
            logger.info(f"Loaded KeyBERT model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load KeyBERT model: {e}")
            self.kw_model = None
    
    def extract_keywords(self, text: str, top_n: int = None, 
                        use_mmr: bool = True, diversity: float = None) -> List[Tuple[str, float]]:
        """
        Extract keywords from text
        
        Args:
            text: Text to extract keywords from
            top_n: Number of keywords to extract (overrides default)
            use_mmr: Use Maximal Marginal Relevance for diversity
            diversity: Diversity parameter (overrides default)
        
        Returns:
            List of (keyword, score) tuples
        """
        if not text or not self.kw_model:
            return []
        
        try:
            # Use default values if not provided
            n = top_n if top_n is not None else self.top_n
            div = diversity if diversity is not None else self.diversity
            
            # Limit text for keyword extraction (10K chars is sufficient)
            text_sample = text[:10000] if len(text) > 10000 else text
            
            # Extract keywords
            if use_mmr:
                keywords = self.kw_model.extract_keywords(
                    text_sample,
                    keyphrase_ngram_range=(1, 2),  # Reduced from (1,3) for speed
                    stop_words='english',
                    use_mmr=True,
                    diversity=div,
                    top_n=n
                )
            else:
                keywords = self.kw_model.extract_keywords(
                    text_sample,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=n
                )
            
            logger.debug(f"Extracted {len(keywords)} keywords")
            return keywords
        
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def extract_research_keywords(self, text: str) -> Dict[str, Any]:
        """
        Extract research-specific keywords with categorization
        
        Args:
            text: Text to extract keywords from
        
        Returns:
            Dictionary with categorized keywords
        """
        if not text or not self.kw_model:
            return {'keywords': [], 'research_areas': [], 'methods': [], 'technologies': []}
        
        # Extract all keywords
        all_keywords = self.extract_keywords(text, top_n=20)
        
        # Research area keywords
        research_areas = [
            'machine learning', 'deep learning', 'computer vision', 'nlp',
            'artificial intelligence', 'data science', 'algorithms', 'theory',
            'systems', 'networks', 'security', 'databases', 'software engineering'
        ]
        
        # Method keywords
        methods = [
            'supervised learning', 'unsupervised learning', 'reinforcement learning',
            'neural network', 'classification', 'regression', 'clustering',
            'optimization', 'simulation', 'modeling', 'analysis'
        ]
        
        # Technology keywords
        technologies = [
            'python', 'tensorflow', 'pytorch', 'java', 'c++',
            'gpu', 'cloud', 'distributed', 'parallel', 'cuda'
        ]
        
        # Categorize keywords
        categorized = {
            'keywords': [kw for kw, score in all_keywords],
            'research_areas': [],
            'methods': [],
            'technologies': []
        }
        
        text_lower = text.lower()
        
        for area in research_areas:
            if area in text_lower:
                categorized['research_areas'].append(area)
        
        for method in methods:
            if method in text_lower:
                categorized['methods'].append(method)
        
        for tech in technologies:
            if tech in text_lower:
                categorized['technologies'].append(tech)
        
        return categorized
    
    def get_keyword_summary(self, keywords: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Get summary statistics for keywords
        
        Args:
            keywords: List of (keyword, score) tuples
        
        Returns:
            Summary statistics
        """
        if not keywords:
            return {
                'total_keywords': 0,
                'avg_score': 0.0,
                'top_keyword': None,
                'score_distribution': {}
            }
        
        scores = [score for _, score in keywords]
        
        summary = {
            'total_keywords': len(keywords),
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'top_keyword': keywords[0][0] if keywords else None,
            'top_keywords': [kw for kw, _ in keywords[:5]]
        }
        
        return summary
