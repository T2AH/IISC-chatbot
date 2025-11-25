"""
Entity extraction using spaCy and BERT
"""

import spacy
from typing import List, Dict, Any, Set
from loguru import logger


class EntityExtractor:
    """Extract entities from text using spaCy and custom patterns"""
    
    def __init__(self, spacy_model: str = "en_core_web_lg"):
        """
        Initialize entity extractor
        
        Args:
            spacy_model: Name of spaCy model to use
        """
        self.model_name = spacy_model
        self.nlp = None
        self._load_model()
        
        # Research domain patterns
        self.research_patterns = {
            'RESEARCH_TOPIC': [
                'machine learning', 'deep learning', 'artificial intelligence',
                'computer vision', 'natural language processing', 'nlp',
                'data science', 'algorithms', 'optimization', 'theory',
                'networks', 'security', 'cryptography', 'databases',
                'software engineering', 'systems', 'architecture',
                'computational biology', 'bioinformatics', 'genomics'
            ],
            'LAB_NAME_PATTERNS': [
                r'\b\w+\s+Lab(?:oratory)?\b',
                r'\b\w+\s+Research\s+Group\b',
                r'\b\w+\s+Research\s+Lab\b',
                r'\bLab\s+for\s+\w+\b'
            ],
            'COURSE_CODE_PATTERNS': [
                r'\b[A-Z]{2,4}[\s-]?\d{3,4}\b',  # e.g., CS6190, E1-245
                r'\b[A-Z]\d-\d{3}\b'
            ]
        }
    
    def _load_model(self):
        """Load spaCy model"""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.warning(f"spaCy model '{self.model_name}' not found. Attempting to download...")
            try:
                import subprocess
                subprocess.run(['python', '-m', 'spacy', 'download', self.model_name], check=True)
                self.nlp = spacy.load(self.model_name)
                logger.info(f"Downloaded and loaded spaCy model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to download spaCy model: {e}")
                # Fallback to smaller model
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.warning("Using fallback model: en_core_web_sm")
                except:
                    logger.error("No spaCy model available. Entity extraction will be limited.")
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities from text
        
        Args:
            text: Text to extract entities from
        
        Returns:
            Dictionary mapping entity types to lists of entity dictionaries
        """
        if not text or not self.nlp:
            return {}
        
        # Limit text length for performance - 50K chars is sufficient for entity extraction
        # This prevents spaCy from hanging on very large documents while maintaining quality
        text_to_process = text[:50000] if len(text) > 50000 else text
        
        # Process text with spaCy (disable parser and lemmatizer for speed)
        doc = self.nlp(text_to_process, disable=['parser', 'lemmatizer'])
        
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entity
            'RESEARCH_TOPIC': [],
            'LAB_NAME': [],
            'COURSE_CODE': [],
            'DATE': [],
            'PRODUCT': []
        }
        
        # Extract standard entities
        seen = set()  # Track seen entities to avoid duplicates
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entity_text = ent.text.strip()
                if entity_text and entity_text.lower() not in seen:
                    entities[ent.label_].append({
                        'text': entity_text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
                    seen.add(entity_text.lower())
        
        # Extract research topics using keyword matching
        text_lower = text.lower()
        for topic in self.research_patterns['RESEARCH_TOPIC']:
            if topic in text_lower:
                if topic not in seen:
                    entities['RESEARCH_TOPIC'].append({
                        'text': topic,
                        'label': 'RESEARCH_TOPIC',
                        'start': text_lower.index(topic),
                        'end': text_lower.index(topic) + len(topic)
                    })
                    seen.add(topic)
        
        # Extract lab names using patterns
        import re
        for pattern in self.research_patterns['LAB_NAME_PATTERNS']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                lab_name = match.group().strip()
                if lab_name.lower() not in seen:
                    entities['LAB_NAME'].append({
                        'text': lab_name,
                        'label': 'LAB_NAME',
                        'start': match.start(),
                        'end': match.end()
                    })
                    seen.add(lab_name.lower())
        
        # Extract course codes
        for pattern in self.research_patterns['COURSE_CODE_PATTERNS']:
            for match in re.finditer(pattern, text):
                course_code = match.group().strip()
                if course_code.lower() not in seen:
                    entities['COURSE_CODE'].append({
                        'text': course_code,
                        'label': 'COURSE_CODE',
                        'start': match.start(),
                        'end': match.end()
                    })
                    seen.add(course_code.lower())
        
        # Remove empty entity types
        entities = {k: v for k, v in entities.items() if v}
        
        logger.debug(f"Extracted {sum(len(v) for v in entities.values())} entities")
        
        return entities
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text
        
        Args:
            text: Text to extract noun phrases from
        
        Returns:
            List of noun phrases
        """
        if not text or not self.nlp:
            return []
        
        # Limit text for performance
        text_to_process = text[:50000] if len(text) > 50000 else text
        doc = self.nlp(text_to_process)
        
        # Extract noun chunks
        noun_phrases = [chunk.text.strip() for chunk in doc.noun_chunks]
        
        # Filter and deduplicate
        noun_phrases = list(set([np for np in noun_phrases if len(np.split()) >= 2]))
        
        return noun_phrases
    
    def get_entity_summary(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Get summary statistics for entities
        
        Args:
            entities: Entity dictionary from extract_entities
        
        Returns:
            Summary statistics
        """
        summary = {
            'total_entities': sum(len(v) for v in entities.values()),
            'entity_types': len(entities),
            'counts_by_type': {k: len(v) for k, v in entities.items()}
        }
        
        return summary
