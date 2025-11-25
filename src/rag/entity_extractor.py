"""
Generic Entity Extractor with SpaCy NER and Configurable Patterns
"""
import re
import spacy
from typing import Dict, Set, List, Any
from collections import defaultdict
from loguru import logger
from src.config.entity_config import EntityExtractionConfig

class EntityExtractor:
    """Generic entity extraction with NER and configurable patterns"""
    
    def __init__(self, config: EntityExtractionConfig = None):
        """
        Initialize entity extractor
        
        Args:
            config: Entity extraction configuration
        """
        self.config = config or EntityExtractionConfig()
        self.spacy_config = self.config.get_spacy_config()
        self.nlp = self._load_spacy_model()
        self.department_cache = {}
    
    def _load_spacy_model(self):
        """Load SpaCy model"""
        model_name = self.spacy_config.get('model', 'en_core_web_sm')
        try:
            nlp = spacy.load(model_name)
            logger.info(f"✓ SpaCy model '{model_name}' loaded")
            return nlp
        except:
            logger.warning(f"SpaCy model '{model_name}' not found, downloading...")
            import os
            os.system(f"python -m spacy download {model_name}")
            return spacy.load(model_name)
    
    def extract_entities(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Set]:
        """
        Extract entities from text using NER and patterns
        
        Args:
            text: Input text
            metadata: Document metadata
        
        Returns:
            Dictionary of entity types to sets of entity names
        """
        entities = defaultdict(set)
        
        # SpaCy NER extraction
        self._extract_with_spacy(text, entities)
        
        # Pattern-based extraction
        self._extract_with_patterns(text, metadata, entities)
        
        # Extract departments if enabled
        if self.config.is_department_detection_enabled():
            departments = self._extract_departments(metadata.get('url', ''), metadata)
            if departments:
                entities['departments'].update(departments)
        
        # Clean and filter
        entities = self._clean_entities(entities)
        
        return dict(entities)
    
    def _extract_with_spacy(self, text: str, entities: Dict):
        """Extract entities using SpaCy NER"""
        max_length = self.spacy_config.get('max_text_length', 5000)
        min_length = self.spacy_config.get('min_entity_length', 3)
        max_words = self.spacy_config.get('max_name_words', 4)
        
        doc = self.nlp(text[:max_length])
        
        entity_label_mapping = self.spacy_config.get('entity_labels', {})
        
        for ent in doc.ents:
            # Person extraction
            if ent.label_ in entity_label_mapping.get('person', ['PERSON']):
                name = ent.text.strip()
                if len(name) > min_length and len(name.split()) <= max_words:
                    entities['Person'].add(name)
            
            # Organization extraction
            elif ent.label_ in entity_label_mapping.get('organization', ['ORG']):
                org = ent.text.strip()
                if len(org) > min_length:
                    entities['Organization'].add(org)
    
    def _extract_with_patterns(self, text: str, metadata: Dict, entities: Dict):
        """Extract entities using configurable regex patterns"""
        text_lower = text.lower()
        
        # Extract for each configured entity type
        for entity_type in self.config.get_entity_types():
            patterns = self.config.get_extraction_patterns(entity_type)
            
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group(1).strip() if match.lastindex else match.group(0).strip()
                    if len(entity_name) > 3:
                        entities[entity_type].add(entity_name.title())
        
        # Extract from metadata (URL, title)
        self._extract_from_metadata(metadata, entities)
    
    def _extract_from_metadata(self, metadata: Dict, entities: Dict):
        """Extract entities from metadata (URL, title, etc.)"""
        url = metadata.get('url', '')
        title = metadata.get('title', '')
        
        # Faculty/people pages
        if any(pattern in url for pattern in ['/faculty/', '/people/', '/staff/']):
            path_parts = url.split('/')
            for part in path_parts:
                if part and len(part) > 2 and not part.startswith('http'):
                    clean_name = part.replace('-', ' ').replace('_', ' ').strip()
                    exclude_terms = ['faculty', 'people', 'staff', 'www', 'http', 'html', 'php']
                    if clean_name and not any(x in clean_name.lower() for x in exclude_terms):
                        if len(clean_name.split()) <= 3:
                            entities['Person'].add(clean_name.title())
        
        # Extract from title
        if title:
            # Lab names
            if 'lab' in title.lower():
                lab_match = re.search(r'(\w+(?:\s+\w+){0,3})\s+lab', title, re.IGNORECASE)
                if lab_match:
                    entities['Lab'].add(lab_match.group(1).strip().title())
            
            # Prof/Dr names
            name_pattern = r'(?:Prof(?:essor)?|Dr)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
            for match in re.finditer(name_pattern, title):
                entities['Person'].add(match.group(1).strip())
    
    def _extract_departments(self, url: str, metadata: Dict) -> Set[str]:
        """
        Auto-detect departments from URL patterns
        
        Args:
            url: Page URL
            metadata: Page metadata
        
        Returns:
            Set of department codes/names
        """
        if not self.config.should_auto_detect_departments():
            return set()
        
        departments = set()
        url_lower = url.lower()
        
        # Extract from URL path segments
        url_pattern = re.compile(r'/([a-z]{2,10})/')
        matches = url_pattern.findall(url_lower)
        
        for match in matches:
            # Filter out common non-department segments
            exclude = ['www', 'http', 'https', 'faculty', 'people', 'staff', 'research', 
                      'about', 'contact', 'news', 'events', 'index', 'home']
            if match not in exclude and len(match) >= 2:
                departments.add(match.upper())
        
        # Extract from subdomain
        subdomain_pattern = re.compile(r'https?://([a-z]+)\.')
        subdomain_match = subdomain_pattern.search(url_lower)
        if subdomain_match:
            subdomain = subdomain_match.group(1)
            if subdomain not in ['www', 'web', 'portal']:
                departments.add(subdomain.upper())
        
        # Extract from title
        title = metadata.get('title', '')
        dept_in_title = re.search(r'department\s+of\s+([a-z\s]+)', title.lower())
        if dept_in_title:
            dept_name = dept_in_title.group(1).strip()
            # Use acronym if too long
            if len(dept_name) > 15:
                acronym = ''.join([word[0] for word in dept_name.split()])
                departments.add(acronym.upper())
            else:
                departments.add(dept_name.upper())
        
        return departments
    
    def _clean_entities(self, entities: Dict[str, Set]) -> Dict[str, Set]:
        """Clean and filter extracted entities"""
        cleaned = {}
        min_length = self.spacy_config.get('min_entity_length', 3)
        
        for entity_type, entity_set in entities.items():
            cleaned[entity_type] = {
                e for e in entity_set 
                if e and len(e) > min_length
            }
        
        return cleaned
