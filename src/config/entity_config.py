"""
Generic Entity Extraction Configuration Loader
"""
import yaml
import os
from typing import Dict, List, Any
from pathlib import Path
from loguru import logger

class EntityExtractionConfig:
    """Load and manage entity extraction configuration"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to YAML config file (default: config/entity_extraction_config.yaml)
        """
        if config_path is None:
            config_path = os.path.join(
                Path(__file__).parent.parent.parent,
                "config",
                "entity_extraction_config.yaml"
            )
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'entity_types': ['Person', 'Lab', 'Organization', 'ResearchTopic'],
            'department_detection': {'enabled': True, 'auto_detect': True},
            'spacy': {'model': 'en_core_web_sm', 'max_text_length': 5000},
            'performance': {'batch_size': 100}
        }
    
    def get_entity_types(self) -> List[str]:
        """Get configured entity types"""
        return self.config.get('entity_types', [])
    
    def get_extraction_patterns(self, entity_type: str) -> List[str]:
        """Get regex patterns for entity type"""
        patterns = self.config.get('extraction_patterns', {})
        return patterns.get(entity_type.lower(), [])
    
    def get_query_intent_patterns(self, intent_type: str) -> List[str]:
        """Get query intent detection patterns"""
        intents = self.config.get('query_intents', {})
        return intents.get(f'{intent_type}_indicators', [])
    
    def get_spacy_config(self) -> Dict[str, Any]:
        """Get SpaCy configuration"""
        return self.config.get('spacy', {})
    
    def get_neo4j_schema(self) -> Dict[str, Any]:
        """Get Neo4j schema configuration"""
        return self.config.get('neo4j_schema', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance settings"""
        return self.config.get('performance', {})
    
    def is_department_detection_enabled(self) -> bool:
        """Check if department detection is enabled"""
        dept_config = self.config.get('department_detection', {})
        return dept_config.get('enabled', True)
    
    def should_auto_detect_departments(self) -> bool:
        """Check if auto-detection of departments is enabled"""
        dept_config = self.config.get('department_detection', {})
        return dept_config.get('auto_detect', True)
