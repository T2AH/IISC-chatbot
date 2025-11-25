"""
Configuration loader module
Handles loading of configuration from YAML and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from loguru import logger


class Config:
    """Configuration manager for the IISc chatbot project"""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML and environment variables"""
        # Load environment variables
        load_dotenv()
        
        # Get project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"
        
        # Load YAML configuration
        if config_path.exists():
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found at {config_path}")
            self._config = {}
        
        # Override with environment variables
        self._load_env_overrides()
    
    def _load_env_overrides(self):
        """Override configuration with environment variables"""
        env_mappings = {
            'OPENAI_API_KEY': ['rag', 'openai_api_key'],
            'OPENAI_MODEL': ['rag', 'generation', 'model'],
            'NEO4J_URI': ['database', 'neo4j', 'uri'],
            'NEO4J_USERNAME': ['database', 'neo4j', 'username'],
            'NEO4J_PASSWORD': ['database', 'neo4j', 'password'],
            'CHROMADB_PERSIST_DIR': ['database', 'chromadb', 'persist_dir'],
            'LOG_LEVEL': ['logging', 'level'],
        }
        
        for env_key, config_path in env_mappings.items():
            value = os.getenv(env_key)
            if value:
                self._set_nested_value(config_path, value)
    
    def _set_nested_value(self, path: list, value: Any):
        """Set a nested configuration value"""
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get configuration value by key path
        
        Args:
            *keys: Path to configuration value (e.g., 'crawler', 'depth_rules')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self._config.copy()
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key"""
        return os.getenv('OPENAI_API_KEY', '')
    
    @property
    def neo4j_uri(self) -> str:
        """Get Neo4j URI"""
        return os.getenv('NEO4J_URI', 'neo4j://127.0.0.1:7687')
    
    @property
    def neo4j_username(self) -> str:
        """Get Neo4j username"""
        return os.getenv('NEO4J_USERNAME', 'neo4j')
    
    @property
    def neo4j_password(self) -> str:
        """Get Neo4j password"""
        return os.getenv('NEO4J_PASSWORD', '')
    
    @property
    def chromadb_persist_dir(self) -> str:
        """Get ChromaDB persistence directory"""
        return os.getenv('CHROMADB_PERSIST_DIR', './data/chromadb')
    
    @property
    def log_level(self) -> str:
        """Get log level"""
        return os.getenv('LOG_LEVEL', 'INFO')


# Global configuration instance
config = Config()
