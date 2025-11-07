# Database Configuration
DATABASE = {
    'host': 'localhost',
    'database': 'research_rag_db',
    'user': 'your_username',
    'password': 'your_password',
    'port': 5432
}

# Embedding model config
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384

# Chunking config
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# API config
API_HOST = '0.0.0.0'
API_PORT = 5000
DEBUG = True

# Ollama config
OLLAMA_URL = 'http://localhost:11434'
OLLAMA_MODEL = 'llama3.2'

# Paths
DATA_RAW_PATH = 'data/raw'
DATA_PROCESSED_PATH = 'data/processed'
DATA_EMBEDDINGS_PATH = 'data/embeddings'
LOGS_PATH = 'logs'
