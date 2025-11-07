# c:\Users\harsh\Documents\chat application\config.py
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

CRAWLER_CONFIG = {
    "start_urls": ["http://cds.iisc.ac.in"],
    "allowed_domains": ["cds.iisc.ac.in", "github.io"],
    "output_file": "data/processed/crawled_data.json",
    "max_depth": 5
}

CHUNKER_CONFIG = {
    "chunk_size": 1000,
    "overlap": 200
}

VECTOR_STORE_CONFIG = {
    "dimension": 384, # Example dimension for sentence-transformers/all-MiniLM-L6-v2
    "index_path": "data/processed/faiss.index",
    "chunks_path": "data/processed/chunks.txt",
    "metadata_path": "data/processed/chunks_meta.jsonl",
    "embeddings_path": "data/processed/embeddings.npy"
}
