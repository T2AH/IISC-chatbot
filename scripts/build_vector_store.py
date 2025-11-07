# c:\Users\harsh\Documents\chat application\scripts\build_vector_store.py
import os
import json
from sentence_transformers import SentenceTransformer
from src.data_processing.chunker import Chunker
from src.vector_store.faiss_store import FaissStore
from src.data_processing.text_clean import basic_clean
from src.utils.helpers import get_config

def main():
    config = get_config()
    raw_data_dir = config['crawler']['output_dir']
    
    # Load raw data
    all_text = ""
    for filename in os.listdir(raw_data_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(raw_data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_text += data['content'] + "\n\n"

    # Chunk the data
    chunker = Chunker(all_text, config['chunker']['chunk_size'], config['chunker']['overlap'])
    chunks = chunker.chunk()

    # Generate embeddings
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def _clean_text(text: str) -> str:
    return basic_clean(text)

    print("Vector store built successfully!")

if __name__ == "__main__":
    main()
