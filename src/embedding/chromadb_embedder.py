#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromaDB Embedder for Hierarchical RAG System
Embeds chunks and stores them in ChromaDB with hierarchy metadata
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm

class ChromaDBEmbedder:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize ChromaDB embedder"""
        self.persist_directory = persist_directory
        
        print("🚀 Initializing ChromaDB Embedder...")
        
        # Initialize ChromaDB client with persistence (ChromaDB 1.1+)
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )
        
        # Load sentence transformer model
        print("📥 Loading embedding model: all-MiniLM-L6-v2...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded! Embedding dimension: {self.embedding_dim}")
        
        # Create or get collection
        self.collection_name = "cds_hierarchical_chunks"
        self.collection = None
        
    def create_collection(self, reset=False):
        """Create or reset ChromaDB collection"""
        
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"🗑️  Deleted existing collection: {self.collection_name}")
            except:
                pass
        
        # Create collection with metadata
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "description": "CDS IISc hierarchical chunks with embeddings",
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dim": self.embedding_dim
            }
        )
        
        print(f"✅ Created collection: {self.collection_name}")
        
    def load_chunks(self, chunks_file):
        """Load chunks from JSONL file"""
        print(f"\n📂 Loading chunks from: {chunks_file}")
        
        chunks = []
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks.append(chunk)
        
        print(f"✅ Loaded {len(chunks)} chunks")
        return chunks
    
    def prepare_chunk_data(self, chunk):
        """Prepare chunk data for ChromaDB"""
        metadata = chunk['metadata']
        
        # ChromaDB metadata (only supports: str, int, float, bool)
        # Convert lists to JSON strings
        chroma_metadata = {
            # Basic info
            'chunk_id': chunk['chunk_id'],
            'doc_id': chunk['doc_id'],
            'chunk_index': chunk['chunk_index'],
            'total_chunks': chunk['total_chunks'],
            'token_count': chunk['token_count'],
            
            # Page metadata
            'url': metadata.get('url', ''),
            'title': metadata.get('title', ''),
            'domain': metadata.get('domain', ''),
            
            # Hierarchy metadata (PRESERVED!)
            'hierarchy_level': metadata.get('hierarchy_level', -1),
            'parent_id': metadata.get('parent_id', '') or '',
            'node_type': metadata.get('node_type', ''),
            'children_ids': json.dumps(metadata.get('children_ids', [])),  # JSON string
            
            # Chunk classification
            'chunk_type': metadata.get('chunk_type', ''),
            'has_faculty_info': metadata.get('has_faculty_info', False),
            'has_research_info': metadata.get('has_research_info', False),
            
            # Entities (as JSON strings for ChromaDB)
            'faculty_names': json.dumps(metadata.get('chunk_faculty_names', [])),
            'departments': json.dumps(metadata.get('chunk_departments', [])),
            'research_areas': json.dumps(metadata.get('chunk_research_areas', [])),
            'positions': json.dumps(metadata.get('chunk_positions', []))
        }
        
        return chroma_metadata
    
    def embed_and_store_chunks(self, chunks, batch_size=100):
        """Generate embeddings and store in ChromaDB"""
        print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")
        print(f"   Batch size: {batch_size}")
        
        # Process in batches
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
            batch = chunks[i:i + batch_size]
            
            # Prepare batch data
            ids = []
            texts = []
            metadatas = []
            
            for chunk in batch:
                ids.append(chunk['chunk_id'])
                texts.append(chunk['chunk_text'])
                metadatas.append(self.prepare_chunk_data(chunk))
            
            # Generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # Store in ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
        
        print(f"✅ Successfully embedded and stored {len(chunks)} chunks!")
    
    def verify_collection(self):
        """Verify the collection and show stats"""
        print("\n📊 Collection Statistics:")
        print(f"   Name: {self.collection.name}")
        print(f"   Total chunks: {self.collection.count()}")
        
        # Sample query to verify
        print("\n🔍 Testing sample query...")
        results = self.collection.query(
            query_texts=["machine learning research"],
            n_results=3
        )
        
        if results['documents']:
            print(f"\n✅ Sample query successful! Top 3 results:")
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0],
                results['distances'][0]
            )):
                print(f"\n   Result {i+1}:")
                print(f"   - Chunk ID: {meta['chunk_id']}")
                print(f"   - Title: {meta.get('title', 'N/A')[:60]}...")
                print(f"   - Hierarchy Level: {meta.get('hierarchy_level')}")
                print(f"   - Node Type: {meta.get('node_type')}")
                print(f"   - Chunk Type: {meta.get('chunk_type')}")
                print(f"   - Distance: {dist:.4f}")
                print(f"   - Text: {doc[:100]}...")
    
    def save_and_persist(self):
        """Persist the database to disk"""
        print(f"\n💾 Persisting database to: {self.persist_directory}")
        # ChromaDB auto-persists, but we can explicitly trigger it
        print("✅ Database persisted!")

def main():
    print("=" * 70)
    print("  ChromaDB Hierarchical Embedding Pipeline")
    print("=" * 70)
    
    # Configuration
    CHUNKS_FILE = "data/processed/cds_smart_chunks.jsonl"
    CHROMA_DIR = "./chroma_db"
    BATCH_SIZE = 100
    RESET_DB = True  # Set to False to append to existing DB
    
    # Check if chunks file exists
    if not os.path.exists(CHUNKS_FILE):
        print(f"❌ Error: Chunks file not found: {CHUNKS_FILE}")
        print("   Please run the chunker first:")
        print("   python src/chunking/clean_chunker.py")
        return
    
    # Initialize embedder
    embedder = ChromaDBEmbedder(persist_directory=CHROMA_DIR)
    
    # Create collection
    embedder.create_collection(reset=RESET_DB)
    
    # Load chunks
    chunks = embedder.load_chunks(CHUNKS_FILE)
    
    # Embed and store
    embedder.embed_and_store_chunks(chunks, batch_size=BATCH_SIZE)
    
    # Verify
    embedder.verify_collection()
    
    # Persist
    embedder.save_and_persist()
    
    print("\n" + "=" * 70)
    print("✅ Embedding pipeline completed successfully!")
    print("=" * 70)
    print(f"\n📁 ChromaDB location: {CHROMA_DIR}")
    print(f"🔢 Total chunks embedded: {len(chunks)}")
    print(f"📊 Embedding dimension: {embedder.embedding_dim}")
    print("\n🎯 Next step: Run the RAG API server")
    print("   python src/rag/cds_rag_api.py")

if __name__ == "__main__":
    main()
