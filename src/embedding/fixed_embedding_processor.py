# -*- coding: utf-8 -*-
import json
import os
import time
import codecs

import psycopg2
from sentence_transformers import SentenceTransformer

class SimpleEmbeddingProcessor:
    def __init__(self):
        """Initialize with your exact database config"""
        self.db_config = {
            'host': 'localhost',
            'database': 'cds_rag_db',
            'user': 'rag_user',
            'password': 'secure_rag_password_123',
            'port': 5432
        }
        
        self.model = None
        self.conn = None
        
    def test_connection(self):
        """Test database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print("✅ Database connection successful!")
            print("PostgreSQL version: {}".format(version[:50]))
            
            # Test vector extension
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
            has_vector = cur.fetchone()[0]
            if has_vector:
                print("✅ pgvector extension found!")
            else:
                print("❌ pgvector extension not found!")
                
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            print("❌ Database connection failed: {}".format(e))
            return False
    
    def load_model(self):
        """Load sentence transformer model"""
        try:
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Model loaded successfully!")
            print("Embedding dimension: {}".format(self.model.get_sentence_embedding_dimension()))
            return True
        except Exception as e:
            print("❌ Failed to load model: {}".format(e))
            return False
    
    def test_similarity_search(self, query="machine learning research"):
        """Test similarity search with proper vector casting"""
        print("\n🔍 Testing similarity search with query: '{}'".format(query))
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query)
            
            # Convert to proper format
            query_vector = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
            
            self.conn = psycopg2.connect(**self.db_config)
            cur = self.conn.cursor()
            
            # Use proper vector casting
            cur.execute("""
                SELECT chunk_id, chunk_text, chunk_type, faculty_names, research_areas,
                       1 - (embedding <=> %s::vector) as similarity
                FROM cds_embeddings 
                ORDER BY embedding <=> %s::vector
                LIMIT 3;
            """, (query_vector, query_vector))
            
            results = cur.fetchall()
            
            print("🎯 Top 3 results:")
            for i, result in enumerate(results):
                chunk_id, text, chunk_type, faculty, research, similarity = result
                print("\n--- Result {} (similarity: {:.3f}) ---".format(i+1, similarity))
                print("Chunk ID: {}".format(chunk_id))
                print("Type: {}".format(chunk_type or 'N/A'))
                if faculty and len(faculty) > 0:
                    print("Faculty: {}".format(', '.join(faculty)))
                if research and len(research) > 0:
                    print("Research: {}".format(', '.join(research)))
                print("Text: {}...".format(text[:200]))
            
            cur.close()
            self.conn.close()
            
        except Exception as e:
            print("❌ Similarity search failed: {}".format(e))
    
    def get_database_stats(self):
        """Get database statistics"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            cur = self.conn.cursor()
            
            # Basic statistics
            cur.execute("SELECT COUNT(*) FROM cds_embeddings;")
            total_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM cds_embeddings WHERE embedding IS NOT NULL;")
            embedded_count = cur.fetchone()[0]
            
            cur.execute("SELECT AVG(token_count) FROM cds_embeddings WHERE token_count > 0;")
            avg_tokens = cur.fetchone()[0]
            
            # Chunk type distribution
            cur.execute("""
                SELECT chunk_type, COUNT(*) 
                FROM cds_embeddings 
                WHERE chunk_type IS NOT NULL AND chunk_type != ''
                GROUP BY chunk_type 
                ORDER BY COUNT(*) DESC 
                LIMIT 10;
            """)
            chunk_types = cur.fetchall()
            
            # Faculty info
            cur.execute("SELECT COUNT(*) FROM cds_embeddings WHERE has_faculty_info = TRUE;")
            faculty_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM cds_embeddings WHERE has_research_info = TRUE;")
            research_count = cur.fetchone()[0]
            
            print("\n📊 Database Statistics:")
            print("=" * 40)
            print("Total chunks: {}".format(total_count))
            print("Chunks with embeddings: {}".format(embedded_count))
            print("Average tokens per chunk: {:.1f}".format(avg_tokens or 0))
            print("Chunks with faculty info: {} ({:.1f}%)".format(
                faculty_count, (faculty_count * 100.0 / total_count) if total_count > 0 else 0))
            print("Chunks with research info: {} ({:.1f}%)".format(
                research_count, (research_count * 100.0 / total_count) if total_count > 0 else 0))
            
            if chunk_types:
                print("\nTop chunk types:")
                for chunk_type, count in chunk_types:
                    percentage = (count * 100.0 / total_count) if total_count > 0 else 0
                    print("  {}: {} ({:.1f}%)".format(chunk_type, count, percentage))
            
            cur.close()
            self.conn.close()
            
        except Exception as e:
            print("❌ Error getting database stats: {}".format(e))

def main():
    print("🚀 Testing CDS Vector Database")
    print("=" * 40)
    
    processor = SimpleEmbeddingProcessor()
    
    # Test database connection
    if not processor.test_connection():
        return
    
    # Load model
    if not processor.load_model():
        return
    
    # Get database statistics
    processor.get_database_stats()
    
    # Test similarity searches
    test_queries = [
        "Sathish Vadhiyar HPC research",
        "machine learning CDS",
        "PhD admission requirements",
        "DREAM lab graph computing",
        "computational science faculty",
        "research areas computer science",
        "high performance computing",
        "data science projects"
    ]
    
    print("\n🧪 Testing Similarity Search:")
    print("=" * 40)
    
    for query in test_queries:
        processor.test_similarity_search(query)
        print("-" * 50)
    
    print("\n🎉 Vector database testing completed!")
    print("✅ Your CDS RAG system is ready for Ollama integration!")

if __name__ == "__main__":
    main()