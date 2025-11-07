#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script for ChromaDB RAG API
Tests the system before starting the web server
"""

import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json

def test_chromadb_connection():
    """Test ChromaDB connection and data"""
    print("=" * 70)
    print("Testing ChromaDB Connection...")
    print("=" * 70)
    
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        
        collection = client.get_collection(name="cds_hierarchical_chunks")
        count = collection.count()
        
        print(f"✅ ChromaDB connected successfully!")
        print(f"📊 Total chunks in database: {count}")
        
        return True
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        return False

def test_embedding_model():
    """Test embedding model loading"""
    print("\n" + "=" * 70)
    print("Testing Embedding Model...")
    print("=" * 70)
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_text = "machine learning research"
        embedding = model.encode(test_text)
        
        print(f"✅ Embedding model loaded successfully!")
        print(f"📊 Embedding dimension: {len(embedding)}")
        
        return True
    except Exception as e:
        print(f"❌ Embedding model failed: {e}")
        return False

def test_semantic_search():
    """Test semantic search functionality"""
    print("\n" + "=" * 70)
    print("Testing Semantic Search...")
    print("=" * 70)
    
    try:
        # Load ChromaDB
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="cds_hierarchical_chunks")
        
        # Load model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test queries
        test_queries = [
            "Which professors work on machine learning?",
            "What are the admission requirements?",
            "Tell me about research labs"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Generate embedding
            query_embedding = model.encode(query)
            
            # Search
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=3
            )
            
            if results['documents'] and results['documents'][0]:
                print(f"   ✅ Found {len(results['documents'][0])} results")
                
                # Show top result
                top_result = results['documents'][0][0]
                top_meta = results['metadatas'][0][0]
                top_dist = results['distances'][0][0]
                
                print(f"   📄 Top Result:")
                print(f"      Title: {top_meta.get('title', 'N/A')[:50]}...")
                print(f"      Level: {top_meta.get('hierarchy_level', 'N/A')}")
                print(f"      Type: {top_meta.get('chunk_type', 'N/A')}")
                print(f"      Distance: {top_dist:.4f}")
                print(f"      Text: {top_result[:100]}...")
            else:
                print(f"   ⚠️  No results found")
        
        return True
    except Exception as e:
        print(f"❌ Semantic search failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama API connection"""
    print("\n" + "=" * 70)
    print("Testing Ollama Connection...")
    print("=" * 70)
    
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama is running!")
            print(f"📊 Available models: {len(models)}")
            
            for model in models:
                model_name = model.get('name', 'Unknown')
                size_gb = model.get('size', 0) / 1e9
                print(f"   - {model_name} ({size_gb:.2f} GB)")
            
            # Check for recommended models
            model_names = [m.get('name', '') for m in models]
            if any('qwen2.5:7b' in name for name in model_names):
                print(f"\n✅ qwen2.5:7b found! Ready for production.")
            elif any('llama3.2:3b' in name for name in model_names):
                print(f"\n✅ llama3.2:3b found! Good for testing.")
            elif models:
                print(f"\n⚠️  No recommended model found. Consider downloading:")
                print(f"   ollama pull qwen2.5:7b")
            else:
                print(f"\n⚠️  No models installed. Download one:")
                print(f"   ollama pull qwen2.5:7b")
                return False
            
            return True
        else:
            print(f"❌ Ollama returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Ollama at http://localhost:11434")
        print(f"   Make sure Ollama is running:")
        print(f"   ollama serve")
        return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def main():
    print("\n" + "=" * 70)
    print("  CDS ChromaDB RAG System - Pre-flight Check")
    print("=" * 70 + "\n")
    
    results = {
        "ChromaDB": test_chromadb_connection(),
        "Embedding Model": test_embedding_model(),
        "Semantic Search": test_semantic_search(),
        "Ollama": test_ollama_connection()
    }
    
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 All tests passed! Your RAG system is ready!")
        print("\n🚀 Start the server with:")
        print("   python src/rag/cds_chromadb_rag_api.py")
        print("\n   Then open: http://localhost:8000")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        
        if not results["Ollama"]:
            print("\n💡 Waiting for Ollama model download?")
            print("   Check download progress in the other terminal.")
            print("   Once complete, run this test again:")
            print("   python test_chromadb_rag.py")
    
    print()

if __name__ == "__main__":
    main()
