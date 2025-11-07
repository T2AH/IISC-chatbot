#!/usr/bin/env python3
"""
Test different Ollama models for RAG quality
"""
import requests
import json
import time

OLLAMA_URL = "http://localhost:11434"

def test_model(model_name, prompt):
    """Test a model with a sample prompt"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '')
            print(f"\n✓ Response time: {elapsed:.2f}s")
            print(f"\nAnswer:\n{answer}")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Connection failed: {e}")
        return False

def check_available_models():
    """Check which models are installed"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("Available models:")
            for model in models:
                name = model.get('name', 'unknown')
                size = model.get('size', 0) / (1024**3)  # Convert to GB
                print(f"  - {name} ({size:.2f} GB)")
            return [m.get('name') for m in models]
        else:
            print("Could not fetch models")
            return []
    except Exception as e:
        print(f"Error checking models: {e}")
        return []

def main():
    # Sample RAG prompt
    sample_prompt = """Based on the following information about CDS at IISc:

Context:
"Department of Computational and Data Sciences (CDS) is an interdisciplinary 
engineering department spanning computational science & engineering and scalable 
computer & data systems. Research activities are categorized into two streams: 
Computational Science and Computer & Data Systems."

Question: What are the two main research streams at CDS?

Answer concisely based only on the context provided."""

    print("🤖 Ollama Model Comparison for CDS RAG System")
    print("=" * 60)
    
    # Check available models
    available = check_available_models()
    
    if not available:
        print("\n⚠️  No models found. Please install models first:")
        print("   ollama pull qwen2.5:7b")
        print("   ollama pull llama3.2:3b")
        print("   ollama pull phi3:mini")
        return
    
    # Test each available model
    test_models = ['qwen2.5:7b', 'llama3.2:3b', 'llama3.1:8b', 'phi3:mini', 'llama3.2']
    
    for model in test_models:
        if model in available or any(model in m for m in available):
            test_model(model, sample_prompt)
            time.sleep(1)  # Brief pause between tests
        else:
            print(f"\n⚠️  {model} not installed (skipping)")
    
    print("\n" + "="*60)
    print("Recommendation: Compare response quality and speed above")
    print("For academic RAG, prioritize quality over speed")
    print("="*60)

if __name__ == "__main__":
    main()
