# Quick Integration Guide for Enhanced RAG

## Quick Start

### Option 1: Drop-in Replacement (Easiest)

Simply update your imports in existing code:

```python
# In your main.py, api.py, or wherever you use the RAG system

# OLD imports
# from src.rag.retriever import HybridRetriever
# from src.rag.reranker import Reranker
# from src.rag.chatbot import RAGChatbot

# NEW imports (drop-in replacements)
from src.rag.retriever_enhanced import EnhancedHybridRetriever as HybridRetriever
from src.rag.reranker_enhanced import EnhancedReranker as Reranker
from src.rag.chatbot_enhanced import EnhancedRAGChatbot as RAGChatbot

# Rest of your code stays the same!
retriever = HybridRetriever()
reranker = Reranker()
chatbot = RAGChatbot(retriever, reranker)

response = chatbot.chat("Who are the CDS faculty?")
print(response['response'])
```

### Option 2: Use Enhanced Classes Directly

```python
from src.rag.retriever_enhanced import EnhancedHybridRetriever
from src.rag.reranker_enhanced import EnhancedReranker
from src.rag.chatbot_enhanced import EnhancedRAGChatbot

# Initialize with enhanced features
retriever = EnhancedHybridRetriever()
reranker = EnhancedReranker(use_llm=True)  # Enable LLM intelligence
chatbot = EnhancedRAGChatbot(retriever, reranker, debug=True)  # Enable debug

# Use the chatbot
response = chatbot.chat("Who are the CDS faculty?", debug=True)

# Access enhanced features
print(f"Strategies used: {response['query_analysis']}")
print(f"Departments detected: {response.get('debug', {}).get('matched_departments', [])}")
print(response['response'])
```

## Update Specific Files

### Update `main.py`

If you have a `main.py` that uses the chatbot, update the imports:

```python
# Around line 10-15, find:
# from src.rag.chatbot import RAGChatbot

# Replace with:
from src.rag.chatbot_enhanced import EnhancedRAGChatbot as RAGChatbot

# Or if you want debug mode:
from src.rag.chatbot_enhanced import EnhancedRAGChatbot

def chat_improved(query, debug=False):
    chatbot = EnhancedRAGChatbot(debug=debug)
    response = chatbot.chat(query, debug=debug)
    return response
```

### Update `api.py` or `start_web_server.py`

If you have an API endpoint:

```python
# In your API file, update imports:
from src.rag.chatbot_enhanced import EnhancedRAGChatbot

# Update your chatbot initialization
chatbot = EnhancedRAGChatbot(debug=False)  # Set debug=True for development

@app.post("/chat")
def chat_endpoint(query: str):
    response = chatbot.chat(query)
    return {
        "response": response['response'],
        "sources": response.get('sources', []),
        "num_sources": response.get('num_sources', 0)
    }
```

### Update `enhanced_chatbot.py` (if you have one)

If you already have an `enhanced_chatbot.py`, you can replace it or merge:

```python
# Simply import the new enhanced version
from src.rag.chatbot_enhanced import EnhancedRAGChatbot

# Use it as your enhanced chatbot
EnhancedChatbot = EnhancedRAGChatbot
```

## Test the Integration

### Step 1: Run the test suite

```bash
# Test everything
python test_enhanced_rag.py

# Test specific component
python test_enhanced_rag.py --test retrieval
python test_enhanced_rag.py --test reranking
python test_enhanced_rag.py --test chatbot

# Test with custom query
python test_enhanced_rag.py --query "Who are the CDS faculty?" --debug
```

### Step 2: Test your existing application

```bash
# If you have a main.py with chat command
python main.py chat -q "Who are the CDS faculty?" -d

# If you have a web server
python start_web_server.py
# Then test at http://localhost:5000 or your port
```

### Step 3: Verify improvements

Look for these improvements:

1. **More relevant results**: Faculty directory pages appear in top 3
2. **Complete lists**: "Who are the faculty?" returns ALL faculty, not just 3-5
3. **Better understanding**: Queries about departments are matched correctly
4. **Diversity**: Results come from different sources, not all from one page
5. **Debug info**: If debug=True, you see detailed scoring and reasoning

## Troubleshooting

### Issue: Import errors

```python
# Make sure your Python path includes the project root
import sys
sys.path.append('c:/Users/cdsmt/OneDrive/Documents/bot')

# Then import
from src.rag.chatbot_enhanced import EnhancedRAGChatbot
```

### Issue: Missing dependencies

```bash
# Install required packages
pip install sentence-transformers  # For cross-encoder
pip install openai  # For LLM intelligence
pip install numpy  # For MMR
```

### Issue: OpenAI API key

```python
# Make sure your config.yaml has the API key
# Or set environment variable
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
```

### Issue: ChromaDB or Neo4j not initialized

The enhanced modules will gracefully fallback if databases aren't available:
- No ChromaDB: Graph retrieval only
- No Neo4j: Vector retrieval only
- No embedder: Uses simpler scoring

But for best results, ensure both are running:

```bash
# Check ChromaDB
python -c "from src.database.chromadb_client import ChromaDBClient; c = ChromaDBClient(); print('ChromaDB OK')"

# Check Neo4j
python -c "from src.database.neo4j_client import Neo4jClient; n = Neo4jClient(); print('Neo4j OK')"
```

## Configuration

The enhanced modules use the same config.yaml settings:

```yaml
# config.yaml
rag:
  retrieval:
    top_k_vectors: 100  # More results per strategy
    top_k_graph: 20
    
  generation:
    model: gpt-4-turbo-preview  # Or gpt-4, gpt-3.5-turbo
    temperature: 0.7
    max_tokens: 500  # Auto-increased for list queries
```

## Comparison: Before vs After

### Before (Original System)

```python
from src.rag.chatbot import RAGChatbot

chatbot = RAGChatbot()
response = chatbot.chat("Who are the CDS faculty?")

# Typical result:
# - 3-5 faculty members mentioned
# - May not include faculty directory page
# - Focused on individual profiles
# - "...and others" or truncated list
```

### After (Enhanced System)

```python
from src.rag.chatbot_enhanced import EnhancedRAGChatbot

chatbot = EnhancedRAGChatbot(debug=True)
response = chatbot.chat("Who are the CDS faculty?")

# Enhanced result:
# - ALL faculty members listed (15-20+)
# - Faculty directory page in top results
# - Comprehensive numbered list
# - Complete with names, titles, research areas
# - Debug info shows why results were chosen
```

## Next Steps

1. **Test with your queries**: Run test_enhanced_rag.py with your common queries
2. **Enable debug mode**: See what's happening under the hood
3. **Adjust configuration**: Fine-tune for your specific needs
4. **Monitor performance**: Check logs for any warnings or errors
5. **Iterate**: Use debug info to understand and improve results

## Support

- Documentation: `docs/ENHANCED_RAG_DOCUMENTATION.md`
- Test suite: `test_enhanced_rag.py`
- Examples: See docstrings in each enhanced module
- Logs: Check loguru output for detailed operation info
