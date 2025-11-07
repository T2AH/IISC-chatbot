# Contributing to IISc CDS RAG Chatbot

Thank you for your interest in contributing! This document provides guidelines and workflows for contributing to the project.

## 📁 Repository Structure

```
IISC-RAG-based-chatbot/
├── README.md                       # Main project documentation
├── requirements.txt                # Python dependencies
├── CONTRIBUTING.md                 # This file
│
├── config/                         # Configuration files
│   └── settings.py                 # Application settings
│
├── crawler/                        # Web crawling module
│   └── cds_hierarchical_crawler.py # Async crawler with hierarchy
│
├── src/                            # Main source code
│   ├── chunking/                   # Content chunking
│   │   └── clean_chunker.py
│   ├── cleaning/                   # HTML cleaning
│   │   └── html_cleaner.py
│   ├── embedding/                  # Vector embeddings
│   │   ├── chromadb_embedder.py
│   │   └── fixed_embedding_processor.py
│   ├── rag/                        # RAG system
│   │   ├── simple_rag_api.py       # Main server (use this)
│   │   ├── cds_chromadb_rag_api.py # Alternative implementation
│   │   └── cds_rag_api.py          # Legacy PostgreSQL version
│   └── utils/                      # Utility functions
│       ├── database.py
│       └── logger.py
│
├── scripts/                        # Utility scripts
│   ├── crawler.py                  # Alternative crawler
│   ├── analyze_chunk_hierarchy.py  # Analyze chunk structure
│   ├── convert_corpus_to_cleaned.py # Data conversion
│   └── parse_cds_jsonl.py          # JSONL parser
│
├── tests/                          # Test files
│   ├── test_chromadb_rag.py        # System validation
│   ├── test_faculty_data.py        # Data quality tests
│   └── test_ollama_models.py       # Model comparison
│
├── docs/                           # Documentation
│   ├── PROJECT_PROGRESS_REPORT.md  # Full project report
│   ├── PROJECT_SUMMARY.md          # Quick overview
│   ├── GITHUB_WORKFLOW.md          # Git workflow guide
│   ├── SETUP_MAC_M4.md             # Mac setup instructions
│   ├── HIERARCHICAL_GRAPH_SEARCH.md # Technical docs
│   ├── FUZZY_LAB_MATCHING_FIX.md
│   ├── README_HIERARCHICAL_RAG.md
│   ├── HIERARCHICAL_RAG_SUMMARY.md
│   ├── HIERARCHICAL_EMBEDDING_STRATEGIES.md
│   ├── CHROMADB_VS_POSTGRESQL_COMPARISON.md
│   ├── HIERARCHY_VISUAL_GUIDE.txt
│   └── api.md
│
├── data/                           # Data directory (gitignored)
│   └── processed/
│       ├── cds_cleaned.jsonl
│       └── cds_smart_chunks.jsonl
│
└── chroma_db/                      # Vector database (gitignored)
```

## 🚀 Development Workflow

### 1. Initial Setup

```bash
# Clone repository
git clone https://github.com/T2AH/IISC-RAG-based-chatbot.git
cd IISC-RAG-based-chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Mac/Linux
# or .venv\Scripts\activate on Windows

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install Ollama
# Mac: brew install ollama
# Or download from https://ollama.ai
ollama pull qwen2.5:7b
```

### 2. Data Pipeline (Optional - data already included)

If you need to regenerate the data:

```bash
# Step 1: Crawl website (5-10 minutes)
python crawler/cds_hierarchical_crawler.py
# Output: cds_hierarchical_corpus.json

# Step 2: Convert and chunk data
python scripts/convert_corpus_to_cleaned.py
python src/chunking/clean_chunker.py
# Output: data/processed/cds_smart_chunks.jsonl

# Step 3: Generate embeddings
python src/embedding/chromadb_embedder.py
# Output: chroma_db/ directory
```

### 3. Running the System

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start RAG server
source .venv/bin/activate
python src/rag/simple_rag_api.py

# Open browser: http://localhost:8001
```

### 4. Testing

```bash
# Run all tests
python tests/test_chromadb_rag.py
python tests/test_faculty_data.py
python tests/test_ollama_models.py

# Or run specific tests
python -m pytest tests/
```

## 🔧 Making Changes

### Branch Naming Convention

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/updates

### Git Workflow

```bash
# 1. Create a feature branch
git checkout -b feature/your-feature-name

# 2. Make your changes
# ... edit files ...

# 3. Test your changes
python tests/test_chromadb_rag.py

# 4. Stage and commit
git add .
git commit -m "feat: Add your descriptive message

- Bullet point of changes
- Another change
- etc."

# 5. Push to GitHub
git push origin feature/your-feature-name

# 6. Create Pull Request on GitHub
```

### Commit Message Format

Follow conventional commits:

```
type(scope): Short description

Longer description if needed

- Bullet points for details
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

**Examples:**
```
feat(rag): Add fuzzy lab name matching

- Implemented flexible LLM prompt for name variations
- Handles "Dream Lab" vs "DREAM:Lab" cases
- Prevents extraction failures

fix(chunking): Fix token counting for large documents

refactor(api): Simplify hierarchical search logic

docs(readme): Update installation instructions
```

## 📝 Code Style Guidelines

### Python

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions/classes
- Keep functions focused and small
- Use type hints where helpful

```python
def extract_faculty_from_lab(lab_chunk_text: str, lab_name: str) -> str | None:
    """
    Extract faculty name for a specific lab using LLM.
    
    Args:
        lab_chunk_text: Text containing lab descriptions
        lab_name: Name of the lab to search for
        
    Returns:
        Faculty name or None if not found
    """
    # Implementation...
```

### Logging

Use the logger instead of print statements:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("✅ Process completed successfully")
logger.warning("⚠️ No results found")
logger.error("❌ Error occurred")
logger.debug("🔍 Debug information")
```

## 🧪 Testing Guidelines

### Writing Tests

- Test files go in `tests/` directory
- Name test files `test_*.py`
- Test functions should start with `test_`
- Use descriptive test names

```python
def test_hierarchical_search_finds_students():
    """Test that hierarchical search can navigate Lab -> Faculty -> Students"""
    # Arrange
    query = "students of Dream Lab"
    
    # Act
    results = rag_service.hierarchical_graph_search(["Dream Lab"], "students")
    
    # Assert
    assert len(results) > 0
    assert any("student" in chunk['metadata'].get('chunk_type', '') 
               for chunk in results)
```

## 📚 Documentation Guidelines

### Code Documentation

- Add docstrings to all public functions/classes
- Include parameter descriptions and return types
- Provide usage examples for complex functions

### Project Documentation

- Update README.md for user-facing changes
- Add technical details to docs/ for complex features
- Include diagrams/flowcharts where helpful
- Update CHANGELOG.md with notable changes

## 🐛 Bug Reports

When reporting bugs, include:

1. **Description**: What happened vs what you expected
2. **Steps to Reproduce**: Exact steps to trigger the bug
3. **Environment**: OS, Python version, dependencies
4. **Logs**: Relevant error messages or logs
5. **Screenshots**: If applicable

## 💡 Feature Requests

When suggesting features, include:

1. **Problem**: What problem does this solve?
2. **Proposed Solution**: How would it work?
3. **Alternatives**: Other approaches considered
4. **Examples**: Similar implementations elsewhere

## 📊 Project Modules Explanation

### Core Components

1. **Crawler** (`crawler/`)
   - Crawls CDS website with async HTTP
   - Preserves parent-child hierarchy
   - Extracts clean content

2. **Chunker** (`src/chunking/`)
   - Splits documents into semantic chunks
   - Extracts metadata (faculty, research areas, etc.)
   - Maintains hierarchy relationships

3. **Embedder** (`src/embedding/`)
   - Generates vector embeddings using sentence-transformers
   - Stores in ChromaDB with metadata
   - Enables semantic search

4. **RAG System** (`src/rag/`)
   - **simple_rag_api.py**: Main production server
     - Hierarchical graph search
     - LLM query decomposition
     - Context-aware extraction
     - Fuzzy matching
   - Alternative implementations for reference

### Key Features to Understand

1. **Hierarchical Graph Search**
   - Navigates Lab → Faculty → Students relationships
   - Multi-hop reasoning
   - See: `docs/HIERARCHICAL_GRAPH_SEARCH.md`

2. **Context-Aware Extraction**
   - Extracts info specific to queried entity
   - Prevents cross-contamination
   - Passes entity context to LLM

3. **Fuzzy Matching**
   - Handles name variations
   - "Dream Lab" = "DREAM:Lab" = "DREAMLab"
   - See: `docs/FUZZY_LAB_MATCHING_FIX.md`

## 🔍 Debugging Tips

### Common Issues

**Ollama not responding:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

**ChromaDB errors:**
```bash
# Delete and regenerate database
rm -rf chroma_db/
python src/embedding/chromadb_embedder.py
```

**Import errors:**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Logging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📮 Getting Help

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check `/docs` directory first

## 📄 License

[Add license information here]

---

**Thank you for contributing to the IISc CDS RAG Chatbot!** 🎉
