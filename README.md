# IISc Research Chatbot

A production-ready Python application for building an AI-powered research and academic chatbot for the Indian Institute of Science (IISc). This system combines web crawling, NLP processing, knowledge graph storage, and retrieval-augmented generation (RAG) to help students and researchers find information about faculty, labs, courses, projects, and research topics.

## Project Context

This project was developed as part of the **DS252 Introduction to Cloud Computing (Aug, 2025)** course at the Indian Institute of Science (IISc), Department of Computational and Data Sciences (CDS).

## Team 

This project was carried out by the following team members:
- Amitesh Pandey – amiteshp@iisc.ac.in
- Harsh Saxena – harshsaxena@iisc.ac.in
- Abhinav Rawat – abhinavrawat@iisc.ac.in


## 🌐 Access the Live Chatbot

**Public Access:** [http://13.200.45.148:8080](http://13.200.45.148:8080)

The chatbot is deployed on AWS EC2 and running 24/7. Simply visit the URL above to start chatting!

### Alternative Access Methods

**For Development/Testing:**

If you want to run the chatbot locally or connect to the production deployment:

#### Option 1: Docker Compose (Recommended for Production)

```bash
# Clone the repository
git clone <repository-url>
cd iisc-chatbot

# Create .env file with your credentials
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
EOF

# Start all services (chatbot, Neo4j, ChromaDB)
docker compose -f docker-compose.prod.yml up -d

# Access at http://localhost:8080
```

**Check status:**
```bash
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f chatbot
```

**Stop services:**
```bash
docker compose -f docker-compose.prod.yml down
```

#### Option 2: Local Python Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start Neo4j Desktop (or Docker Neo4j container)
# Start ChromaDB (included in requirements)

# Configure .env
OPENAI_API_KEY=your_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Run the web server
python start_web_server.py

# Access at http://localhost:8080
```

#### Option 3: Connect to Production EC2 Instance

**SSH Access (for administrators):**

```bash
# SSH into EC2 instance
ssh -i /path/to/iisc-chatbot-key.pem ubuntu@13.200.45.148

# Check running containers
docker compose -f ~/iisc-chatbot/docker-compose.prod.yml ps

# View logs
docker compose -f ~/iisc-chatbot/docker-compose.prod.yml logs -f chatbot

# Restart services
docker compose -f ~/iisc-chatbot/docker-compose.prod.yml restart chatbot
```

**Data Persistence:**

All containers use Docker volumes for data persistence:
- Neo4j data: `neo4j_data` volume
- ChromaDB data: `./data/chromadb` (198,197 documents)
- Chat history: SQLite database in container

## Features

- **Multi-Domain Web Crawling**: Structure-aware crawling for IISc sites and intelligent NLP-based crawling for external faculty/lab websites
- **Advanced NLP Processing**: Entity extraction (spaCy + BERT), keyword extraction (KeyBERT), and semantic embeddings (Sentence Transformers)
- **Dual Database Architecture**: 
  - Neo4j knowledge graph for entities and relationships
  - ChromaDB vector database for semantic search
- **RAG-Powered Chatbot**: LangChain + OpenAI integration for intelligent, context-aware responses
- **Scalable Pipeline**: Modular architecture supporting the full pipeline from crawling to querying

## Recent Updates (2025-11-20)

- **Production Deployment:** Chatbot deployed on AWS EC2 at `http://13.200.45.148:8080` with full Docker orchestration
- **Knowledge Graph Migration:** Successfully migrated 198,196 Page nodes and 31,304 Entity nodes from local Neo4j to production
- **CSV-based Import:** Implemented reliable CSV import pipeline for Neo4j data transfer (handles special characters correctly)
- **UI & Thread Management:** Modern sidebar UI with chat thread management. Frontend files in `static/` (`static/index.html`, `static/css/style.css`, `static/js/chat.js`). Chat threads persisted in SQLite (`data/chat_threads.db`)
- **Retrieval Improvements:** Hybrid retriever tuned for higher recall (larger `top_k_vectors`) with non-hardcoded department-aware query expansion
- **Neo4j & ChromaDB diagnostics:** Inspection scripts (`check_neo4j_state.py`, `check_chromadb_cds.py`) for validating entity and vector counts
- **CORS Configuration:** Fixed frontend API connectivity issues with proper CORS middleware and dynamic API_URL configuration
- **Docker Persistence:** All services configured with `restart: always` for automatic recovery on reboot
- **Port 8080 troubleshooting**: If `start_web_server.py` fails due to port contention on `8080`, check and free the port on Windows PowerShell:

```powershell
netstat -ano | findstr ":8080"
taskkill /PID <pid> /F
```

- **Quick test (local)**: Start the server and send a simple chat request to validate the API:

```powershell
python start_web_server.py
curl -X POST "http://localhost:8080/chat" -H "Content-Type: application/json" -d "{\"query\":\"Who are the faculty in the CDS department?\"}"
```

If you get a `422 Unprocessable Entity` on POST requests from the frontend, confirm the request body shape matches the API schema (some endpoints accept different JSON shapes depending on recent edits).


## Table of Contents

- [Access the Live Chatbot](#-access-the-live-chatbot)
- [Architecture](#architecture)
- [Deployment](#deployment)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Components](#components)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Chatbot (LangChain)                       │
│  ┌──────────────────┐         ┌─────────────────────────────┐  │
│  │ Hybrid Retriever │────────▶│  OpenAI GPT-4 Generation    │  │
│  └──────────────────┘         └─────────────────────────────┘  │
└────────┬──────────────────────────────────┬────────────────────┘
         │                                   │
         ▼                                   ▼
┌──────────────────────┐          ┌──────────────────────┐
│   ChromaDB (Vectors) │          │ Neo4j (Knowledge     │
│   - Embeddings       │          │ Graph)               │
│   - Semantic Search  │          │ - Entities           │
└──────────┬───────────┘          │ - Relationships      │
           │                       └──────────┬───────────┘
           │                                  │
           └──────────────┬───────────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   NLP Processing       │
              │   - Entity Extraction  │
              │   - Keyword Extraction │
              │   - Text Chunking      │
              │   - Embeddings         │
              └────────┬───────────────┘
                       │
                       ▼
              ┌────────────────────────┐
              │   Web Crawler          │
              │   - Scrapy Spiders     │
              │   - Dynamic Depth      │
              │   - Multi-Domain       │
              └────────────────────────┘
```

## Deployment

### Production Deployment (AWS EC2)

The chatbot is currently deployed on AWS EC2 with the following setup:

**Infrastructure:**
- **Instance Type:** AWS EC2 (Ubuntu 22.04)
- **IP Address:** `13.200.45.148`
- **Port:** `8080`
- **Deployment Method:** Docker Compose
- **Auto-restart:** Enabled (`restart: always`)

**Services Running:**
1. **Chatbot API** (`iisc-chatbot-prod`) - FastAPI + Uvicorn on port 8080
2. **Neo4j Database** (`iisc-neo4j-prod`) - Knowledge Graph on ports 7474 (HTTP), 7687 (Bolt)
3. **ChromaDB** - Vector database (embedded in chatbot container)

**Data Status:**
- ✅ 198,197 documents indexed in ChromaDB
- ✅ 198,196 Page nodes in Neo4j
- ✅ 31,304 Entity nodes (Labs, People, Topics, Organizations)
- ✅ 138,632 knowledge graph relationships

**Deployment Commands:**

```bash
# Deploy from local machine to EC2
scp -i /path/to/key.pem -r . ubuntu@13.200.45.148:~/iisc-chatbot/

# SSH to EC2
ssh -i /path/to/key.pem ubuntu@13.200.45.148

# Start services
cd ~/iisc-chatbot
docker compose -f docker-compose.prod.yml up -d

# Monitor logs
docker compose -f docker-compose.prod.yml logs -f chatbot

# Check health
curl http://localhost:8080/health
```

**Security Considerations:**
- EC2 Security Groups configured for ports 22 (SSH), 8080 (HTTP), 7474, 7687 (Neo4j)
- SSH key-based authentication only
- Neo4j password-protected
- OpenAI API key stored in environment variables

**Monitoring:**

```bash
# Container status
docker compose -f docker-compose.prod.yml ps

# Resource usage
docker stats

# Application logs
docker compose -f docker-compose.prod.yml logs --tail=100 chatbot

# Neo4j logs
docker compose -f docker-compose.prod.yml logs --tail=100 neo4j
```

**Backup & Maintenance:**

```bash
# Backup Neo4j data
docker exec iisc-neo4j-prod neo4j-admin database dump neo4j \
  --to-path=/backups/backup-$(date +%Y%m%d).dump

# Backup ChromaDB data
tar -czf chromadb-backup-$(date +%Y%m%d).tar.gz ./data/chromadb/

# Update deployment
git pull origin main
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
```

## Installation

### Prerequisites

- Python 3.9 or higher
- Neo4j 5.x (for knowledge graph)
- 8GB+ RAM recommended
- OpenAI API key

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd bot
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download spaCy Model

```bash
python -m spacy download en_core_web_lg
```

### Step 5: Setup Neo4j

1. Download and install Neo4j Desktop from https://neo4j.com/download/
2. Create a new database with password
3. Start the database

### Step 6: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
```env
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

## Configuration

The main configuration file is `config.yaml`. Key sections:

### Crawler Configuration

```yaml
crawler:
  depth_rules:
    department_sites:
      max_depth: 4
    external_sites:
      max_depth: 3
```

### NLP Configuration

```yaml
nlp:
  text_processing:
    chunk_size: 250  # words per chunk
    chunk_overlap: 50
  models:
    spacy_model: en_core_web_lg
    embedding_model: sentence-transformers/all-MiniLM-L6-v2
```

### RAG Configuration

```yaml
rag:
  retrieval:
    top_k_vectors: 5
    similarity_threshold: 0.7
  generation:
    temperature: 0.7
    max_tokens: 500
```

## Usage

### Command-Line Interface

The main entry point is `main.py`. Available commands:

#### 1. Web Crawling

**Crawl IISc CDS Department:**
```bash
python main.py crawl --spider iisc
```

**Crawl a specific URL:**
```bash
python main.py crawl --spider iisc --url https://cds.iisc.ac.in
```

**Crawl external faculty website:**
```bash
python main.py crawl --spider generic --url https://faculty-website.edu
```

#### 2. NLP Processing

Process crawled data:
```bash
python main.py process --input data/crawled_pages/pages_iisc_spider_20231105.jsonl
```

With custom output:
```bash
python main.py process --input data/crawled_pages/pages.jsonl --output data/processed.jsonl
```

#### 3. Database Import

Import processed data to Neo4j and ChromaDB:
```bash
python main.py import --input data/crawled_pages/processed_pages.jsonl
```

#### 4. RAG Chatbot

**Single Query:**
```bash
python main.py chat --query "Who are the faculty working on machine learning?"
```

**Interactive Mode:**
```bash
python main.py chat --interactive
```

Example interaction:
```
You: Tell me about the CDS department
Bot: The Computational and Data Sciences (CDS) department at IISc...

You: Who are the faculty members?
Bot: The CDS department has several faculty members including...

You: exit
```

#### 5. Full Pipeline

Run the complete workflow:
```bash
python main.py pipeline
```

Skip specific steps:
```bash
python main.py pipeline --skip-crawl  # Use existing crawled data
python main.py pipeline --skip-nlp    # Use existing processed data
```

## Project Structure

```
project2/
├── src/
│   ├── crawler/              # Web crawling components
│   │   ├── spiders/
│   │   │   ├── iisc_spider.py       # IISc structure-aware spider
│   │   │   └── generic_spider.py    # Generic NLP-based spider
│   │   ├── middlewares.py           # Scrapy middlewares
│   │   ├── pipelines.py             # Scrapy pipelines
│   │   ├── items.py                 # Scrapy items
│   │   └── settings.py              # Scrapy settings
│   │
│   ├── nlp/                  # NLP processing
│   │   ├── text_processing.py       # Text cleaning & chunking
│   │   ├── entity_extraction.py     # Entity recognition (spaCy)
│   │   ├── keyword_extraction.py    # Keyword extraction (KeyBERT)
│   │   ├── embedding_generation.py  # Embeddings (Sentence Transformers)
│   │   └── pipeline.py              # NLP pipeline orchestration
│   │
│   ├── database/             # Database integration
│   │   ├── neo4j_client.py          # Neo4j knowledge graph
│   │   ├── chromadb_client.py       # ChromaDB vector store
│   │   └── manager.py               # Database orchestration
│   │
│   ├── rag/                  # RAG chatbot
│   │   ├── retriever.py             # Hybrid retrieval
│   │   └── chatbot.py               # LangChain + OpenAI chatbot
│   │
│   ├── utils/                # Utilities
│   │   └── logger.py                # Logging configuration
│   │
│   └── config.py             # Configuration loader
│
├── examples/
│   └── usage_examples.py     # Usage examples
│
├── data/                     # Data directory (created at runtime)
│   ├── crawled_pages/        # Crawled data
│   └── chromadb/             # ChromaDB persistence
│
├── logs/                     # Log files (created at runtime)
│
├── main.py                   # Main CLI entry point
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## Components

### 1. Crawler Layer

**Features:**
- Structure-aware crawling for IISc sites (3-4 levels deep)
- Generic NLP-based crawling for external sites (2-3 levels deep)
- Dynamic depth adjustment based on content quality
- Respects robots.txt with internal override option
- Retry logic with exponential backoff
- Domain whitelist/blacklist filtering

**Key Files:**
- `src/crawler/spiders/iisc_spider.py`: IISc-specific spider
- `src/crawler/spiders/generic_spider.py`: Generic spider for external sites

### 2. NLP Processing Layer

**Features:**
- Text cleaning and chunking (~250 words per chunk with 50-word overlap)
- Entity extraction: People, Organizations, Labs, Research Topics, Course Codes
- Keyword extraction with diversity control
- Semantic embeddings for each chunk
- Research domain-specific patterns

**Key Files:**
- `src/nlp/entity_extraction.py`: spaCy + BERT entity recognition
- `src/nlp/keyword_extraction.py`: KeyBERT keyword extraction
- `src/nlp/embedding_generation.py`: Sentence Transformers embeddings

### 3. Database Layer

**Neo4j Knowledge Graph:**
- Nodes: Faculty, Lab, Project, ResearchTopic, Course, Department
- Relationships: WORKS_IN, LEADS, CONDUCTS, COVERS, TEACHES, COLLABORATES_WITH
- Indexed and constrained for performance

**ChromaDB Vector Store:**
- Stores text chunks with embeddings
- Metadata: page_id, url, domain, page_type, title
- Cosine similarity search

**Key Files:**
- `src/database/neo4j_client.py`: Neo4j operations
- `src/database/chromadb_client.py`: ChromaDB operations
- `src/database/manager.py`: Unified database interface

### 4. RAG Chatbot Layer

**Features:**
- Hybrid retrieval (vector + knowledge graph)
- Context-aware response generation
- Source attribution
- Multi-turn conversation support
- Specialized queries (faculty, research topics, courses)

**Key Files:**
- `src/rag/retriever.py`: Hybrid retrieval system
- `src/rag/chatbot.py`: LangChain + OpenAI chatbot

## Examples

### Example 1: Crawl and Process CDS Department

```bash
# Step 1: Crawl
python main.py crawl --spider iisc --url https://cds.iisc.ac.in

# Step 2: Process (use the generated file from step 1)
python main.py process --input data/crawled_pages/pages_iisc_spider_*.jsonl

# Step 3: Import
python main.py import --input data/crawled_pages/processed_pages_*.jsonl

# Step 4: Query
python main.py chat --query "What research is being done in machine learning at CDS?"
```

### Example 2: Interactive Chatbot Session

```bash
python main.py chat --interactive
```

```
You: Who are the faculty in the CDS department?
Bot: The Computational and Data Sciences (CDS) department has several distinguished faculty members...

You: Tell me more about their machine learning research
Bot: Machine learning research at CDS focuses on...

You: What courses do they teach?
Bot: The faculty members teach various courses including...
```

### Example 3: Python API Usage

```python
from src.rag.chatbot import RAGChatbot
from src.utils.logger import setup_logging

# Setup
setup_logging("INFO")
chatbot = RAGChatbot()

# Single query
response = chatbot.chat("Who works on computer vision?")
print(response['response'])

# Faculty-specific query
faculty_info = chatbot.ask_about_faculty("Dr. John Doe")
print(faculty_info['response'])

# Topic-specific query
topic_info = chatbot.ask_about_research_topic("deep learning")
print(topic_info['response'])
```

## Troubleshooting

### Issue: OpenAI API Error

**Solution:** Ensure your API key is set in `.env`:
```env
OPENAI_API_KEY=sk-...
```

### Issue: Port 8080 In Use

If `start_web_server.py` fails with an error like:

```
ERROR:    [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8080): only one usage of each socket address (protocol/network address/port) is normally permitted
```

**Windows Solution:**

Run these commands in PowerShell to identify and free the port:

```powershell
# Show listener(s) on port 8080
netstat -ano | findstr ":8080"

# Show process details for a PID (example: 16936)
Get-Process -Id 16936 | Format-List Name,Id,Path,StartTime,MainWindowTitle

# Kill the process (PowerShell)
Stop-Process -Id 16936 -Force

# Or using taskkill
taskkill /PID 16936 /F

# Verify port freed
netstat -ano | findstr ":8080"

# Restart the web server
python .\start_web_server.py
```

**Linux/EC2 Solution:**

```bash
# Find process using port 8080
sudo lsof -i :8080

# Kill the process
sudo kill -9 <PID>

# Or stop Docker container
docker compose -f docker-compose.prod.yml restart chatbot
```

### Issue: Frontend Shows "Failed to Send Message"

**Symptom:** Browser console shows network errors or messages fail to send.

**Solution:** Update `static/js/chat.js` to use the correct API URL:

```javascript
// Change line 6 from:
const API_URL = 'http://localhost:8080';

// To (for production):
const API_URL = window.location.origin;

// Or hardcode EC2 IP:
const API_URL = 'http://13.200.45.148:8080';
```

Then restart the chatbot:
```bash
docker compose -f docker-compose.prod.yml restart chatbot
```

Clear browser cache: `Ctrl+Shift+R` or `Ctrl+F5`

### Issue: Neo4j Data Not Transferred to EC2

**Solution:** Use CSV export/import method:

**On Local Machine (Windows):**
```powershell
# Export from local Neo4j to CSV
python export_neo4j_to_csv.py

# Transfer to EC2
scp -i "C:/Users/your-username/Downloads/key.pem" pages.csv entities.csv mentions.csv ubuntu@13.200.45.148:~/iisc-chatbot/
```

**On EC2:**
```bash
# Import CSVs to Neo4j
cd ~/iisc-chatbot
pip3 install neo4j tqdm
python3 import_csv.py

# Verify import
docker exec iisc-neo4j-prod cypher-shell -u neo4j -p password \
  "MATCH (n) RETURN labels(n)[0] as type, count(n) ORDER BY count DESC"
```

If you want a quick analysis of token usage and context sizes (helpful for cost and truncation decisions), see `TOKEN_ANALYSIS.md` in the repository root.

### Issue: Neo4j Connection Failed

**Solution:** 
1. Verify Neo4j is running
2. Check credentials in `.env`
3. Test connection: `bolt://localhost:7687`

### Issue: spaCy Model Not Found

**Solution:**
```bash
python -m spacy download en_core_web_lg
```

### Issue: ChromaDB Permission Error

**Solution:** Ensure the data directory is writable:
```bash
mkdir -p data/chromadb
```

### Issue: Out of Memory

**Solution:** 
- Reduce chunk size in `config.yaml`
- Process smaller batches
- Increase system RAM or use cloud instance

### Issue: Slow Crawling

**Solution:** Adjust settings in `config.yaml`:
```yaml
crawler:
  settings:## Project Context

This project was developed as part of the **DS252 Introduction to Cloud Computing (Aug, 2025)** course at the Indian Institute of Science (IISc), Department of Computational and Data Sciences (CDS). The project Git repository and this README have been shared with the evaluation panel members.
    concurrent_requests: 16  # Increase concurrency
    download_delay: 0.5      # Reduce delay
```

## Pipeline Output Format

### Processed Page JSON

```json
{
  "page_id": "cds_iisc_ac_in_abc123",
  "url": "https://cds.iisc.ac.in/faculty/john-doe",
  "domain": "cds.iisc.ac.in",
  "page_type": "faculty",
  "title": "Dr. John Doe - Faculty Profile",
  "entities": {
    "PERSON": [{"text": "John Doe", "label": "PERSON"}],
    "RESEARCH_TOPIC": [{"text": "machine learning", "label": "RESEARCH_TOPIC"}]
  },
  "keywords": [
    {"keyword": "machine learning", "score": 0.85},
    {"keyword": "deep learning", "score": 0.78}
  ],
  "chunks": [
    {
      "chunk_id": 1,
      "text": "Dr. John Doe is a Professor...",
      "embedding": [0.123, -0.456, ...],
      "word_count": 250,
      "page_id": "cds_iisc_ac_in_abc123"
    }
  ]
}
```

## Use of AI and Coding Assistants

The team made responsible use of AI and coding assistants (such as GitHub Copilot, CatGPT, and similar tools) to help with tasks including code suggestions, documentation drafts, and debugging. All generated code and documentation were reviewed, tested, and adapted by the team to ensure correctness, security, and alignment with the course requirements.


