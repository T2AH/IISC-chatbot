# IISc Research Chatbot

A production-ready Python application for building an AI-powered research and academic chatbot for the Indian Institute of Science (IISc). This system combines web crawling, NLP processing, knowledge graph storage, and retrieval-augmented generation (RAG) to help students and researchers find information about faculty, labs, courses, projects, and research topics.

## Project Context

This project was developed as part of the **DS252 Introduction to Cloud Computing (Aug, 2025)** course at the Indian Institute of Science (IISc), Department of Computational and Data Sciences (CDS). The project Git repository and this README have been shared with the evaluation panel members.

## Team 

This project was carried out by the following team members:
- Amitesh Pandey – [amiteshp@iisc.ac.in](mailto:amiteshp@iisc.ac.in)
- Harsh Saxena – [harshsaxena@iisc.ac.in](mailto:harshsaxena@iisc.ac.in)
- Abhinav Rawat – [abhinavrawat@iisc.ac.in](mailto:abhinavrawat@iisc.ac.in)

## Table of Contents

- [Access the Live Chatbot](#access-the-live-chatbot)
- [Features](#features)
- [Recent Updates](#recent-updates)
- [Architecture](#architecture)
- [System Design](#system-design)
- [Deployment](#deployment)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Embedding Upgrade](#embedding-upgrade)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Use of AI and Coding Assistants](#use-of-ai-and-coding-assistants)


## • Access the Live Chatbot

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

## • Features

- **Multi-Domain Web Crawling**: Structure-aware crawling for IISc sites and intelligent NLP-based crawling for external faculty/lab websites
- **Advanced NLP Processing**: Entity extraction (spaCy + BERT), keyword extraction (KeyBERT), and semantic embeddings (Sentence Transformers)
- **Dual Database Architecture**: 
  - Neo4j knowledge graph for entities and relationships
  - ChromaDB vector database for semantic search
- **RAG-Powered Chatbot**: LangGraph + OpenAI integration for intelligent, context-aware responses
- **Scalable Pipeline**: Modular architecture supporting the full pipeline from crawling to querying
- **Production Deployment**: AWS EC2 with Docker orchestration and auto-recovery

## • Recent Updates 

- **Embedding Upgrade**: Re-embedded all 198K documents with OpenAI `text-embedding-3-large` (3072 dimensions) for superior semantic matching
- **LangGraph Integration**: Upgraded from LangChain to LangGraph for stateful multi-turn conversations and advanced agent workflows
- **Production Deployment:** Chatbot deployed on AWS EC2 at `http://13.200.45.148:8080` with full Docker orchestration
- **Knowledge Graph Migration:** Successfully migrated 198,196 Page nodes and 31,304 Entity nodes from local Neo4j to production
- **CSV-based Import:** Implemented reliable CSV import pipeline for Neo4j data transfer (handles special characters correctly)
- **UI & Thread Management:** Modern sidebar UI with chat thread management. Frontend files in `static/` (`static/index.html`, `static/css/style.css`, `static/js/chat.js`). Chat threads persisted in SQLite (`data/chat_threads.db`)
- **Retrieval Improvements:** Hybrid retriever tuned for higher recall (larger `top_k_vectors`) with non-hardcoded department-aware query expansion
- **Neo4j & ChromaDB diagnostics:** Inspection scripts (`check_neo4j_state.py`, `check_chromadb_cds.py`) for validating entity and vector counts
- **CORS Configuration:** Fixed frontend API connectivity issues with proper CORS middleware and dynamic API_URL configuration
- **Docker Persistence:** All services configured with `restart: always` for automatic recovery on reboot

**Port 8080 troubleshooting**: If `start_web_server.py` fails due to port contention on `8080`, check and free the port on Windows PowerShell:

```powershell
netstat -ano | findstr ":8080"
taskkill /PID <pid> /F
```

**Quick test (local)**: Start the server and send a simple chat request to validate the API:

```powershell
python start_web_server.py
curl -X POST "http://localhost:8080/chat" -H "Content-Type: application/json" -d "{\"query\":\"Who are the faculty in the CDS department?\"}"
```

If you get a `422 Unprocessable Entity` on POST requests from the frontend, confirm the request body shape matches the API schema (some endpoints accept different JSON shapes depending on recent edits).


## • Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                 RAG Chatbot (LangGraph)                        │
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
           │                      └──────────┬───────────┘
           │                                 │
           └──────────────┬──────────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   NLP Processing       │
              │   - Entity Extraction  │
              │   - Keyword Extraction │
              │   - Text Chunking      │
              │   - Embeddings         │
              └──────────┬─────────────┘
                         │
                         ▼
              ┌────────────────────────┐
              │   Web Crawler          │
              │   - Scrapy Spiders     │
              │   - Dynamic Depth      │
              │   - Multi-Domain       │
              └────────────────────────┘
```

## • System Design

### Crawler Layer (Scrapy + BeautifulSoup)

**Core Functionality:**
- Multi-domain crawling: IISc main site + external faculty/lab websites
- Support for HTTP & HTTPS URLs with automatic protocol handling
- Respectful crawling: Respects robots.txt with internal research crawling override
- Resilient error handling: Retry failed/unresponsive requests, gracefully skip broken pages
- Structured extraction: Convert raw HTML to clean JSON with metadata

**Output Format:**
```json
{
  "page_id": "unique_identifier",
  "url": "https://example.com/page",
  "title": "Page Title",
  "domain": "example.com",
  "page_type": "faculty|lab|course|project",
  "crawl_date": "2025-11-28T10:00:00Z",
  "content": "Extracted text content...",
  "raw_html": "Optional: raw HTML for post-processing"
}
```

**Crawl Depth Rules:**
- **Department Sites** (IISc main domain): 3–4 levels deep
  - Level 1: Department homepage
  - Level 2: Faculty profiles, Labs, Courses
  - Level 3: Projects, Publications, Research details
  - Level 4: Resources, Additional materials
  
- **External Faculty/Lab Sites**: 2–3 levels deep
  - Level 1: Homepage
  - Level 2: Research areas, Projects, Publications
  - Level 3: Subproject details, Linked resources

**Dynamic Depth Adjustment:**
- Uses NLP filtering to allow deeper crawling for high-content pages (labs, projects)
- Stops early for low-content or irrelevant pages (news, menus, events)
- Content quality scoring prevents wasted requests on navigation-heavy pages

**Key Features:**
- Chunking: ~250 words per chunk for optimal embeddings
- Structure-aware crawling for IISc main site
- Generic NLP-based crawling for external sites
- User-Agent rotation and middleware support

### NLP Layer (spaCy + BERT + KeyBERT + Sentence Transformers)

**Text Processing Pipeline:**
1. **HTML Cleaning**: Extract visible text only, remove scripts/styles
2. **Text Normalization**: Remove extra whitespace, handle unicode
3. **Entity Recognition**: 
   - General entities: PERSON, ORG, LOCATION (spaCy)
   - Domain-specific: RESEARCH_TOPIC, LAB, PROJECT, COURSE_CODE (BERT)
4. **Keyword Extraction**: Extract top-K research keywords with diversity control (KeyBERT)
5. **Text Chunking**: Split into ~250-word chunks with 50-word overlap
6. **Embedding Generation**: Create vector embeddings (Sentence Transformers for local use, OpenAI for production)

**Entity Types Extracted:**
- `PERSON`: Faculty, researchers, authors
- `ORGANIZATION`: Labs, departments, institutes
- `RESEARCH_TOPIC`: ML, AI, Data Science, etc.
- `PROJECT`: Research projects, initiatives
- `COURSE_CODE`: Course identifiers (CS, DS, etc.)
- `PUBLICATION`: Papers, articles, publications

**Output Structure:**
```json
{
  "page_id": "unique_id",
  "url": "page_url",
  "entities": {
    "PERSON": [{"text": "Dr. John Doe", "score": 0.95}],
    "RESEARCH_TOPIC": [{"text": "machine learning", "score": 0.89}]
  },
  "keywords": [
    {"keyword": "deep learning", "score": 0.85},
    {"keyword": "neural networks", "score": 0.78}
  ],
  "chunks": [
    {
      "chunk_id": 1,
      "text": "Chunk text content...",
      "embedding": [0.123, -0.456, ...],
      "word_count": 245
    }
  ]
}
```

### Database Layer

**Neo4j Knowledge Graph:**
- Nodes: Faculty, Lab, Project, ResearchTopic, Course, Department, Organization
- Relationships: WORKS_IN, LEADS, CONDUCTS, COVERS, TEACHES, COLLABORATES_WITH, PUBLISHES, FUNDS
- Indexed and constrained for performance
- Full-text search support for entity queries

**ChromaDB Vector Store:**
- Stores text chunks with vector embeddings (3072-dim with OpenAI text-embedding-3-large)
- Metadata: page_id, url, domain, page_type, title, crawl_date
- Cosine similarity search for semantic retrieval
- Hybrid search combining vector + metadata filters

**Data Persistence:**
- Neo4j: 198K+ pages, 31K+ entities, 138K+ relationships
- ChromaDB: 198K vector embeddings with full metadata

## • Deployment

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
-  198,197 documents indexed in ChromaDB
-  198,196 Page nodes in Neo4j
-  31,304 Entity nodes (Labs, People, Topics, Organizations)
-  138,632 knowledge graph relationships

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
- CORS configured for secure frontend communication

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

# Check database health
docker exec iisc-neo4j-prod cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n) as total_nodes"
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

# Verify deployment
curl http://13.200.45.148:8080/health
```

## • Installation

### Prerequisites

- Python 3.9 or higher
- Neo4j 5.x (for knowledge graph)
- 8GB+ RAM recommended
- OpenAI API key (for production embeddings)
- Docker & Docker Compose (for containerized deployment)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd iisc-chatbot
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

1. Download and install Neo4j Desktop from [https://neo4j.com/download/](https://neo4j.com/download/)
2. Create a new database with secure password
3. Start the database
4. Verify connection on `bolt://localhost:7687`

### Step 6: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
```env
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password
EMBEDDING_MODEL=text-embedding-3-large
```

## • Configuration

The main configuration file is `config.yaml`. Key sections:

### Crawler Configuration

```yaml
crawler:
  depth_rules:
    department_sites:
      max_depth: 4
      timeout: 30
    external_sites:
      max_depth: 3
      timeout: 20
  settings:
    concurrent_requests: 16
    download_delay: 1.0
    respect_robots_txt: true
    retry_attempts: 3
    retry_delay: 5
```

### NLP Configuration

```yaml
nlp:
  text_processing:
    chunk_size: 250  # words per chunk
    chunk_overlap: 50
    min_chunk_size: 50
  models:
    spacy_model: en_core_web_lg
    embedding_model: text-embedding-3-large  # Production
    embedding_dimensions: 3072
    bert_model: bert-base-uncased
  entity_extraction:
    confidence_threshold: 0.7
    max_entities_per_type: 20
```

### RAG Configuration

```yaml
rag:
  retrieval:
    top_k_vectors: 5
    similarity_threshold: 0.7
    hybrid_weight: 0.6  # 60% vector, 40% graph
    use_reranking: true
  generation:
    model: gpt-4
    temperature: 0.7
    max_tokens: 500
    top_p: 0.9
```

## • Usage

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

**Crawl with depth limit:**
```bash
python main.py crawl --spider iisc --max-depth 2
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

With entity extraction:
```bash
python main.py process --input data/crawled_pages/pages.jsonl --extract-entities
```

#### 3. Database Import

Import processed data to Neo4j and ChromaDB:
```bash
python main.py import --input data/crawled_pages/processed_pages.jsonl
```

Verify import:
```bash
python main.py verify --check-entities --check-vectors
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

#### 6. Web Server

Start the FastAPI web server:
```bash
python start_web_server.py
```

Access API documentation: `http://localhost:8080/docs`

## • Project Structure

```
iisc-chatbot/
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
│   │   ├── entity_extraction.py     # Entity recognition (spaCy + BERT)
│   │   ├── keyword_extraction.py    # Keyword extraction (KeyBERT)
│   │   ├── embedding_generation.py  # Embeddings (Sentence Transformers)
│   │   ├── reembed_chromadb.py      # Re-embed with OpenAI text-embedding-3-large
│   │   └── pipeline.py              # NLP pipeline orchestration
│   │
│   ├── database/             # Database integration
│   │   ├── neo4j_client.py          # Neo4j knowledge graph operations
│   │   ├── chromadb_client.py       # ChromaDB vector store operations
│   │   └── manager.py               # Unified database interface
│   │
│   ├── rag/                  # RAG chatbot
│   │   ├── retriever.py             # Hybrid retrieval system
│   │   ├── reranker.py              # Query result reranking
│   │   └── chatbot.py               # LangGraph + OpenAI chatbot
│   │
│   ├── web/                  # Web API
│   │   ├── app.py                   # FastAPI application
│   │   ├── routes.py                # API endpoints
│   │   └── models.py                # Request/response schemas
│   │
│   ├── utils/                # Utilities
│   │   ├── logger.py                # Logging configuration
│   │   ├── config_loader.py         # Configuration management
│   │   └── helpers.py               # Helper functions
│   │
│   └── config.py             # Configuration entry point
│
├── static/                   # Frontend files
│   ├── index.html            # Web UI
│   ├── css/
│   │   └── style.css         # Styling
│   └── js/
│       └── chat.js           # Client-side logic
│
├── data/                     # Data directory (created at runtime)
│   ├── crawled_pages/        # Crawled raw data
│   ├── processed_pages/      # Processed data
│   ├── chromadb/             # ChromaDB persistence
│   └── chat_threads.db       # Chat history (SQLite)
│
├── logs/                     # Log files (created at runtime)
│
├── docker-compose.prod.yml   # Production orchestration
├── Dockerfile                # Container definition
├── main.py                   # Main CLI entry point
├── start_web_server.py       # Web server entry point
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```


## • Embedding Upgrade

### Re-embed Documents with OpenAI text-embedding-3-large

**Upgraded 198K documents** from basic embeddings to OpenAI `text-embedding-3-large` (3072 dimensions) for superior semantic matching.

**Run the upgrade:**
```bash
python scripts/reembed_chromadb.py
```

**What it does:**
- Creates automatic backup of existing embeddings
- Generates new embeddings in optimized batches of 100
- Replaces old collection with verified new embeddings
- Tracks metadata: model version, embedding dimension, timestamp

**Performance:**
- Estimated cost: $7-10 for ~200k documents
- Estimated time: 10-20 minutes
- Speed: ~100 docs/minute with batch processing

**Key Improvements:**
- Higher dimensional embeddings (3072 vs 384)
- Better semantic matching for research queries
- Superior performance on specialized domain terms
- Metadata tracking for version control

**Verification:**
```bash
# Check embedding dimensions
python -c "import chromadb; db = chromadb.PersistentClient('./data/chromadb'); c = db.get_collection('iisc_research_docs'); print(f'Embeddings: {len(c.get()[\"embeddings\"][0])} dims, {c.count()} docs')"

# Verify metadata
curl http://localhost:8080/api/stats
```

## • Examples

### Example 1: Crawl and Process CDS Department

```bash
# Step 1: Crawl
python main.py crawl --spider iisc --url https://cds.iisc.ac.in

# Step 2: Process (use the generated file from step 1)
python main.py process --input data/crawled_pages/pages_iisc_spider_*.jsonl

# Step 3: Import to databases
python main.py import --input data/crawled_pages/processed_pages_*.jsonl

# Step 4: Query the chatbot
python main.py chat --query "What research is being done in machine learning at CDS?"
```

### Example 2: Interactive Chatbot Session

```bash
python main.py chat --interactive
```

```
You: Who are the faculty in the CDS department?
Bot: The Computational and Data Sciences (CDS) department has several distinguished faculty members working across AI, ML, and Data Science...

You: Tell me more about their machine learning research
Bot: Machine learning research at CDS focuses on several key areas including deep learning, reinforcement learning, and applications to real-world problems...

You: What courses do they teach?
Bot: The faculty members teach various courses including CS269: Machine Learning, CS270: Deep Learning, DS230: Data Science Fundamentals...

You: exit
```

### Example 3: Upgrade Embeddings

```bash
# Re-embed with OpenAI text-embedding-3-large
python scripts/reembed_chromadb.py

# Follow prompts to confirm and monitor progress
# ~15 minutes for 198K documents
# Results visible immediately in semantic search quality
```

### Example 4: Python API Usage

```python
from src.rag.chatbot import RAGChatbot
from src.utils.logger import setup_logging

# Setup
setup_logging("INFO")
chatbot = RAGChatbot()

# Single query with retrieval
response = chatbot.chat("Who works on computer vision?")
print(f"Answer: {response['response']}")
print(f"Sources: {response['sources']}")

# Faculty-specific query
faculty_info = chatbot.ask_about_faculty("Dr. John Doe")
print(f"Faculty Info: {faculty_info['response']}")

# Topic-specific query
topic_info = chatbot.ask_about_research_topic("deep learning")
print(f"Topic Info: {topic_info['response']}")
print(f"Related Faculty: {topic_info['related_entities']}")
```

## • Troubleshooting

### Issue: OpenAI API Error

**Symptom:** `AuthenticationError` or `APIError`

**Solution:** Ensure your API key is set in `.env`:
```env
OPENAI_API_KEY=sk-...
```

Verify key is valid:
```bash
python -c "from openai import OpenAI; OpenAI(api_key='your_key').models.list()"
```

### Issue: Port 8080 In Use

**Symptom:** `error while attempting to bind on address ('0.0.0.0', 8080)`

**Windows Solution:**

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

**Symptom:** Browser console shows network errors or CORS issues

**Solution:** Update `static/js/chat.js` to use correct API URL:

```javascript
// For production (auto-detect)
const API_URL = window.location.origin;

// Or hardcode EC2 IP
const API_URL = 'http://13.200.45.148:8080';

// Or localhost for development
const API_URL = 'http://localhost:8080';
```

Restart chatbot:
```bash
docker compose -f docker-compose.prod.yml restart chatbot
```

Clear browser cache: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)

### Issue: Neo4j Data Not Transferred to EC2

**Solution:** Use CSV export/import method:

**On Local Machine (Windows):**
```powershell
# Export from local Neo4j to CSV
python scripts/export_neo4j_to_csv.py

# Transfer to EC2
scp -i "C:/Users/your-username/Downloads/key.pem" pages.csv entities.csv mentions.csv ubuntu@13.200.45.148:~/iisc-chatbot/
```

**On EC2:**
```bash
# Import CSVs to Neo4j
cd ~/iisc-chatbot
pip3 install neo4j tqdm
python3 scripts/import_csv.py

# Verify import
docker exec iisc-neo4j-prod cypher-shell -u neo4j -p password \
  "MATCH (n) RETURN labels(n)[0] as type, count(n) ORDER BY count DESC"
```

### Issue: Neo4j Connection Failed

**Symptom:** `ConnectionError: Failed to establish connection`

**Solution:**
1. Verify Neo4j is running:
   ```bash
   docker compose -f docker-compose.prod.yml ps neo4j
   ```

2. Check credentials in `.env`

3. Test connection:
   ```bash
   python -c "from neo4j import GraphDatabase; GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')).close()"
   ```

### Issue: spaCy Model Not Found

**Symptom:** `OSError: [E050] Can't find model 'en_core_web_lg'`

**Solution:**
```bash
python -m spacy download en_core_web_lg
```

### Issue: ChromaDB Permission Error

**Symptom:** `PermissionError: [Errno 13] Permission denied`

**Solution:** Ensure data directory is writable:
```bash
mkdir -p data/chromadb
chmod 755 data/chromadb
```

### Issue: Out of Memory During Processing

**Symptom:** Process crashes with memory exhaustion

**Solution:**
- Reduce chunk size in `config.yaml`: `chunk_size: 150`
- Process smaller batches: `python main.py process --batch-size 100`
- Increase system RAM or use cloud instance with more memory

### Issue: Slow Crawling

**Symptom:** Crawling takes excessive time

**Solution:** Adjust settings in `config.yaml`:
```yaml
crawler:
  settings:
    concurrent_requests: 32  # Increase from default 16
    download_delay: 0.5      # Reduce from default 1.0
    timeout: 15              # Reduce timeout
```

Or use command-line options:
```bash
python main.py crawl --spider iisc --concurrent 32
```

## • Pipeline Output Format

### Crawled Page JSON

```json
{
  "page_id": "cds_iisc_ac_in_faculty_john_doe_abc123",
  "url": "https://cds.iisc.ac.in/faculty/john-doe",
  "domain": "cds.iisc.ac.in",
  "page_type": "faculty",
  "title": "Dr. John Doe - Faculty Profile",
  "crawl_date": "2025-11-28T10:00:00Z",
  "content": "Dr. John Doe is a Professor in the Computational and Data Sciences department...",
  "metadata": {
    "language": "en",
    "content_length": 5432,
    "status_code": 200
  }
}
```

### Processed Page JSON

```json
{
  "page_id": "cds_iisc_ac_in_abc123",
  "url": "https://cds.iisc.ac.in/faculty/john-doe",
  "domain": "cds.iisc.ac.in",
  "page_type": "faculty",
  "title": "Dr. John Doe - Faculty Profile",
  "entities": {
    "PERSON": [
      {"text": "John Doe", "label": "PERSON", "score": 0.98},
      {"text": "Jane Smith", "label": "PERSON", "score": 0.92}
    ],
    "RESEARCH_TOPIC": [
      {"text": "machine learning", "label": "RESEARCH_TOPIC", "score": 0.89},
      {"text": "deep learning", "label": "RESEARCH_TOPIC", "score": 0.85}
    ]
  },
  "keywords": [
    {"keyword": "neural networks", "score": 0.87},
    {"keyword": "optimization", "score": 0.82},
    {"keyword": "data science", "score": 0.79}
  ],
  "chunks": [
    {
      "chunk_id": 1,
      "text": "Dr. John Doe is a Professor in the Computational and Data Sciences department at IISc...",
      "embedding": [0.123, -0.456, 0.789, ...],  // 3072 dimensions
      "word_count": 247,
      "page_id": "cds_iisc_ac_in_abc123"
    },
    {
      "chunk_id": 2,
      "text": "His research interests span machine learning, deep learning, and their applications...",
      "embedding": [0.234, -0.567, 0.890, ...],
      "word_count": 241,
      "page_id": "cds_iisc_ac_in_abc123"
    }
  ]
}
```

## • Use of AI and Coding Assistants

The team made responsible use of AI and coding assistants (such as GitHub Copilot, ChatGPT, and similar tools) to help with tasks including code suggestions, documentation drafts, and debugging. All generated code and documentation were reviewed, tested, and adapted by the team to ensure correctness, security, and alignment with the course requirements.

---

**Developed with dedication for DS252: Introduction to Cloud Computing (Aug, 2025)**
