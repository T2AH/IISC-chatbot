# IISc Chatbot Architecture - Complete Explanation

**Date**: November 20, 2025  
**Status**: Current System + Planned Improvements

---

## 📊 Current System Overview

### Data Flow: User Query → Response

```
┌─────────────┐
│   USER      │
│   Query     │ "Which labs work on cloud computing?"
└──────┬──────┘
       │
       v
┌─────────────────────────────────────────────────────┐
│  1. QUERY PROCESSING                                 │
│     - Expand acronyms: "cds" → "computational..."   │
│     - Resolve context: "his lab" → add prev query   │
│     - Generate embedding (3072-dim vector)          │
└──────┬──────────────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────────────┐
│  2. HYBRID RETRIEVAL (Parallel)                     │
│                                                      │
│  ┌────────────────┐          ┌──────────────────┐  │
│  │ Vector Search  │          │  Graph Search    │  │
│  │  (ChromaDB)    │          │   (Neo4j)        │  │
│  │                │          │                  │  │
│  │ • Embeds query │          │ • Entity lookup  │  │
│  │ • Finds k=50   │          │ • MENTIONS rel   │  │
│  │ • Cosine sim   │          │ • Get top pages  │  │
│  └────────┬───────┘          └──────┬───────────┘  │
│           │                         │              │
│           └─────────┬───────────────┘              │
│                     │                              │
└─────────────────────┼──────────────────────────────┘
                      │
                      v
┌─────────────────────────────────────────────────────┐
│  3. RERANKING                                        │
│     - Cross-encoder scoring                         │
│     - Metadata boost (faculty pages, labs)          │
│     - URL pattern matching                          │
│     - Take top 7-10 results                         │
└──────┬──────────────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────────────┐
│  4. RESPONSE GENERATION                              │
│     - LLM (GPT-4/Claude) receives:                  │
│       • Original query                              │
│       • Top 7-10 context chunks                     │
│       • Conversation history                        │
│     - Generates answer with citations               │
└──────┬──────────────────────────────────────────────┘
       │
       v
┌─────────────┐
│  RESPONSE   │
│  to User    │
└─────────────┘
```

---

## 🗄️ Database Architecture

### ChromaDB (Vector Database)

```
Collection: iisc_research_docs
├── Documents: 198,196 chunks
├── Embeddings: text-embedding-3-large (3072 dimensions)
├── Metadata per chunk:
│   ├── url (source page)
│   ├── title
│   ├── page_type (faculty/lab/department/general)
│   ├── domain (cds.iisc.ac.in, etc.)
│   ├── chunk_index (0, 1, 2...)
│   └── timestamp
└── Search Method: Cosine similarity (L2 distance)
```

**How Vector Search Works:**
1. Query "cloud computing labs" → Embedding vector [3072 numbers]
2. Compare with all 198K document embeddings
3. Find k=50 most similar (lowest L2 distance)
4. Return chunks with metadata

**Problem**: Each chunk is independent - loses surrounding context!

---

### Neo4j (Knowledge Graph)

**Current Schema (After populate_neo4j_entities.py):**

```cypher
# Nodes
(:Page {page_id, url, title, page_type, domain})
(:Entity {name, type})  # type = Person/Lab/ResearchTopic/Organization

# Relationships
(:Page)-[:MENTIONS {count}]->(:Entity)
```

**Example Graph:**
```
(Page: "https://cds.iisc.ac.in/faculty/")
    ├─[:MENTIONS {count: 5}]→ (Entity {name: "Prof. Murugesh", type: "Person"})
    ├─[:MENTIONS {count: 3}]→ (Entity {name: "DREAM Lab", type: "Lab"})
    └─[:MENTIONS {count: 8}]→ (Entity {name: "Cloud Computing", type: "ResearchTopic"})
```

**How Graph Search Works:**
1. Query "cloud computing labs"
2. Extract terms: ["cloud", "computing", "labs"]
3. Cypher query:
```cypher
MATCH (p:Page)-[r:MENTIONS]->(e:Entity)
WHERE toLower(e.name) CONTAINS "cloud" OR toLower(e.name) CONTAINS "computing"
WITH e, count(DISTINCT p) as page_count, sum(r.count) as total_mentions
ORDER BY total_mentions DESC
LIMIT 20
RETURN e.name, e.type, page_count, [page.url FOR page IN pages]
```
4. Returns entities ranked by how often they're mentioned

**Current Problem**: Only basic MENTIONS relationship exists!

---

## 🔍 Where Relationships Are Defined

### 1. **Neo4j Schema Creation** (Database Level)

**File**: `setup_neo4j_fresh.py` or `populate_neo4j_entities.py`

```python
# This creates the STRUCTURE (what relationships CAN exist)
with session.run("""
    CREATE CONSTRAINT entity_name IF NOT EXISTS 
    FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE
""")
```

**Currently Defined**:
- `:MENTIONS` - Page mentions Entity (✅ EXISTS)

**Missing Relationships** (Need to Add):
- `:WORKS_AT` - Person works at Lab/Organization
- `:LEADS` - Person leads Lab
- `:RESEARCHES` - Person/Lab researches Topic
- `:PART_OF` - Lab is part of Department
- `:COLLABORATES_WITH` - Person/Lab collaborates with another

---

### 2. **Relationship Population** (Data Level)

**File**: `populate_neo4j_entities.py`

**Current Code** (lines ~180-200):
```python
# Creates MENTIONS relationship
session.run("""
    MERGE (e:Entity {name: $name, type: $type})
    MERGE (p:Page {page_id: $page_id})
    MERGE (p)-[r:MENTIONS]->(e)
    ON CREATE SET r.count = 1
    ON MATCH SET r.count = r.count + 1
""", name=entity_name, type=entity_type, page_id=page_id)
```

**What We Need to Add** (Advanced relationships):
```python
# If entity is a person AND page is faculty page:
if entity_type == "Person" and "faculty" in page_url:
    # Extract department from URL
    dept = extract_department(page_url)
    if dept:
        session.run("""
            MATCH (person:Entity {name: $person_name, type: 'Person'})
            MATCH (dept:Entity {name: $dept_name, type: 'Organization'})
            MERGE (person)-[:WORKS_AT]->(dept)
        """, person_name=person_name, dept_name=dept)

# If page mentions "led by Prof X" near "Y Lab":
if "led by" in text or "headed by" in text:
    # Extract leader-lab pairs
    session.run("""
        MATCH (person:Entity {name: $person_name, type: 'Person'})
        MATCH (lab:Entity {name: $lab_name, type: 'Lab'})
        MERGE (person)-[:LEADS]->(lab)
    """, person_name=person, lab_name=lab)
```

---

### 3. **Relationship Querying** (Retrieval Level)

**File**: `src/rag/retriever.py` - `_graph_search()` method (line ~240)

**Current Query** (Simple):
```python
query = """
    MATCH (p:Page)-[r:MENTIONS]->(e:Entity)
    WHERE toLower(e.name) CONTAINS $term
    RETURN e.name, count(p) as pages
"""
```

**What We Should Add** (Multi-hop):
```python
# Query: "Who leads labs working on cloud computing?"
query = """
    MATCH (topic:Entity {type: 'ResearchTopic'})-[:RESEARCHED_BY]-(lab:Entity {type: 'Lab'})
    MATCH (person:Entity {type: 'Person'})-[:LEADS]->(lab)
    WHERE toLower(topic.name) CONTAINS 'cloud computing'
    RETURN person.name, lab.name, topic.name
"""
```

---

## 🎯 Current Problems & Solutions

### Problem 1: Insufficient Context Retrieval ❌

**Issue**: 
- Retrieving k=50 chunks from 198K documents (0.025%)
- Each chunk is 500-1000 chars, missing surrounding context

**Solution**:
```python
# retriever.py - Increase initial retrieval
self.top_k_vectors = 100  # Was 50

# After getting top chunks, fetch neighbors
def _get_neighboring_chunks(self, chunk_metadata):
    """Fetch chunks before/after for more context"""
    url = chunk_metadata['url']
    chunk_index = chunk_metadata['chunk_index']
    
    # Get previous and next chunks from same page
    neighbors = self.chromadb.query(
        where={
            "url": url,
            "chunk_index": {"$in": [chunk_index-1, chunk_index+1]}
        }
    )
    return neighbors
```

---

### Problem 2: Query-Document Embedding Mismatch ❌

**Issue**:
- Documents embedded with full content
- Queries are short (5-10 words)
- Semantic gap causes poor matching

**Solution - Query Expansion**:
```python
def _expand_query_smart(self, query: str) -> str:
    """Expand query using multiple strategies"""
    
    # 1. Acronym expansion (already done ✅)
    expanded = expand_acronyms(query)
    
    # 2. Add synonyms
    if "lab" in query:
        expanded += " laboratory research group center"
    
    # 3. Add context keywords
    if "faculty" in query:
        expanded += " professor researcher scientist staff member"
    
    # 4. Use LLM to generate search terms
    search_terms = llm.generate(f"Generate 5 search terms for: {query}")
    expanded += " " + search_terms
    
    return expanded
```

---

### Problem 3: No Multi-Hop Graph Queries ❌

**Issue**: 
- Only using basic MENTIONS relationship
- Can't answer "Who leads labs researching X?"

**Solution - Add Complex Relationships**:

**Step 1**: Create relationship extraction script
**Step 2**: Populate new relationships
**Step 3**: Update retriever queries

See "Advanced Improvements" section below.

---

### Problem 4: No Persistent Chat History ❌

**Current**: 
- Sessions stored in memory (`sessions: Dict` in api.py)
- Lost on server restart
- No chat thread history in UI

**Solution**: 
- Store sessions in SQLite/PostgreSQL
- Add chat history UI with thread management
- See "Chat Persistence" section below.

---

## 💾 Chat Persistence Architecture (TO BE IMPLEMENTED)

### Database Schema

```sql
-- Chat threads
CREATE TABLE chat_threads (
    thread_id TEXT PRIMARY KEY,
    user_id TEXT,  -- Optional: for multi-user
    title TEXT,    -- Auto-generated from first query
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    status TEXT    -- 'active' or 'archived'
);

-- Individual messages
CREATE TABLE chat_messages (
    message_id TEXT PRIMARY KEY,
    thread_id TEXT REFERENCES chat_threads(thread_id),
    role TEXT,     -- 'user' or 'assistant'
    content TEXT,
    context_used INT,  -- Number of chunks used
    created_at TIMESTAMP
);

-- Message context (for debugging)
CREATE TABLE message_contexts (
    context_id TEXT PRIMARY KEY,
    message_id TEXT REFERENCES chat_messages(message_id),
    chunk_url TEXT,
    chunk_text TEXT,
    relevance_score FLOAT
);
```

### API Endpoints to Add

```python
# Get all threads
GET /api/threads
Response: [
    {
        "thread_id": "abc123",
        "title": "CDS Faculty Questions",
        "created_at": "2025-11-20 15:30:00",
        "message_count": 12
    }
]

# Get thread messages
GET /api/threads/{thread_id}/messages
Response: {
    "thread_id": "abc123",
    "messages": [
        {"role": "user", "content": "...", "timestamp": "..."},
        {"role": "assistant", "content": "...", "timestamp": "..."}
    ]
}

# Create new thread
POST /api/threads
Body: {"initial_query": "Tell me about DREAM Lab"}
Response: {"thread_id": "new123", "message": {...}}

# Continue thread
POST /api/threads/{thread_id}/messages
Body: {"query": "Who leads it?"}
Response: {"answer": "...", "context_used": 7}

# Delete thread
DELETE /api/threads/{thread_id}
```

### UI Changes Needed

```html
<!-- Sidebar: Thread List -->
<div class="sidebar">
    <button onclick="newThread()">+ New Chat</button>
    <div class="thread-list">
        <div class="thread-item active">
            <h4>CDS Faculty Questions</h4>
            <span>12 messages • 2 hours ago</span>
        </div>
        <div class="thread-item">
            <h4>DREAM Lab Research</h4>
            <span>5 messages • Yesterday</span>
        </div>
    </div>
</div>

<!-- Main: Chat Messages -->
<div class="chat-area">
    <div class="thread-header">
        <h3>CDS Faculty Questions</h3>
        <button onclick="deleteThread()">🗑️</button>
    </div>
    <div class="messages">
        <!-- Messages here -->
    </div>
</div>
```

---

## 📁 Improved Folder Structure

We'll organize the project properly:

```
bot/
├── src/                          # Source code
│   ├── api/                      # API layer (NEW)
│   │   ├── __init__.py
│   │   ├── routes.py            # All endpoints
│   │   ├── models.py            # Pydantic models
│   │   └── middleware.py        # CORS, auth, etc.
│   │
│   ├── database/                # Database clients
│   │   ├── chromadb_client.py   # Vector DB
│   │   ├── neo4j_client.py      # Graph DB
│   │   └── chat_db.py           # Chat persistence (NEW)
│   │
│   ├── rag/                     # RAG pipeline
│   │   ├── retriever.py         # Hybrid retrieval
│   │   ├── reranker.py          # Context reranking
│   │   ├── query_embedder.py    # Query embedding
│   │   └── langgraph_chatbot.py # Chat logic
│   │
│   └── utils/                   # Utilities
│       ├── query_expansion.py   # Query processing (NEW)
│       └── relationship_extractor.py  # Entity relationships (NEW)
│
├── scripts/                     # Setup & maintenance scripts (NEW)
│   ├── setup/
│   │   ├── setup_neo4j_fresh.py
│   │   └── populate_neo4j_entities.py
│   │
│   ├── maintenance/
│   │   ├── backup_databases.py
│   │   └── cleanup_old_sessions.py
│   │
│   └── testing/
│       ├── test_improvements.py
│       └── benchmark_retrieval.py
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE_EXPLANATION.md  (THIS FILE)
│   ├── API_REFERENCE.md
│   ├── DEPLOYMENT_GUIDE.md
│   └── IMPROVEMENT_ROADMAP.md
│
├── static/                      # Frontend
│   ├── index.html
│   ├── css/
│   │   └── style.css           # Separate styles (NEW)
│   └── js/
│       ├── chat.js             # Chat logic (NEW)
│       └── threads.js          # Thread management (NEW)
│
├── data/                        # Data storage
│   ├── chromadb/               # Vector embeddings
│   ├── chat_threads.db         # Chat persistence (NEW)
│   └── crawled_pages/          # Original crawl data
│
├── logs/                        # Application logs
├── tests/                       # Unit tests (NEW)
└── config.yaml                  # Configuration
```

---

## 🚀 Implementation Priority

### Phase 1: Critical Fixes (DONE ✅)
1. ✅ Fix acronym expansion
2. ✅ Add session management
3. ✅ Add context resolution
4. ✅ Populate Entity nodes

### Phase 2: Retrieval Improvements (NEXT)
1. ⏳ Increase retrieval k (50→100)
2. ⏳ Add neighboring chunk fetching
3. ⏳ Improve query expansion
4. ⏳ Add semantic caching

### Phase 3: Chat Persistence (HIGH PRIORITY)
1. ⏳ Create SQLite chat database
2. ⏳ Add thread management API
3. ⏳ Build thread history UI
4. ⏳ Add thread search/filter

### Phase 4: Advanced Graph Features
1. ⏳ Extract WORKS_AT relationships
2. ⏳ Extract LEADS relationships
3. ⏳ Extract RESEARCHES relationships
4. ⏳ Multi-hop graph queries

---

## 📊 Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| **Query latency** | 3-5s | 2-4s |
| **Retrieval k** | 50 | 100 |
| **Answer accuracy** | 70% | 90% |
| **Context awareness** | 80% | 95% |
| **Session persistence** | Memory only | Database backed |
| **Chat threads** | None | Full history |

---

## 🔧 Key Decisions to Make

### 1. Chat Persistence Database
- **Option A**: SQLite (simple, local, no setup)
- **Option B**: PostgreSQL (scalable, production-ready)
- **Recommendation**: Start with SQLite, migrate to Postgres later

### 2. Relationship Extraction
- **Option A**: Rule-based (regex patterns)
- **Option B**: NLP-based (spaCy, dependency parsing)
- **Option C**: LLM-based (GPT-4 extracts relationships)
- **Recommendation**: Hybrid (rules + NLP for accuracy)

### 3. Query Expansion Strategy
- **Option A**: Static rules (fast, predictable)
- **Option B**: LLM-generated (flexible, slower)
- **Option C**: Embedding-based (semantic similarity)
- **Recommendation**: Static rules + LLM fallback

---

**Next Steps**: 
1. Review this architecture
2. Decide on chat persistence approach
3. I'll implement the improvements systematically

Would you like me to proceed with implementing chat persistence or retrieval improvements first?
