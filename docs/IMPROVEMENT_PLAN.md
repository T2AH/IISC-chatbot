# IISc Chatbot Improvement Plan

## Current Issues

### 1. **Neo4j Structure Problems** ❌
- Only has Page nodes (198K)
- No Person, Lab, ResearchTopic entities
- No relationships (MENTIONS, RESEARCHES, etc.)
- Retriever queries fail silently

### 2. **Poor Context Awareness** ❌
- No conversation memory
- Each query treated independently
- Can't follow up on previous questions
- Example: "what is his lab" → doesn't know who "his" refers to

### 3. **Retrieval Quality Issues** ❌
- Only using vector search (7-10 results)
- Graph search returns 0 results
- Not finding relevant chunks for specific queries
- URL boosting not helping much

### 4. **Query Understanding** ❌
- Doesn't expand context properly
- "dream Lab" → Should find "DREAM:Lab" or "Distributed Systems Lab"
- "CDS faculty" → Should prioritize CDS department pages

## Solutions

### Solution 1: Fix Neo4j Knowledge Graph (CRITICAL)

**Problem**: No entities or relationships exist

**Actions**:
1. Re-run NLP processing on crawled data to extract entities
2. Properly populate Neo4j with:
   - Person nodes (faculty, researchers)
   - Lab nodes (research groups)
   - ResearchTopic nodes
   - Course nodes
   - Proper relationships: WORKS_IN, LEADS, RESEARCHES, TEACHES

**Implementation**:
```python
# Fix the database import to actually extract and store entities
# Current: Only stores pages
# Needed: Extract entities from NLP-processed data and create nodes + relationships
```

### Solution 2: Add Conversation Memory (HIGH PRIORITY)

**Problem**: No context between queries

**Actions**:
1. Store conversation history in session
2. Use LangChain ConversationBufferMemory
3. Resolve pronouns (he/she/it/this/that) using conversation context
4. Pass recent context to retrieval

**Implementation**:
```python
# In langgraph_chatbot.py
from langchain.memory import ConversationBufferMemory

class LangGraphChatbot:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
    
    def chat(self, query, session_id=None):
        # Get conversation history
        history = self.memory.load_memory_variables({})
        
        # Resolve references using history
        expanded_query = self._resolve_references(query, history)
        
        # Use expanded query for retrieval
        context = self._retrieve_context(expanded_query)
        ...
```

### Solution 3: Improve Query Understanding

**Problem**: Queries not understood properly

**Actions**:
1. Add query expansion with synonyms
2. Entity recognition in user queries
3. Acronym expansion (CDS → Computational and Data Sciences)
4. Context-aware reformulation

**Implementation**:
```python
def _expand_query(self, query: str, conversation_history: list) -> str:
    """Expand query with context and synonyms"""
    
    # Extract entities mentioned in history
    previous_entities = self._extract_entities_from_history(conversation_history)
    
    # Resolve pronouns
    if has_pronoun(query):
        query = resolve_pronoun(query, previous_entities)
    
    # Expand acronyms
    query = expand_acronyms(query)  # CDS → Computational and Data Sciences
    
    # Add context from previous relevant queries
    if is_follow_up_question(query):
        previous_context = get_previous_query_context(conversation_history)
        query = f"{previous_context}. {query}"
    
    return query
```

### Solution 4: Better Hybrid Retrieval

**Problem**: Graph search returns nothing, vector search not focused

**Actions**:
1. Fix Neo4j schema first (Solution 1)
2. Add query-type detection (faculty query vs research query vs course query)
3. Adjust retrieval strategy based on query type
4. Use MMR (Maximal Marginal Relevance) for diversity

**Implementation**:
```python
def retrieve(self, query: str, query_type: str = None) -> List[Document]:
    # Detect query type
    if not query_type:
        query_type = self._detect_query_type(query)
    
    if query_type == "faculty":
        # Prioritize faculty pages, use graph for faculty relationships
        vector_results = self._vector_search(query, filters={"page_type": "faculty"})
        graph_results = self._graph_search_faculty(query)
        
    elif query_type == "lab":
        # Focus on lab pages, get lab members from graph
        vector_results = self._vector_search(query, filters={"page_type": "lab"})
        graph_results = self._graph_search_labs(query)
        
    elif query_type == "research":
        # Search research topics, get related faculty/labs
        vector_results = self._vector_search(query)
        graph_results = self._graph_search_topics(query)
    
    # Combine and rerank with MMR
    return self._mmr_rerank(vector_results + graph_results, query)
```

### Solution 5: Enhanced Reranking

**Problem**: Cross-encoder not enough, no conversation awareness

**Actions**:
1. Add conversation-aware reranking
2. Boost results that match conversation context
3. Use GPT-4 for final relevance scoring on top candidates

**Implementation**:
```python
def _rerank_with_context(self, results: List[Document], query: str, history: list):
    # Stage 1: Cross-encoder
    results = self._cross_encoder_rerank(results, query)
    
    # Stage 2: Conversation context boost
    if history:
        conversation_entities = extract_entities_from_history(history)
        for result in results:
            if any(entity in result.content for entity in conversation_entities):
                result.score *= 1.3  # Boost conversationally relevant results
    
    # Stage 3: GPT-4 relevance scoring for top 10
    top_results = results[:10]
    relevance_scores = self._gpt4_relevance_score(top_results, query, history)
    
    # Re-sort by relevance
    return sorted(zip(top_results, relevance_scores), key=lambda x: x[1], reverse=True)
```

## Priority Implementation Order

### Phase 1: Critical Fixes (Do First)
1. ✅ **Fix Neo4j population** - Run proper NLP processing and entity extraction
2. ✅ **Add conversation memory** - Store and use chat history
3. ✅ **Fix retriever schema** - Update queries to match actual Neo4j structure

### Phase 2: Query Understanding (Do Second)
4. **Query expansion** - Synonyms, acronyms, context
5. **Reference resolution** - Handle "he", "his lab", "this", etc.
6. **Query type detection** - Faculty vs Lab vs Research queries

### Phase 3: Better Retrieval (Do Third)
7. **Type-specific retrieval strategies**
8. **MMR for diversity**
9. **Conversation-aware reranking**

## Quick Wins (Can Do Now)

### 1. Add Simple Conversation Memory
```python
# In start_web_server.py or api.py
sessions = {}  # session_id -> conversation_history

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = []
    
    # Get history
    history = sessions[session_id]
    
    # Chat with history
    response = chatbot.chat(request.query, conversation_history=history)
    
    # Update history
    sessions[session_id].append({"role": "user", "content": request.query})
    sessions[session_id].append({"role": "assistant", "content": response["response"]})
    
    return response
```

### 2. Better System Prompt
```python
SYSTEM_PROMPT = """You are an expert assistant for the Indian Institute of Science (IISc) with deep knowledge of:
- Faculty members, their research areas, and publications
- Research labs and groups
- Computational and Data Sciences (CDS) department
- Academic programs and courses

When answering:
1. Be specific and cite sources
2. If asked about "his/her lab" or similar, refer to the person mentioned in recent conversation
3. For acronyms like CDS, expand to full department name
4. If information is unclear, say so rather than guessing
5. Provide URLs to relevant pages when available

Current conversation context will help you understand follow-up questions."""
```

### 3. Query Preprocessing
```python
def preprocess_query(query: str, history: list) -> str:
    """Quick query improvements"""
    
    # Expand common acronyms
    acronyms = {
        "cds": "Computational and Data Sciences",
        "csa": "Computer Science and Automation",
        "serc": "Supercomputer Education and Research Centre",
        "iisc": "Indian Institute of Science"
    }
    
    for acronym, full in acronyms.items():
        if f" {acronym} " in f" {query.lower()} " or query.lower().startswith(f"{acronym} "):
            query = query.lower().replace(acronym, full)
    
    # Resolve simple pronouns if history exists
    if history and any(word in query.lower() for word in ["his", "her", "their", "this", "that"]):
        # Get last mentioned person/lab
        last_entity = extract_last_entity(history)
        if last_entity:
            query = f"{query} (referring to {last_entity})"
    
    return query
```

## Expected Improvements

After implementing:

**Before**:
- Query: "which lab works on cloud computing" → Generic results, no lab-specific info
- Query: "tell me about his lab" → Fails completely, no context

**After**:
- Query: "which lab works on cloud computing" → DREAM:Lab, Prof. Yogesh Simmhan's lab
- Query: "tell me about his lab" → Understands "his" = Yogesh Simmhan, returns DREAM:Lab info

## Testing Plan

1. **Test conversational queries**:
   - "Who works on ML?"
   - "Tell me more about them"
   - "What is their lab?"
   - "Who are the students?"

2. **Test specific queries**:
   - "CDS faculty"
   - "DREAM lab"
   - "Yogesh Simmhan research"
   - "Cloud computing labs"

3. **Test follow-ups**:
   - Ask about a person → Ask about their publications
   - Ask about a lab → Ask about lab members
   - Ask about research topic → Ask about related faculty

## Estimated Effort

- **Phase 1 (Critical)**: 2-3 hours
- **Phase 2 (Understanding)**: 2-3 hours  
- **Phase 3 (Advanced)**: 3-4 hours

**Total**: 7-10 hours for full improvement

## Start Here

1. Run proper NLP processing to extract entities
2. Add conversation memory to API
3. Fix Neo4j population script
4. Test with conversational queries
