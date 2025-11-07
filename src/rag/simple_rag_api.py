#!/usr/bin/env python3
"""
Simple FastAPI RAG Server for CDS Question Answering
Clean HTML frontend with Ollama integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import requests
import chromadb
from sentence_transformers import SentenceTransformer
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answer: str
    query: str

# Initialize FastAPI app
app = FastAPI(title="CDS RAG System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleRAGService:
    def __init__(self, chroma_dir="./chroma_db"):
        """Initialize simple RAG service"""
        logger.info("🚀 Initializing Simple RAG Service...")
        
        # ChromaDB configuration
        self.chroma_dir = chroma_dir
        self.collection_name = "cds_hierarchical_chunks"
        
        # Ollama configuration
        self.ollama_url = "http://localhost:11434"
        self.model_name = "qwen2.5:7b"
        
        # Load embedding model
        logger.info("📥 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        logger.info(f"📂 Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection(name=self.collection_name)
        
        # Test connections
        self._test_connections()
        logger.info("✅ Simple RAG Service initialized!")
    
    def _test_connections(self):
        """Test ChromaDB and Ollama connections"""
        # Test ChromaDB
        count = self.collection.count()
        logger.info(f"✅ ChromaDB: {count} chunks available")
        
        if count == 0:
            raise Exception("ChromaDB collection is empty!")
        
        # Test Ollama
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                logger.info(f"✅ Ollama models: {model_names}")
                
                if self.model_name not in model_names:
                    if model_names:
                        logger.warning(f"Using {model_names[0]} instead of {self.model_name}")
                        self.model_name = model_names[0]
                    else:
                        raise Exception("No Ollama models available!")
        except Exception as e:
            logger.error(f"❌ Ollama error: {e}")
            raise
    
    def get_chunk_by_id(self, chunk_id: str):
        """Fetch a specific chunk by its chunk_id"""
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas']
            )
            if results and results['documents']:
                return {
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
        except Exception as e:
            logger.debug(f"Could not fetch chunk {chunk_id}: {e}")
        return None
    
    def expand_hierarchical_context(self, chunk_id: str, metadata: dict, max_expansion: int = 3):
        """Expand context by fetching parent and sibling chunks"""
        expanded = []
        parent_id = metadata.get('parent_id', '')
        doc_id = metadata.get('doc_id', -1)
        
        # Fetch parent chunk for broader context
        if parent_id and parent_id != '':
            parent_chunk = self.get_chunk_by_id(parent_id)
            if parent_chunk:
                expanded.append({
                    'document': parent_chunk['document'],
                    'metadata': parent_chunk['metadata'],
                    'relation': 'parent'
                })
                logger.info(f"  📤 Expanded to parent: {parent_id}")
        
        # Fetch sibling chunks (chunks from SAME document with same parent_id)
        if parent_id and parent_id != '' and doc_id >= 0:
            try:
                # Query for chunks with same doc_id AND parent_id
                sibling_results = self.collection.get(
                    where={
                        "$and": [
                            {"doc_id": doc_id},
                            {"parent_id": parent_id}
                        ]
                    },
                    limit=max_expansion + 5,  # Get more to filter
                    include=['documents', 'metadatas']
                )
                
                sibling_count = 0
                for i, (doc, meta) in enumerate(zip(
                    sibling_results.get('documents', []),
                    sibling_results.get('metadatas', [])
                )):
                    # Skip the current chunk itself
                    chunk_id_meta = meta.get('chunk_id', '')
                    if chunk_id_meta != chunk_id and sibling_count < max_expansion:
                        expanded.append({
                            'document': doc,
                            'metadata': meta,
                            'relation': 'sibling'
                        })
                        sibling_count += 1
                
                if sibling_count > 0:
                    logger.info(f"  👥 Found {sibling_count} sibling chunks")
            except Exception as e:
                logger.error(f"Could not fetch siblings: {e}")
        
        return expanded
    
    def extract_faculty_from_lab_description(self, lab_chunk_text: str, lab_name: str):
        """Use LLM to extract faculty name from lab description, specific to the given lab"""
        prompt = f"""Extract the faculty member's name who leads a lab matching "{lab_name}".

Text (may contain multiple labs):
{lab_chunk_text[:800]}

IMPORTANT:
- The lab name may appear with variations: "DREAM Lab", "DREAM:Lab", "DREAMLab", etc.
- Match flexibly - look for labs that are similar to "{lab_name}"
- Return ONLY the faculty name for the matching lab
- If no matching lab is found, return "NONE"

Examples:
- Query: "Dream Lab", Text has "DREAM:Lab Faculty: Yogesh Simmhan" → Return: "Yogesh Simmhan"
- Query: "CSL", Text has "Cloud Systems Lab (CSL) Faculty: J. Lakshmi" → Return: "J. Lakshmi"  
- Query: "Dream Lab", Text has "BioMedIA Lab Faculty: Vaanathi" → Return: "NONE"

Return ONLY the faculty name or "NONE":
"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                faculty_name = result.get('response', '').strip()
                # Clean up the response
                faculty_name = faculty_name.replace('Faculty:', '').replace('Dr.', '').strip()
                
                # Skip if LLM returned NONE (lab not found in this chunk)
                if faculty_name.upper() == 'NONE' or len(faculty_name) < 3:
                    logger.info(f"  ⊘ No faculty found for '{lab_name}' in this chunk")
                    return None
                    
                logger.info(f"  🎓 Extracted faculty name for {lab_name}: {faculty_name}")
                return faculty_name
        except Exception as e:
            logger.warning(f"Could not extract faculty name: {e}")
        return None
    
    def hierarchical_graph_search(self, entities: list, intent: str, top_k: int = 10):
        """
        Navigate the document hierarchy intelligently using node_type and relationships.
        
        Strategy for "students of Dream Lab":
        1. Find root/section nodes mentioning the lab
        2. Extract faculty name from lab description using LLM
        3. Search for that faculty member's page
        4. Navigate to student chunks from those faculty pages
        5. Return student-specific chunks
        """
        logger.info(f"🗺️  Starting hierarchical graph traversal for entities: {entities}")
        
        all_chunks = []
        faculty_names_found = set()
        
        # Filter entities - skip generic terms like "students", "faculty", "people"
        skip_terms = {'student', 'students', 'faculty', 'people', 'member', 'members', 'researcher', 'researchers'}
        filtered_entities = [e for e in entities if e.lower() not in skip_terms]
        
        if not filtered_entities:
            logger.info("  ⚠️ No specific entities found after filtering generic terms")
            return []
        
        logger.info(f"  🎯 Filtered entities (removed generic terms): {filtered_entities}")
        
        # STEP 1: Find entry points - search root/section nodes for entities
        for entity in filtered_entities:
            query_embedding = self.embedding_model.encode(entity).tolist()
            
            # Search in root/section nodes first (overview pages, lab pages)
            root_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                where={
                    "$or": [
                        {"node_type": {"$eq": "root"}},
                        {"node_type": {"$eq": "section"}}
                    ]
                }
            )
            
            logger.info(f"  🔍 Found {len(root_results['ids'][0])} root/section nodes for '{entity}'")
            
            # STEP 2: Extract faculty names from lab descriptions
            for idx, (chunk_id, doc, meta) in enumerate(zip(
                root_results['ids'][0],
                root_results['documents'][0],
                root_results['metadatas'][0]
            )):
                all_chunks.append({'document': doc, 'metadata': meta, 'source': 'root_search'})
                
                # If this chunk mentions "Faculty:" or looks like a lab description, extract the PI
                # IMPORTANT: Only extract faculty for THIS specific entity/lab
                if 'faculty:' in doc.lower() or 'lab description' in doc.lower():
                    faculty_name = self.extract_faculty_from_lab_description(doc, entity)
                    if faculty_name and len(faculty_name) > 3:
                        faculty_names_found.add(faculty_name)
                        logger.info(f"  ✅ Found PI for '{entity}': {faculty_name}")
        
        # STEP 3: Search for faculty pages for the extracted faculty names
        if 'student' in intent.lower() or 'people' in intent.lower() or 'member' in intent.lower():
            for faculty_name in faculty_names_found:
                logger.info(f"  🔍 Searching for {faculty_name}'s page...")
                
                # Search for faculty member's page
                faculty_embedding = self.embedding_model.encode(faculty_name).tolist()
                faculty_results = self.collection.query(
                    query_embeddings=[faculty_embedding],
                    n_results=10
                )
                
                # STEP 4: From faculty pages, get student chunks
                for f_chunk_id, f_doc, f_meta in zip(
                    faculty_results['ids'][0],
                    faculty_results['documents'][0],
                    faculty_results['metadatas'][0]
                ):
                    doc_id = f_meta.get('doc_id', -1)
                    url = f_meta.get('url', '')
                    
                    # Add the faculty chunk itself
                    all_chunks.append({'document': f_doc, 'metadata': f_meta, 'source': 'faculty_page'})
                    
                    # Get ALL student chunks from the same document (faculty page)
                    if doc_id >= 0:
                        try:
                            related_chunks = self.collection.get(
                                where={
                                    "$and": [
                                        {"doc_id": {"$eq": doc_id}},
                                        {"chunk_type": {"$eq": "student_research"}}
                                    ]
                                },
                                limit=20,
                                include=['documents', 'metadatas']
                            )
                            
                            logger.info(f"  👥 Found {len(related_chunks['documents'])} student chunks from {faculty_name}'s page (doc {doc_id})")
                            
                            for s_doc, s_meta in zip(related_chunks['documents'], related_chunks['metadatas']):
                                all_chunks.append({
                                    'document': s_doc,
                                    'metadata': s_meta,
                                    'source': 'student_from_faculty_page'
                                })
                        except Exception as e:
                            logger.error(f"Error fetching student chunks: {e}")
        
        logger.info(f"✅ Hierarchical graph search found {len(all_chunks)} total chunks")
        return all_chunks[:top_k * 3]  # Return more for later filtering
    
    def decompose_query_with_llm(self, query: str):
        """Use LLM to understand query structure and extract key search terms + metadata filters"""
        decomposition_prompt = f"""You are a query analyzer for an academic database about IISc CDS department.

Analyze this user query and extract key information:

USER QUERY: "{query}"

Identify:
1. KEY ENTITIES: Lab names, people names, departments, specific topics (HIGHEST PRIORITY)
2. INTENT: What information is needed (students/faculty/research/publications/etc)
3. METADATA FILTERS: What metadata should we filter by?
   - chunk_type: "student_research", "faculty_info", "research_activity", "publication", etc.
   - chunk_research_areas: specific topics like "machine learning", "data science", "quantum computing"
   - node_type: "faculty", "student", "lab", "course"
4. SEARCH QUERIES: Generate 4-5 different search queries to find this information

For example:
- If asking about students → filter by chunk_type: "student_research"
- If asking about a specific research area → filter by chunk_research_areas
- If asking about a faculty member → filter by node_type: "faculty"

Think step-by-step:
- What are the main entities? (e.g., "DREAM Lab" → also search for its PI/faculty page)
- What type of chunks contain this info? (e.g., students → "student_research" chunks)
- What is the user asking for? (e.g., "students" → search for "current students", "PhD candidates")

Respond in this exact JSON format:
{{
  "entities": ["entity1", "entity2"],
  "intent": "what_user_wants",
  "metadata_filters": {{
    "chunk_type": "type_if_applicable",
    "chunk_research_areas": ["area1", "area2"],
    "node_type": "node_type_if_applicable"
  }},
  "search_queries": ["query1", "query2", "query3", "query4"]
}}"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": decomposition_prompt,
                    "stream": False,
                    "temperature": 0.1
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                # Extract JSON from response
                import json
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    query_analysis = json.loads(json_match.group())
                    logger.info(f"🧠 LLM extracted entities: {query_analysis.get('entities', [])}")
                    logger.info(f"🎯 LLM identified intent: {query_analysis.get('intent', 'unknown')}")
                    logger.info(f"🏷️  LLM metadata filters: {query_analysis.get('metadata_filters', {})}")
                    return query_analysis
                else:
                    logger.warning(f"⚠️ LLM response not in JSON format, falling back to simple extraction")
                    return None
            else:
                logger.error(f"LLM decomposition failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error in LLM query decomposition: {e}")
            return None
    
    def generate_search_queries_with_llm(self, query: str, components: dict):
        """Use LLM to intelligently analyze query and generate targeted search queries"""
        prompt = f"""Analyze this query and generate search terms to find the answer in a research department database.

USER QUERY: "{query}"

Your task:
1. Identify the KEY ENTITIES (labs, people, departments - these are MOST IMPORTANT)
2. Identify the INTENT (what info is needed: students, faculty, research, etc.)
3. Generate 4-5 SPECIFIC search queries that would find this information

Rules:
- Focus on the ENTITIES first (lab names, people names are critical)
- Include variations (e.g., "DREAM lab" and "DREAM:Lab")
- Don't include generic words like "of", "the", "can you"
- Be specific and targetedReturn ONLY the search queries, one per line. Example format:

DREAM lab students current
DREAM:Lab PhD research members
Yogesh Simmhan DREAM group
DREAM laboratory faculty page

Now generate for: "{query}"
"""

        try:
            logger.info("🧠 Asking LLM to generate search queries...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}  # Some creativity but mostly focused
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                # Split by newlines and clean
                queries = [q.strip() for q in generated_text.split('\n') if q.strip() and len(q.strip()) > 5]
                # Remove any lines that look like explanations (start with capital letters followed by colon)
                queries = [q for q in queries if not (q[0].isupper() and ':' in q[:20])]
                logger.info(f"✅ LLM generated queries: {queries[:5]}")
                return queries[:5]  # Limit to 5
        except Exception as e:
            logger.warning(f"Could not generate LLM queries: {e}")
        
        return []
    
    def search_context(self, query: str, top_k: int = 15, use_hierarchy: bool = True):
        """Search for relevant context with LLM-powered query understanding and metadata filtering"""
        logger.info(f"🔍 Searching for: '{query}'")
        
        # Use LLM to decompose query and generate smart search queries + metadata filters
        llm_analysis = self.decompose_query_with_llm(query)
        
        # Extract metadata filters if LLM provided them
        # NOTE: We use LENIENT filtering - only chunk_type, and it's optional
        # This prevents the filter from being too restrictive and missing results
        metadata_filters = {}
        if llm_analysis and llm_analysis.get('metadata_filters'):
            filters_from_llm = llm_analysis['metadata_filters']
            
            # ONLY use chunk_type filter, and only if it's specific
            # Ignore node_type because the DB uses "root"/"section" not "lab"/"faculty"
            if filters_from_llm.get('chunk_type'):
                chunk_type = filters_from_llm['chunk_type']
                # Only apply filter if it's a recognized type
                if chunk_type in ['student_research', 'faculty_info', 'research_activity', 'publication', 'academic_program']:
                    metadata_filters = {"chunk_type": {"$eq": chunk_type}}
                    logger.info(f"🏷️  Filtering by chunk_type: {chunk_type}")
                else:
                    logger.info(f"⚠️  Ignoring unrecognized chunk_type: {chunk_type}")
            
            # Log ignored filters for debugging
            if filters_from_llm.get('node_type'):
                logger.info(f"ℹ️  Ignoring node_type filter (not reliable): {filters_from_llm['node_type']}")
        
        # Build search queries from LLM analysis or fallback to original
        if llm_analysis and llm_analysis.get('search_queries'):
            queries_to_search = [query] + llm_analysis['search_queries']
            logger.info(f"🔄 Using {len(queries_to_search)} LLM-generated query variations")
        else:
            queries_to_search = [query]
            logger.info(f"⚠️ LLM decomposition unavailable, using original query only")
        
        all_results_ids = set()
        all_results_docs = []
        all_results_metas = []
        all_results_ids_list = []
        
        # Search with each query variation (with metadata filtering if available)
        for i, search_query in enumerate(queries_to_search):
            logger.info(f"  {i+1}. '{search_query}'")
            query_embedding = self.embedding_model.encode(search_query).tolist()
            
            # Apply metadata filters if available
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k
            }
            if metadata_filters:
                query_params["where"] = metadata_filters
            
            results = self.collection.query(**query_params)
            
            # Collect unique results
            for rid, rdoc, rmeta in zip(
                results['ids'][0] if results['ids'] else [],
                results['documents'][0] if results['documents'] else [],
                results['metadatas'][0] if results['metadatas'] else []
            ):
                if rid not in all_results_ids:
                    all_results_ids.add(rid)
                    all_results_docs.append(rdoc)
                    all_results_metas.append(rmeta)
                    all_results_ids_list.append(rid)
        
        ids = all_results_ids_list
        documents = all_results_docs
        metadatas = all_results_metas
        
        logger.info(f"✅ Found {len(documents)} unique matches from multi-query search")
        
        # NEW: Use hierarchical graph search if we have entities and intent
        # This navigates the document hierarchy intelligently using node_type
        if llm_analysis and llm_analysis.get('entities') and llm_analysis.get('intent'):
            entities = llm_analysis['entities']
            intent = llm_analysis['intent']
            
            # Only use graph search for specific intents (students, faculty, people)
            if any(keyword in intent.lower() for keyword in ['student', 'faculty', 'people', 'member', 'researcher']):
                logger.info(f"🗺️  Activating hierarchical graph search for intent: {intent}")
                graph_chunks = self.hierarchical_graph_search(entities, intent, top_k=top_k)
                
                # Merge graph search results with regular search results
                for chunk in graph_chunks:
                    chunk_id = chunk['metadata'].get('chunk_id', '')
                    if chunk_id and chunk_id not in all_results_ids:
                        all_results_ids.add(chunk_id)
                        all_results_docs.append(chunk['document'])
                        all_results_metas.append(chunk['metadata'])
                        all_results_ids_list.append(chunk_id)
                
                # Update our working sets
                ids = all_results_ids_list
                documents = all_results_docs
                metadatas = all_results_metas
                logger.info(f"✅ After graph search: {len(documents)} total chunks")
        
        # Hierarchical expansion
        if use_hierarchy and documents:
            logger.info(f"🌳 Expanding with hierarchical context...")
            
            expanded_docs = []
            expanded_metas = []
            seen_chunks = set(ids)  # Track to avoid duplicates
            
            # Add initial results
            for doc, meta in zip(documents, metadatas):
                expanded_docs.append(doc)
                expanded_metas.append(meta)
            
            # Expand each top result with parent/sibling context
            # Increase expansion to get more related chunks from same page
            expansion_count = 5  # Expand top 5 results
            expansion_limit = 10  # Get up to 10 siblings for each
            
            for i, (chunk_id, doc, meta) in enumerate(zip(ids[:expansion_count], documents[:expansion_count], metadatas[:expansion_count])):
                logger.info(f"  🔗 Expanding chunk {i+1}: {chunk_id}")
                
                hierarchical_chunks = self.expand_hierarchical_context(chunk_id, meta, max_expansion=expansion_limit)
                
                for h_chunk in hierarchical_chunks:
                    h_chunk_id = h_chunk['metadata'].get('chunk_id', '')
                    if h_chunk_id and h_chunk_id not in seen_chunks:
                        expanded_docs.append(h_chunk['document'])
                        expanded_metas.append(h_chunk['metadata'])
                        seen_chunks.add(h_chunk_id)
            
            logger.info(f"📊 Total chunks after expansion: {len(expanded_docs)}")
            
            # LIMIT total context to prevent overwhelming the LLM
            max_chunks = 30
            if len(expanded_docs) > max_chunks:
                logger.info(f"⚠️ Trimming context from {len(expanded_docs)} to {max_chunks} chunks")
                expanded_docs = expanded_docs[:max_chunks]
                expanded_metas = expanded_metas[:max_chunks]
            
            return expanded_docs, expanded_metas
        
        return documents, metadatas
    
    def create_prompt(self, query: str, context_docs: list, metadatas: list):
        """Create prompt for Ollama with hierarchical context"""
        if not context_docs:
            context_text = "No specific context found."
        else:
            context_parts = []
            for i, (doc, meta) in enumerate(zip(context_docs, metadatas)):
                url = meta.get('url', '')
                chunk_id = meta.get('chunk_id', 'unknown')
                
                # Debug logging for student-related chunks
                if 'student' in doc.lower() or 'ph.d' in doc.lower():
                    logger.info(f"📚 Chunk {chunk_id} contains student info (Position {i+1}/{len(context_docs)})")
                    # Extra debug for chunk 79_137
                    if chunk_id == '79_137':
                        logger.info(f"🎯 CRITICAL CHUNK 79_137 content preview: {doc[:200]}...")
                
                # Simpler, cleaner context format
                context_parts.append(
                    f"[Source {i+1}] (from {url})\n{doc}\n"
                )
            context_text = "\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant for the Department of Computational and Data Sciences (CDS) at IISc Bangalore.

Answer the question using ONLY the information from the context sources below. The sources are ordered by relevance.

INSTRUCTIONS:
1. Read all sources carefully - the answer might span multiple sources
2. If asked about students/people, extract names that appear BEFORE their titles (e.g., "John Doe Ph.D. candidate" → extract "John Doe")
3. If multiple sources are from the same faculty page (same URL), they are related - connect the information
4. List all relevant names/items you find - don't stop at just one or two
5. If you cannot find the answer in the sources, say so clearly

CONTEXT SOURCES:
{context_text}

QUESTION: {query}

ANSWER (based only on the context sources above):"""
        
        return prompt
    
    def query_ollama(self, prompt: str):
        """Send query to Ollama"""
        logger.info(f"🤖 Querying Ollama ({self.model_name})...")
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()["response"]
                logger.info(f"✅ Response generated")
                return result.strip()
            else:
                raise HTTPException(status_code=502, detail=f"Ollama error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            raise HTTPException(status_code=504, detail="Ollama timeout")
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail="Cannot connect to Ollama")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_query(self, query: str):
        """Main RAG processing pipeline"""
        logger.info(f"📝 Processing: '{query}'")
        
        try:
            # 1. Search for context
            context_docs, metadatas = self.search_context(query)
            
            # 2. Create prompt
            prompt = self.create_prompt(query, context_docs, metadatas)
            
            # 3. Query Ollama
            answer = self.query_ollama(prompt)
            
            return RAGResponse(answer=answer, query=query)
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize service
logger.info("🚀 Starting Simple RAG Service...")
rag_service = SimpleRAGService()

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the simple web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CDS RAG System</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
            }
            
            .header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }
            
            .chat-box {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            
            .messages {
                min-height: 400px;
                max-height: 500px;
                overflow-y: auto;
                margin-bottom: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }
            
            .message {
                margin-bottom: 15px;
                padding: 12px 16px;
                border-radius: 10px;
                animation: fadeIn 0.3s;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .message.user {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                margin-left: 60px;
            }
            
            .message.assistant {
                background: white;
                border: 2px solid #e0e0e0;
                margin-right: 60px;
            }
            
            .message-label {
                font-size: 0.85em;
                font-weight: 600;
                margin-bottom: 5px;
                opacity: 0.8;
            }
            
            .input-area {
                display: flex;
                gap: 10px;
            }
            
            #questionInput {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                font-size: 1em;
                outline: none;
            }
            
            #questionInput:focus {
                border-color: #667eea;
            }
            
            #askButton {
                padding: 12px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s;
            }
            
            #askButton:hover:not(:disabled) {
                transform: translateY(-2px);
            }
            
            #askButton:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            
            .loading {
                display: inline-block;
                width: 16px;
                height: 16px;
                border: 2px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top-color: white;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            .examples {
                margin-top: 20px;
                padding: 20px;
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                color: white;
            }
            
            .examples h3 {
                margin-bottom: 10px;
                font-size: 1.1em;
            }
            
            .example-btn {
                display: inline-block;
                margin: 5px;
                padding: 8px 15px;
                background: rgba(255,255,255,0.2);
                border-radius: 20px;
                cursor: pointer;
                font-size: 0.9em;
                transition: all 0.2s;
            }
            
            .example-btn:hover {
                background: rgba(255,255,255,0.3);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎓 CDS RAG System</h1>
                <p>Ask questions about CDS, IISc Bangalore</p>
            </div>
            
            <div class="chat-box">
                <div class="messages" id="messages">
                    <div class="message assistant">
                        <div class="message-label">🤖 Assistant</div>
                        <div>Welcome! I can answer questions about CDS faculty, research, courses, and more. Try asking me something!</div>
                    </div>
                </div>
                
                <div class="input-area">
                    <input type="text" id="questionInput" placeholder="Type your question here..." 
                           onkeypress="if(event.key === 'Enter') askQuestion()">
                    <button id="askButton" onclick="askQuestion()">Ask</button>
                </div>
            </div>
            
            <div class="examples">
                <h3>💡 Try these examples:</h3>
                <span class="example-btn" onclick="setQuestion('Who are the faculty members in CDS?')">
                    Faculty members
                </span>
                <span class="example-btn" onclick="setQuestion('What research areas are covered in CDS?')">
                    Research areas
                </span>
                <span class="example-btn" onclick="setQuestion('Tell me about machine learning research')">
                    ML research
                </span>
                <span class="example-btn" onclick="setQuestion('What courses are offered?')">
                    Courses
                </span>
            </div>
        </div>
        
        <script>
            function setQuestion(text) {
                document.getElementById('questionInput').value = text;
            }
            
            async function askQuestion() {
                const input = document.getElementById('questionInput');
                const button = document.getElementById('askButton');
                const question = input.value.trim();
                
                if (!question) return;
                
                // Add user message
                addMessage('user', question);
                
                // Clear input and disable
                input.value = '';
                button.disabled = true;
                button.innerHTML = '<span class="loading"></span>';
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question }),
                        signal: AbortSignal.timeout(90000)
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }
                    
                    const data = await response.json();
                    addMessage('assistant', data.answer);
                    
                } catch (error) {
                    addMessage('assistant', '❌ Error: ' + error.message);
                }
                
                // Re-enable
                button.disabled = false;
                button.innerHTML = 'Ask';
            }
            
            function addMessage(role, content) {
                const messages = document.getElementById('messages');
                const div = document.createElement('div');
                div.className = `message ${role}`;
                div.innerHTML = `
                    <div class="message-label">${role === 'user' ? '👤 You' : '🤖 Assistant'}</div>
                    <div>${content.replace(/\\n/g, '<br>')}</div>
                `;
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/query", response_model=RAGResponse)
async def query_endpoint(request: QueryRequest):
    """Process a RAG query"""
    return await rag_service.process_query(request.question)

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "chunks": rag_service.collection.count(),
        "model": rag_service.model_name
    }

if __name__ == "__main__":
    logger.info("🌐 Starting server on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
