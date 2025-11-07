#!/usr/bin/env python3
"""
Simple FastAPI RAG Server for CDS Question Answering
Simple HTML frontend with Ollama integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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
    top_k: Optional[int] = 5
    include_sources: Optional[bool] = True
    use_hierarchy: Optional[bool] = True  # Use hierarchical context enhancement

class SourceInfo(BaseModel):
    chunk_id: str
    content: str
    chunk_type: str
    faculty: List[str]
    research_areas: List[str]
    departments: List[str]
    url: Optional[str]
    title: Optional[str]
    hierarchy_level: Optional[int]
    node_type: Optional[str]
    similarity: float

class RAGResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    processing_time: float
    query: str
    model_used: str
    total_chunks_searched: int

# Initialize FastAPI app
app = FastAPI(
    title="CDS ChromaDB RAG System",
    description="Intelligent Question Answering with Hierarchical Context for CDS, IISc Bangalore",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CDSChromaDBRAGService:
    def __init__(self, chroma_dir="./chroma_db"):
        """Initialize RAG service with ChromaDB"""
        logger.info("🚀 Initializing CDS ChromaDB RAG Service...")
        
        # ChromaDB configuration
        self.chroma_dir = chroma_dir
        self.collection_name = "cds_hierarchical_chunks"
        
        # Ollama configuration
        self.ollama_url = "http://localhost:11434"
        self.model_name = "qwen2.5:7b"  # Primary model
        self.fallback_model = "llama3.2:3b"  # Fallback if primary not available
        
        # Load embedding model (same as used in embedder)
        logger.info("📥 Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client
        logger.info(f"📂 Connecting to ChromaDB at: {chroma_dir}")
        self.client = chromadb.PersistentClient(path=chroma_dir)
        
        # Get collection
        self.collection = self.client.get_collection(name=self.collection_name)
        
        # Test connections
        self._test_connections()
        logger.info("✅ CDS ChromaDB RAG Service initialized successfully!")
    
    def _test_connections(self):
        """Test ChromaDB and Ollama connections"""
        try:
            # Test ChromaDB
            count = self.collection.count()
            logger.info(f"✅ ChromaDB connected: {count} embeddings available")
            
            if count == 0:
                raise Exception("ChromaDB collection is empty!")
                
        except Exception as e:
            logger.error(f"❌ ChromaDB connection failed: {e}")
            raise
        
        try:
            # Test Ollama
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                logger.info(f"✅ Ollama connected. Available models: {model_names}")
                
                # Check primary model
                if self.model_name not in model_names:
                    logger.warning(f"⚠️ Primary model '{self.model_name}' not found.")
                    if self.fallback_model in model_names:
                        logger.info(f"✅ Using fallback model: {self.fallback_model}")
                        self.model_name = self.fallback_model
                    elif model_names:
                        logger.warning(f"Using first available: {model_names[0]}")
                        self.model_name = model_names[0]
                    else:
                        raise Exception("No Ollama models available!")
            else:
                raise Exception("Ollama not responding")
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {e}")
            logger.info(f"💡 Run: ollama pull {self.model_name}")
            raise
    
    def extract_all_people(self, faculty_json, departments_json, chunk_text, chunk_type):
        """Extract all person names from metadata and chunk text"""
        people = {
            'faculty': [],
            'students': [],
            'researchers': [],
            'other': []
        }
        
        # Parse JSON strings from ChromaDB metadata
        faculty_list = json.loads(faculty_json) if faculty_json else []
        dept_list = json.loads(departments_json) if departments_json else []
        
        # Enhanced filtering for all types of people
        non_person_terms = [
            'cds', 'about', 'forms', 'programs', 'schedule', 'admissions', 
            'mtech', 'btech', 'acm', 'usenix', 'award', 'medal', 'models', 
            'bangalore', 'home', 'department', 'science', 'institute', 'faq',
            'opportunities', 'contact', 'news', 'events', 'courses', 'png',
            'template', 'iisc', 'wide', 'square', 'image', 'logo', 'icon',
            'page', 'menu', 'navigation', 'header', 'footer', 'link', 'button',
            'search', 'login', 'register', 'download', 'upload', 'file', 'www',
            'http', 'html', 'php', 'css', 'javascript', 'pdf', 'doc', 'computational'
        ]
        
        # Student indicators in text
        student_indicators = [
            'phd student', 'phd scholar', 'doctoral student', 'graduate student',
            'mtech student', 'm.tech student', 'master student', 'masters student',
            'research scholar', 'student researcher', 'pursuing phd', 'pursuing mtech'
        ]
        
        # Faculty indicators in text
        faculty_indicators = [
            'professor', 'prof', 'dr.', 'faculty', 'principal investigator', 'pi',
            'assistant professor', 'associate professor', 'head of department', 'hod',
            'chair professor', 'emeritus'
        ]
        
        def is_person_name(name):
            """Check if a string looks like a person's name"""
            if not name or not isinstance(name, str):
                return False
                
            name_clean = name.strip()
            if len(name_clean) < 3:
                return False
            
            # Remove common titles
            name_clean = name_clean.replace('Dr.', '').replace('Prof.', '').strip()
            
            # Skip if contains non-person terms
            name_lower = name_clean.lower()
            if any(term in name_lower for term in non_person_terms):
                return False
            
            # Skip if all uppercase (likely acronyms)
            if name_clean.isupper() and len(name_clean) > 3:
                return False
            
            # Skip if contains numbers or web-related content
            if (any(char.isdigit() for char in name_clean) or 
                any(ext in name_lower for ext in ['.html', '.php', 'www.', 'http'])):
                return False
            
            # Must contain only letters, spaces, hyphens, dots
            if not all(c.isalpha() or c.isspace() or c in '-.' for c in name_clean):
                return False
            
            # Should have at least 2 words for a proper name
            words = name_clean.split()
            if len(words) < 2:
                return False
            
            # Each word should be reasonable length
            if any(len(word) < 2 for word in words):
                return False
            
            return True
        
        def classify_person(name, context_text, chunk_type):
            """Classify if person is faculty, student, or researcher"""
            context_lower = context_text.lower()
            name_lower = name.lower()
            
            # Check for student indicators around the name
            for indicator in student_indicators:
                if indicator in context_lower and name_lower in context_lower:
                    return 'students'
            
            # Check for faculty indicators
            for indicator in faculty_indicators:
                if indicator in context_lower and name_lower in context_lower:
                    return 'faculty'
            
            # Check chunk type for hints
            if chunk_type:
                if 'student' in chunk_type.lower():
                    return 'students'
                elif 'faculty' in chunk_type.lower():
                    return 'faculty'
                elif 'research' in chunk_type.lower():
                    return 'researchers'
            
            # Default classification based on context patterns
            if any(word in context_lower for word in ['thesis', 'dissertation', 'advisor', 'supervised']):
                return 'students'
            elif any(word in context_lower for word in ['professor', 'faculty', 'teaching']):
                return 'faculty'
            else:
                return 'researchers'
        
        # Process faculty list from metadata
        for name in faculty_list:
            if is_person_name(name):
                category = classify_person(name, chunk_text, chunk_type)
                people[category].append(name.strip())
        
        # Extract additional names from chunk text using NLP patterns
        name_patterns = [
            r'(?:Dr\.?\s+|Prof\.?\s+|Mr\.?\s+|Ms\.?\s+|Mrs\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+(?:is|was|has|received|published|works|studies))',
            r'(?:supervised by|advised by|worked with|collaborated with)\s+(?:Dr\.?\s+|Prof\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'(?:PhD student|M\.?Tech student|research scholar)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, chunk_text)
            for match in matches:
                if is_person_name(match) and match not in [item for sublist in people.values() for item in sublist]:
                    category = classify_person(match, chunk_text, chunk_type)
                    people[category].append(match.strip())
        
        # Remove duplicates and clean up
        for category in people:
            people[category] = list(set(people[category]))
            people[category].sort()
        
        return people, dept_list

    def get_hierarchical_context(self, chunk_metadata):
        """Fetch parent and sibling chunks for hierarchical context"""
        parent_id = chunk_metadata.get('parent_id', '')
        children_ids = chunk_metadata.get('children_ids', '[]')
        
        hierarchical_chunks = []
        
        # Try to get parent chunk for broader context
        if parent_id and parent_id != '':
            try:
                # Find chunks from the same parent page
                parent_results = self.collection.get(
                    where={"doc_id": parent_id},
                    limit=3
                )
                if parent_results and parent_results['documents']:
                    hierarchical_chunks.extend(parent_results['documents'])
            except Exception as e:
                logger.debug(f"Could not fetch parent context: {e}")
        
        return hierarchical_chunks

    def search_chromadb(self, query: str, top_k: int = 5, filters: dict = None):
        """Search ChromaDB with optional metadata filters"""
        logger.info(f"🔍 Searching ChromaDB for: '{query[:50]}...'")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Build query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k * 2  # Get more for filtering
            }
            
            # Add metadata filters if provided
            if filters:
                query_params["where"] = filters
            
            # Query ChromaDB
            results = self.collection.query(**query_params)
            
            logger.info(f"✅ Found {len(results['ids'][0])} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching ChromaDB: {e}")
            raise HTTPException(status_code=500, detail=f"ChromaDB search error: {str(e)}")

    def get_relevant_context_advanced(self, query: str, top_k: int = 5, use_hierarchy: bool = True):
        """Advanced context retrieval with query-type awareness and hierarchy"""
        logger.info(f"🔍 Advanced search for: '{query[:50]}...'")
        
        query_lower = query.lower()
        filters = None
        
        # Determine query type and apply smart filtering
        if any(term in query_lower for term in ['faculty', 'professor', 'prof', 'instructor', 'teacher']):
            # Faculty-focused queries
            logger.info("📚 Faculty-focused query detected")
            filters = {"node_type": "faculty_page"}
            
        elif any(term in query_lower for term in ['student', 'phd', 'mtech', 'm.tech', 'scholar']):
            # Student-focused queries
            logger.info("🎓 Student-focused query detected")
            filters = {"has_faculty_info": True}  # Students often mentioned in faculty contexts
            
        elif any(term in query_lower for term in ['lab', 'laboratory', 'group', 'research group']):
            # Lab-focused queries
            logger.info("🔬 Lab-focused query detected")
            # No specific filter, but we'll boost lab-related content
            
        elif any(term in query_lower for term in ['course', 'curriculum', 'syllabus', 'teaching']):
            # Course-focused queries
            logger.info("📖 Course-focused query detected")
            filters = {"chunk_type": "course_info"}
        
        # Search ChromaDB
        results = self.search_chromadb(query, top_k, filters)
        
        # If hierarchy is enabled, enhance with parent/sibling context
        enhanced_results = []
        seen_chunks = set()
        
        for i in range(len(results['ids'][0])):
            chunk_id = results['ids'][0][i]
            document = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # Primary chunk
            if chunk_id not in seen_chunks:
                enhanced_results.append({
                    'chunk_id': chunk_id,
                    'document': document,
                    'metadata': metadata,
                    'distance': distance,
                    'is_primary': True
                })
                seen_chunks.add(chunk_id)
            
            # Add hierarchical context if enabled
            if use_hierarchy and len(enhanced_results) < top_k:
                hierarchical_chunks = self.get_hierarchical_context(metadata)
                for h_chunk in hierarchical_chunks[:2]:  # Limit hierarchical additions
                    h_chunk_id = f"{metadata.get('doc_id', '')}_{len(enhanced_results)}"
                    if h_chunk_id not in seen_chunks:
                        enhanced_results.append({
                            'chunk_id': h_chunk_id,
                            'document': h_chunk,
                            'metadata': metadata,
                            'distance': distance + 0.1,  # Slight penalty for hierarchical
                            'is_primary': False
                        })
                        seen_chunks.add(h_chunk_id)
        
        return enhanced_results[:top_k]

    def create_advanced_rag_prompt(self, query: str, context_results):
        """Create advanced RAG prompt with comprehensive extraction"""
        if not context_results:
            return f"Answer the following question about CDS (Department of Computational and Data Sciences, IISc Bangalore): {query}"
        
        context_text = ""
        all_people = {
            'faculty': set(),
            'students': set(), 
            'researchers': set(),
            'other': set()
        }
        all_research = set()
        all_departments = set()
        all_labs = set()
        
        for i, result in enumerate(context_results, 1):
            chunk_id = result['chunk_id']
            text = result['document']
            metadata = result['metadata']
            distance = result['distance']
            is_primary = result.get('is_primary', True)
            
            similarity = 1 - distance  # Convert distance to similarity
            
            context_text += f"\n--- Context {i} ({'PRIMARY' if is_primary else 'RELATED'}, Relevance: {similarity:.3f}) ---\n"
            
            # Add metadata info
            chunk_type = metadata.get('chunk_type', '')
            if chunk_type:
                context_text += f"Type: {chunk_type}\n"
            
            hierarchy_level = metadata.get('hierarchy_level', -1)
            node_type = metadata.get('node_type', '')
            if hierarchy_level >= 0:
                context_text += f"Hierarchy: Level {hierarchy_level}, Type: {node_type}\n"
            
            # Extract all people from this chunk
            faculty_json = metadata.get('faculty_names', '[]')
            departments_json = metadata.get('departments', '[]')
            people_found, dept_list = self.extract_all_people(
                faculty_json, departments_json, text, chunk_type
            )
            
            for category, names in people_found.items():
                if names:
                    all_people[category].update(names)
                    context_text += f"{category.title()}: {', '.join(names)}\n"
            
            # Research areas
            research_json = metadata.get('research_areas', '[]')
            research_list = json.loads(research_json) if research_json else []
            if research_list:
                all_research.update(research_list)
                context_text += f"Research Areas: {', '.join(research_list[:5])}\n"
            
            # Departments
            if dept_list:
                all_departments.update(dept_list)
                context_text += f"Departments: {', '.join(dept_list)}\n"
            
            # Extract lab names
            lab_patterns = [
                r'([A-Z][A-Za-z]+\s+Lab(?:oratory)?)',
                r'([A-Z][A-Z]+\s+Lab)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+\s+Lab)'
            ]
            for pattern in lab_patterns:
                labs = re.findall(pattern, text)
                all_labs.update(labs)
            
            if all_labs:
                context_text += f"Labs Mentioned: {', '.join(list(all_labs)[:3])}\n"
            
            # URL and title
            url = metadata.get('url', '')
            title = metadata.get('title', '')
            if title:
                context_text += f"Page Title: {title[:60]}...\n"
            if url:
                context_text += f"Source: {url}\n"
            
            # Content (truncated)
            content = text.strip()
            if len(content) > 400:
                content = content[:400] + "..."
            context_text += f"Content: {content}\n"
        
        # Create comprehensive summary
        summary_sections = []
        
        for category, people in all_people.items():
            if people:
                clean_people = [p for p in sorted(people) if len(p.split()) >= 2]
                if clean_people:
                    summary_sections.append(f"{category.upper()}: {', '.join(clean_people)}")
        
        if all_research:
            research_list = sorted(list(all_research))[:15]
            summary_sections.append(f"RESEARCH AREAS: {', '.join(research_list)}")
        
        if all_departments:
            summary_sections.append(f"DEPARTMENTS: {', '.join(sorted(list(all_departments)))}")
        
        if all_labs:
            summary_sections.append(f"LABS: {', '.join(sorted(list(all_labs)))}")
        
        if summary_sections:
            context_text += f"\n--- COMPREHENSIVE SUMMARY ---\n" + "\n".join(summary_sections) + "\n"
        
        prompt = f"""You are a knowledgeable assistant for the Department of Computational and Data Sciences (CDS) at the Indian Institute of Science (IISc), Bangalore. 

Use the provided hierarchical context to answer questions about people, research areas, labs, courses, and academic information.

CONTEXT FROM CDS DATABASE (with hierarchy preserved):
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Use the COMPREHENSIVE SUMMARY section to identify relevant people (faculty, students, researchers)
2. For research area queries, list people working in those areas from the summary
3. For lab queries, mention lab members, research focus, and hierarchy
4. For people queries, categorize them appropriately (faculty/students/researchers) with their roles
5. Include research areas, departments, and affiliations when available
6. If asking about specific topics (NLP, ML, AI, etc.), list all relevant people and labs
7. Be specific about roles and provide structured information
8. Use hierarchy information to provide context (e.g., "under the faculty page of...")
9. If information is limited, acknowledge it and suggest checking the official CDS website
10. Keep response organized, comprehensive, and well-formatted

ANSWER:"""
        
        return prompt
    
    def query_ollama(self, prompt: str):
        """Send query to Ollama with streaming disabled"""
        logger.info(f"🤖 Querying Ollama with model: {self.model_name}")
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 1500
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()["response"]
                logger.info(f"✅ Generated response ({len(result)} chars)")
                return result
            else:
                error_msg = f"Ollama error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise HTTPException(status_code=502, detail=error_msg)
                
        except requests.exceptions.Timeout:
            logger.error("❌ Ollama request timeout")
            raise HTTPException(status_code=504, detail="Ollama request timeout (model may need to be pulled)")
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to Ollama")
            raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Is it running?")
        except Exception as e:
            logger.error(f"❌ Ollama error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_query(self, query: str, top_k: int = 5, include_sources: bool = True, use_hierarchy: bool = True):
        """Main RAG processing pipeline with ChromaDB and hierarchy"""
        start_time = time.time()
        logger.info(f"📝 Processing query: '{query}'")
        
        try:
            # Get relevant context with advanced search
            context_results = self.get_relevant_context_advanced(query, top_k, use_hierarchy)
            
            # Create RAG prompt
            rag_prompt = self.create_advanced_rag_prompt(query, context_results)
            
            # Query Ollama
            response = self.query_ollama(rag_prompt)
            
            # Prepare sources
            sources = []
            if include_sources:
                for result in context_results:
                    if not result.get('is_primary', True):
                        continue  # Skip hierarchical additions in sources
                    
                    metadata = result['metadata']
                    text = result['document']
                    distance = result['distance']
                    
                    # Extract people info
                    faculty_json = metadata.get('faculty_names', '[]')
                    departments_json = metadata.get('departments', '[]')
                    people_info, dept_list = self.extract_all_people(
                        faculty_json, departments_json, text, metadata.get('chunk_type', '')
                    )
                    
                    all_people = []
                    for category, names in people_info.items():
                        all_people.extend(names)
                    
                    # Research areas
                    research_json = metadata.get('research_areas', '[]')
                    research_list = json.loads(research_json) if research_json else []
                    
                    # Truncate content for display
                    content = text.strip()
                    if len(content) > 300:
                        content = content[:300] + "..."
                    
                    sources.append(SourceInfo(
                        chunk_id=result['chunk_id'],
                        content=content,
                        chunk_type=metadata.get('chunk_type', 'general'),
                        faculty=all_people,
                        research_areas=research_list,
                        departments=dept_list,
                        url=metadata.get('url', ''),
                        title=metadata.get('title', ''),
                        hierarchy_level=metadata.get('hierarchy_level', -1),
                        node_type=metadata.get('node_type', ''),
                        similarity=round(1 - distance, 3)
                    ))
            
            processing_time = round(time.time() - start_time, 2)
            logger.info(f"✅ Query processed in {processing_time}s")
            
            return RAGResponse(
                answer=response.strip(),
                sources=sources,
                processing_time=processing_time,
                query=query,
                model_used=self.model_name,
                total_chunks_searched=self.collection.count()
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize RAG service
logger.info("🚀 Starting CDS ChromaDB RAG Service...")
rag_service = CDSChromaDBRAGService()

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the web interface"""
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
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1000px;
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
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .badge {
                display: inline-block;
                background: rgba(255,255,255,0.2);
                padding: 5px 15px;
                border-radius: 20px;
                margin: 5px;
                font-size: 0.9em;
            }
            
            .chat-container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            
            .chat-messages {
                height: 500px;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9fa;
            }
            
            .message {
                margin-bottom: 20px;
                animation: slideIn 0.3s ease-out;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .message.user {
                text-align: right;
            }
            
            .message-bubble {
                display: inline-block;
                max-width: 80%;
                padding: 15px 20px;
                border-radius: 20px;
                text-align: left;
            }
            
            .message.user .message-bubble {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .message.assistant .message-bubble {
                background: white;
                border: 2px solid #e0e0e0;
                color: #333;
            }
            
            .message-label {
                font-size: 0.8em;
                color: #666;
                margin-bottom: 5px;
                font-weight: 600;
            }
            
            .sources {
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #e0e0e0;
            }
            
            .source-item {
                background: #f8f9fa;
                padding: 10px;
                border-radius: 10px;
                margin: 8px 0;
                font-size: 0.9em;
                border-left: 3px solid #667eea;
            }
            
            .source-title {
                font-weight: 600;
                color: #667eea;
                margin-bottom: 5px;
            }
            
            .source-meta {
                font-size: 0.85em;
                color: #666;
                margin-top: 5px;
            }
            
            .input-container {
                display: flex;
                gap: 10px;
                padding: 20px;
                background: white;
                border-top: 2px solid #e0e0e0;
            }
            
            #questionInput {
                flex: 1;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                font-size: 1em;
                outline: none;
                transition: all 0.3s;
            }
            
            #questionInput:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            #askButton {
                padding: 15px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            
            #askButton:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            
            #askButton:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top-color: white;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            .stats {
                font-size: 0.85em;
                color: #666;
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px solid #e0e0e0;
            }
            
            .example-queries {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 15px;
                margin-top: 20px;
                color: white;
            }
            
            .example-queries h3 {
                margin-bottom: 15px;
            }
            
            .example-query {
                background: rgba(255,255,255,0.2);
                padding: 10px 15px;
                border-radius: 10px;
                margin: 8px 0;
                cursor: pointer;
                transition: all 0.3s;
            }
            
            .example-query:hover {
                background: rgba(255,255,255,0.3);
                transform: translateX(5px);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎓 CDS ChromaDB RAG System</h1>
                <p>Intelligent Question Answering for CDS, IISc Bangalore</p>
                <div>
                    <span class="badge">🔍 Semantic Search</span>
                    <span class="badge">🌳 Hierarchical Context</span>
                    <span class="badge">🤖 LLM-Powered</span>
                    <span class="badge">⚡ ChromaDB</span>
                </div>
            </div>
            
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant">
                        <div class="message-label">🤖 Assistant</div>
                        <div class="message-bubble">
                            <p><strong>Welcome to the CDS RAG System!</strong></p>
                            <p>I can answer questions about:</p>
                            <ul style="margin: 10px 0; padding-left: 20px;">
                                <li>Faculty members and their research</li>
                                <li>Research labs and groups</li>
                                <li>PhD students and scholars</li>
                                <li>Courses and curriculum</li>
                                <li>Research areas and projects</li>
                            </ul>
                            <p>Try asking me anything about CDS!</p>
                        </div>
                    </div>
                </div>
                
                <div class="input-container">
                    <input type="text" id="questionInput" placeholder="Ask a question about CDS..." 
                           onkeypress="if(event.key === 'Enter') askQuestion()">
                    <button id="askButton" onclick="askQuestion()">Ask</button>
                </div>
            </div>
            
            <div class="example-queries">
                <h3>💡 Example Questions:</h3>
                <div class="example-query" onclick="setQuestion(this.textContent)">
                    Which faculty members work on machine learning?
                </div>
                <div class="example-query" onclick="setQuestion(this.textContent)">
                    Tell me about research labs in CDS
                </div>
                <div class="example-query" onclick="setQuestion(this.textContent)">
                    What courses are offered in the MTech program?
                </div>
                <div class="example-query" onclick="setQuestion(this.textContent)">
                    Who works on natural language processing?
                </div>
            </div>
        </div>
        
        <script>
            function setQuestion(text) {
                document.getElementById('questionInput').value = text;
                document.getElementById('questionInput').focus();
            }
            
            async function askQuestion() {
                const input = document.getElementById('questionInput');
                const button = document.getElementById('askButton');
                const messages = document.getElementById('chatMessages');
                const question = input.value.trim();
                
                if (!question) {
                    console.log('Empty question');
                    return;
                }
                
                console.log('Asking question:', question);
                
                // Add user message
                addMessage('user', question);
                
                // Clear input and disable button
                input.value = '';
                button.disabled = true;
                button.innerHTML = '<span class="loading"></span>';
                
                // Add "thinking" message
                addMessage('assistant', '🤔 Thinking... (This may take 30-60 seconds)', null, null, null);
                
                try {
                    console.log('Sending request to /query');
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            question: question,
                            top_k: 5,
                            include_sources: true,
                            use_hierarchy: true
                        }),
                        signal: AbortSignal.timeout(180000)  // 3 minute timeout
                    });
                    
                    console.log('Response status:', response.status);
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        console.error('Error response:', errorText);
                        throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
                    }
                    
                    const data = await response.json();
                    console.log('Response data:', data);
                    
                    // Remove "thinking" message
                    const thinkingMsg = messages.lastElementChild;
                    if (thinkingMsg && thinkingMsg.textContent.includes('Thinking')) {
                        messages.removeChild(thinkingMsg);
                    }
                    
                    // Add assistant message with sources
                    addMessage('assistant', data.answer, data.sources, data.processing_time, data.model_used);
                    
                } catch (error) {
                    console.error('Catch error:', error);
                    
                    // Remove "thinking" message
                    const thinkingMsg = messages.lastElementChild;
                    if (thinkingMsg && thinkingMsg.textContent.includes('Thinking')) {
                        messages.removeChild(thinkingMsg);
                    }
                    
                    if (error.name === 'TimeoutError' || error.message.includes('timeout')) {
                        addMessage('assistant', '⏱️ Request timed out. The AI model took too long to respond. Please try a simpler question or try again later.');
                    } else {
                        addMessage('assistant', '❌ Error: ' + error.message + '\n\nThe server might be processing your request. Please wait a moment and check the response below.');
                    }
                }
                
                // Re-enable button
                button.disabled = false;
                button.innerHTML = 'Ask';
            }
            
            function addMessage(role, content, sources = null, time = null, model = null) {
                const messages = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role}`;
                
                let html = `
                    <div class="message-label">${role === 'user' ? '👤 You' : '🤖 Assistant'}</div>
                    <div class="message-bubble">
                        <p>${content.replace(/\n/g, '<br>')}</p>
                `;
                
                if (sources && sources.length > 0) {
                    html += '<div class="sources"><strong>📚 Sources:</strong>';
                    sources.forEach((source, idx) => {
                        html += `
                            <div class="source-item">
                                <div class="source-title">Source ${idx + 1}: ${source.title || 'CDS Content'}</div>
                                <div>${source.content.substring(0, 150)}...</div>
                                <div class="source-meta">
                                    Type: ${source.chunk_type} | 
                                    Similarity: ${(source.similarity * 100).toFixed(1)}% | 
                                    Level: ${source.hierarchy_level} | 
                                    ${source.url ? '<a href="' + source.url + '" target="_blank">View Source</a>' : ''}
                                </div>
                            </div>
                        `;
                    });
                    html += '</div>';
                }
                
                if (time && model) {
                    html += `<div class="stats">⏱️ ${time}s | 🤖 ${model}</div>`;
                }
                
                html += '</div>';
                messageDiv.innerHTML = html;
                messages.appendChild(messageDiv);
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
    return await rag_service.process_query(
        query=request.question,
        top_k=request.top_k,
        include_sources=request.include_sources,
        use_hierarchy=request.use_hierarchy
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chromadb_connected": True,
        "total_chunks": rag_service.collection.count(),
        "model": rag_service.model_name,
        "ollama_url": rag_service.ollama_url
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "total_chunks": rag_service.collection.count(),
        "collection_name": rag_service.collection_name,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "llm_model": rag_service.model_name,
        "chroma_directory": rag_service.chroma_dir
    }

if __name__ == "__main__":
    logger.info("🌐 Starting FastAPI server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
