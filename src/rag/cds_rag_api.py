#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
FastAPI RAG Server for CDS Question Answering
Advanced system with comprehensive people search
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import requests
import psycopg2
from sentence_transformers import SentenceTransformer
import uvicorn
import time
import json
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    include_sources: Optional[bool] = True

class SourceInfo(BaseModel):
    chunk_id: str
    content: str
    chunk_type: str
    faculty: List[str]
    research_areas: List[str]
    url: Optional[str]
    similarity: float

class RAGResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    processing_time: float
    query: str
    model_used: str

# Initialize FastAPI app
app = FastAPI(
    title="CDS Advanced RAG System",
    description="Intelligent Question Answering with Comprehensive People Search for CDS, IISc Bangalore",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

class CDSRAGService:
    def __init__(self):
        """Initialize RAG service"""
        logger.info("🚀 Initializing Advanced CDS RAG Service...")
        
        # Database configuration
        self.db_config = {
            'host': 'localhost',
            'database': 'cds_rag_db',
            'user': 'rag_user',
            'password': 'secure_rag_password_123',
            'port': 5432
        }
        
        # Ollama configuration
        self.ollama_url = "http://localhost:11434"
        self.model_name = "phi3:mini"
        
        # Load embedding model
        logger.info("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test connections
        self._test_connections()
        logger.info("✅ Advanced CDS RAG Service initialized successfully!")
    
    def _test_connections(self):
        """Test database and Ollama connections"""
        try:
            # Test database
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM cds_embeddings;")
            count = cur.fetchone()[0]
            logger.info(f"✅ Database connected: {count} embeddings available")
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
        
        try:
            # Test Ollama
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                logger.info(f"✅ Ollama connected. Available models: {model_names}")
                
                if self.model_name not in model_names:
                    logger.warning(f"⚠️ Model '{self.model_name}' not found. Using first available.")
                    if model_names:
                        self.model_name = model_names[0]
            else:
                raise Exception("Ollama not responding")
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {e}")
            raise
    
    def extract_all_people(self, faculty_list, chunk_text, chunk_type):
        """Extract all person names from faculty list and chunk text"""
        people = {
            'faculty': [],
            'students': [],
            'researchers': [],
            'unknown': []
        }
        
        # Enhanced filtering for all types of people
        non_person_terms = [
            'cds', 'about', 'forms', 'programs', 'schedule', 'admissions', 
            'mtech', 'btech', 'acm', 'usenix', 'award', 'medal', 'models', 
            'bangalore', 'home', 'department', 'science', 'institute', 'faq',
            'opportunities', 'contact', 'news', 'events', 'courses', 'png',
            'template', 'iisc', 'wide', 'square', 'image', 'logo', 'icon',
            'page', 'menu', 'navigation', 'header', 'footer', 'link', 'button',
            'search', 'login', 'register', 'download', 'upload', 'file', 'www',
            'http', 'html', 'php', 'css', 'javascript', 'pdf', 'doc'
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
            'assistant professor', 'associate professor', 'head of department', 'hod'
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
            name_context = f"{name.lower()} {context_text.lower()}"
            
            # Check for student indicators around the name
            for indicator in student_indicators:
                if indicator in context_lower:
                    # Check if name appears near student indicators
                    if name.lower() in context_lower:
                        return 'students'
            
            # Check for faculty indicators
            for indicator in faculty_indicators:
                if indicator in context_lower:
                    if name.lower() in context_lower:
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
                return 'researchers'  # Default for research contexts
        
        # Process faculty list
        if faculty_list:
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
        
        return people

    def get_relevant_context_advanced(self, query: str, top_k: int = 5):
        """Advanced context retrieval for people and research area queries"""
        logger.info(f"🔍 Advanced search for: '{query[:50]}...'")
        
        try:
            query_embedding = self.embedding_model.encode(query)
            query_vector = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
            
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Determine query type
            query_lower = query.lower()
            
            # People-focused queries
            if any(term in query_lower for term in ['faculty', 'professor', 'student', 'researcher', 'people', 'who', 'names']):
                # Search for people with priority on faculty/student chunks
                cur.execute("""
                    SELECT chunk_id, chunk_text, chunk_type, faculty_names, research_areas, url,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM cds_embeddings 
                    WHERE (
                        faculty_names IS NOT NULL 
                        AND array_length(faculty_names, 1) > 0
                    )
                    OR chunk_text ILIKE '%%faculty%%' 
                    OR chunk_text ILIKE '%%professor%%'
                    OR chunk_text ILIKE '%%student%%'
                    OR chunk_text ILIKE '%%phd%%'
                    OR chunk_text ILIKE '%%mtech%%'
                    OR chunk_text ILIKE '%%researcher%%'
                    ORDER BY 
                        CASE 
                            WHEN faculty_names IS NOT NULL THEN 0 
                            WHEN chunk_text ILIKE '%%student%%' THEN 1
                            ELSE 2 
                        END,
                        embedding <=> %s::vector
                    LIMIT %s;
                """, (query_vector, query_vector, top_k * 2))
            
            # Research area queries
            elif any(term in query_lower for term in ['research', 'working on', 'area', 'field', 'nlp', 'ml', 'ai', 'data science']):
                # Search with focus on research areas and matching content
                cur.execute("""
                    SELECT chunk_id, chunk_text, chunk_type, faculty_names, research_areas, url,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM cds_embeddings 
                    WHERE (
                        research_areas IS NOT NULL 
                        AND array_length(research_areas, 1) > 0
                    )
                    OR chunk_text ILIKE '%%research%%'
                    OR chunk_text ILIKE '%%nlp%%'
                    OR chunk_text ILIKE '%%machine learning%%'
                    OR chunk_text ILIKE '%%artificial intelligence%%'
                    ORDER BY 
                        CASE WHEN research_areas IS NOT NULL THEN 0 ELSE 1 END,
                        embedding <=> %s::vector
                    LIMIT %s;
                """, (query_vector, query_vector, top_k * 2))
            
            # Lab-specific queries
            elif any(term in query_lower for term in ['lab', 'laboratory', 'group', 'team']):
                cur.execute("""
                    SELECT chunk_id, chunk_text, chunk_type, faculty_names, research_areas, url,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM cds_embeddings 
                    WHERE chunk_text ILIKE '%%lab%%' 
                    OR chunk_text ILIKE '%%laboratory%%'
                    OR chunk_text ILIKE '%%group%%'
                    OR chunk_text ILIKE '%%team%%'
                    OR chunk_type ILIKE '%%lab%%'
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (query_vector, query_vector, top_k * 2))
            
            else:
                # Standard similarity search
                cur.execute("""
                    SELECT chunk_id, chunk_text, chunk_type, faculty_names, research_areas, url,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM cds_embeddings 
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (query_vector, query_vector, top_k))
            
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            logger.info(f"✅ Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in advanced search: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    def create_advanced_rag_prompt(self, query: str, context_results):
        """Create advanced RAG prompt with comprehensive people extraction"""
        if not context_results:
            return f"Answer the following question about CDS (Department of Computational and Data Sciences, IISc Bangalore): {query}"
        
        context_text = ""
        all_people = {
            'faculty': set(),
            'students': set(), 
            'researchers': set(),
            'unknown': set()
        }
        all_research = set()
        all_labs = set()
        
        for i, result in enumerate(context_results, 1):
            chunk_id, text, chunk_type, faculty, research, url, similarity = result
            
            context_text += f"\n--- Context {i} (Relevance: {similarity:.3f}) ---\n"
            if chunk_type:
                context_text += f"Type: {chunk_type}\n"
            
            # Extract all people from this chunk
            people_found = self.extract_all_people(faculty, text, chunk_type)
            
            for category, names in people_found.items():
                if names:
                    all_people[category].update(names)
                    context_text += f"{category.title()}: {', '.join(names)}\n"
            
            if research:
                all_research.update(research)
                context_text += f"Research Areas: {', '.join(research[:5])}\n"
            
            # Extract lab names
            lab_patterns = [r'([A-Z][A-Za-z]*\s+Lab)', r'([A-Z][A-Za-z]*\s+Laboratory)', r'([A-Z][A-Z]+\s+Lab)']
            for pattern in lab_patterns:
                labs = re.findall(pattern, text)
                all_labs.update(labs)
            
            if all_labs:
                context_text += f"Labs Mentioned: {', '.join(list(all_labs)[:3])}\n"
            
            if url:
                context_text += f"Source: {url}\n"
            
            content = text.strip()
            if len(content) > 300:
                content = content[:300] + "..."
            context_text += f"Content: {content}\n"
        
        # Create comprehensive summary
        summary_sections = []
        
        for category, people in all_people.items():
            if people:
                clean_people = [p for p in sorted(people) if len(p.split()) >= 2]
                if clean_people:
                    summary_sections.append(f"{category.upper()}: {', '.join(clean_people)}")
        
        if all_research:
            research_list = sorted(list(all_research))[:10]  # Top 10 research areas
            summary_sections.append(f"RESEARCH AREAS: {', '.join(research_list)}")
        
        if all_labs:
            summary_sections.append(f"LABS: {', '.join(sorted(list(all_labs)))}")
        
        if summary_sections:
            context_text += f"\n--- COMPREHENSIVE SUMMARY ---\n" + "\n".join(summary_sections) + "\n"
        
        prompt = f"""You are a knowledgeable assistant for the Department of Computational and Data Sciences (CDS) at the Indian Institute of Science (IISc), Bangalore. 

Use the provided context to answer questions about people, research areas, labs, and academic information.

CONTEXT FROM CDS DATABASE:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Use the COMPREHENSIVE SUMMARY section to identify relevant people (faculty, students, researchers)
2. For research area queries, list people working in those areas from the summary
3. For lab queries, mention lab members and research focus
4. For people queries, categorize them appropriately (faculty/students/researchers)
5. Include research areas and affiliations when available
6. If asking about specific research areas (like NLP, ML, AI), list all relevant people
7. Be specific about roles: "Faculty: [names]", "PhD Students: [names]", etc.
8. If information is limited, suggest checking the official CDS website
9. Keep response organized and comprehensive

ANSWER:"""
        
        return prompt
    
    def query_ollama(self, prompt: str):
        """Send query to Ollama"""
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
                        "num_predict": 1000
                    }
                },
                timeout=600
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
            raise HTTPException(status_code=504, detail="Ollama request timeout")
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to Ollama")
            raise HTTPException(status_code=503, detail="Cannot connect to Ollama")
        except Exception as e:
            logger.error(f"❌ Ollama error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_query(self, query: str, top_k: int = 5, include_sources: bool = True):
        """Main RAG processing pipeline with advanced people search"""
        start_time = time.time()
        logger.info(f"📝 Processing advanced query: '{query}'")
        
        try:
            # Use advanced context retrieval
            context_results = self.get_relevant_context_advanced(query, top_k)
            
            # Use advanced prompt creation
            rag_prompt = self.create_advanced_rag_prompt(query, context_results)
            
            response = self.query_ollama(rag_prompt)
            
            sources = []
            if include_sources:
                for result in context_results:
                    chunk_id, text, chunk_type, faculty, research, url, similarity = result
                    
                    # Extract comprehensive people info for sources
                    people_info = self.extract_all_people(faculty, text, chunk_type)
                    all_people = []
                    for category, names in people_info.items():
                        all_people.extend(names)
                    
                    content = text.strip()
                    if len(content) > 300:
                        content = content[:300] + "..."
                    
                    sources.append(SourceInfo(
                        chunk_id=chunk_id,
                        content=content,
                        chunk_type=chunk_type or "general",
                        faculty=all_people,  # All people found
                        research_areas=research or [],
                        url=url or "",
                        similarity=round(similarity, 3)
                    ))
            
            processing_time = round(time.time() - start_time, 2)
            logger.info(f"✅ Advanced query processed in {processing_time}s")
            
            return RAGResponse(
                answer=response.strip(),
                sources=sources,
                processing_time=processing_time,
                query=query,
                model_used=self.model_name
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing advanced query: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize RAG service
logger.info("🚀 Starting Advanced CDS RAG Service...")
rag_service = CDSRAGService()

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CDS Advanced RAG System</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { color: #2c3e50; margin-bottom: 10px; }
            .header p { color: #7f8c8d; }
            .chat-container { margin-bottom: 30px; }
            .input-section { display: flex; gap: 10px; margin-bottom: 20px; }
            .input-section input { flex: 1; padding: 12px; border: 2px solid #e0e0e0; border-radius: 6px; font-size: 16px; }
            .input-section button { padding: 12px 24px; background: #3498db; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; }
            .input-section button:hover { background: #2980b9; }
            .input-section button:disabled { background: #bdc3c7; cursor: not-allowed; }
            .response { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; margin: 10px 0; }
            .sources { background: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 10px; border: 1px solid #ffeaa7; }
            .source-item { margin-bottom: 10px; padding: 10px; background: white; border-radius: 4px; }
            .loading { text-align: center; color: #7f8c8d; }
            .examples { margin-top: 30px; }
            .example-btn { background: #ecf0f1; color: #2c3e50; border: 1px solid #bdc3c7; padding: 8px 12px; margin: 5px; border-radius: 4px; cursor: pointer; display: inline-block; }
            .example-btn:hover { background: #d5dbdb; }
            .status { padding: 10px; background: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; margin-bottom: 20px; }
            .feature-highlight { background: #e8f5e8; padding: 8px; border-radius: 4px; margin: 5px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 CDS Advanced RAG System</h1>
                <p>Intelligent Question Answering with Comprehensive People Search for CDS, IISc Bangalore</p>
                <div class="status">
                    ✅ System Ready | 🤖 Model: phi3:mini | 📊 Database: Connected | 👥 Advanced People Search: Active
                </div>
                <div class="feature-highlight">
                    🎯 <strong>New Features:</strong> Faculty Search | Student Discovery | Research Area Mapping | Lab Member Identification
                </div>
            </div>
            
            <div class="chat-container">
                <div class="input-section">
                    <input type="text" id="questionInput" placeholder="Ask about faculty, students, research areas, labs, or any CDS information..." />
                    <button id="askButton" onclick="askQuestion()">Ask Question</button>
                </div>
                
                <div id="responseArea"></div>
            </div>
            
            <div class="examples">
                <h3>💡 Try These Advanced Queries:</h3>
                <span class="example-btn" onclick="setQuestion('Who are the faculty members in CDS?')">All Faculty</span>
                <span class="example-btn" onclick="setQuestion('Students working in DREAM lab')">Lab Students</span>
                <span class="example-btn" onclick="setQuestion('People working in NLP and machine learning')">Research Areas</span>
                <span class="example-btn" onclick="setQuestion('PhD students in computational biology')">PhD Students</span>
                <span class="example-btn" onclick="setQuestion('What research does Danish Pruthi do?')">Faculty Research</span>
                <span class="example-btn" onclick="setQuestion('M.Tech students in data science')">M.Tech Students</span>
                <span class="example-btn" onclick="setQuestion('Researchers in artificial intelligence')">AI Researchers</span>
                <span class="example-btn" onclick="setQuestion('Faculty and students in machine learning lab')">Lab Members</span>
            </div>
        </div>
        
        <script>
            function setQuestion(question) {
                document.getElementById('questionInput').value = question;
            }
            
            function askQuestion() {
                const question = document.getElementById('questionInput').value.trim();
                if (!question) return;
                
                const responseArea = document.getElementById('responseArea');
                const askButton = document.getElementById('askButton');
                
                askButton.disabled = true;
                askButton.textContent = 'Processing...';
                responseArea.innerHTML = '<div class="loading">🤖 Advanced search in progress...</div>';
                
                fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question, top_k: 6, include_sources: true })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.detail) {
                        responseArea.innerHTML = `<div class="response" style="border-left-color: #e74c3c;"><strong>Error:</strong> ${data.detail}</div>`;
                        return;
                    }
                    
                    let html = `
                        <div class="response">
                            <strong>Q:</strong> ${data.query}<br><br>
                            <strong>A:</strong> ${data.answer}
                            <br><br>
                            <small>⏱️ Response time: ${data.processing_time}s | 🤖 Model: ${data.model_used} | 🎯 Advanced Search</small>
                        </div>
                    `;
                    
                    if (data.sources && data.sources.length > 0) {
                        html += '<div class="sources"><strong>📚 Sources Used:</strong>';
                        data.sources.forEach((source, i) => {
                            html += `
                                <div class="source-item">
                                    <strong>Source ${i+1}</strong> (${(source.similarity * 100).toFixed(1)}% relevance)<br>
                                    <strong>Type:</strong> ${source.chunk_type}<br>
                                    ${source.faculty.length > 0 ? `<strong>People:</strong> ${source.faculty.join(', ')}<br>` : ''}
                                    ${source.research_areas.length > 0 ? `<strong>Research:</strong> ${source.research_areas.join(', ')}<br>` : ''}
                                    <strong>Content:</strong> ${source.content}
                                    ${source.url ? `<br><strong>URL:</strong> <a href="${source.url}" target="_blank">${source.url}</a>` : ''}
                                </div>
                            `;
                        });
                        html += '</div>';
                    }
                    
                    responseArea.innerHTML = html;
                })
                .catch(error => {
                    responseArea.innerHTML = `<div class="response" style="border-left-color: #e74c3c;"><strong>Error:</strong> ${error.message}</div>`;
                })
                .finally(() => {
                    askButton.disabled = false;
                    askButton.textContent = 'Ask Question';
                });
            }
            
            document.getElementById('questionInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !document.getElementById('askButton').disabled) {
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/ask", response_model=RAGResponse)
async def ask_question(request: QueryRequest):
    """API endpoint to ask questions about CDS"""
    return await rag_service.process_query(
        query=request.question,
        top_k=request.top_k,
        include_sources=request.include_sources
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = psycopg2.connect(**rag_service.db_config)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM cds_embeddings;")
        db_count = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        response = requests.get(f"{rag_service.ollama_url}/api/tags", timeout=5)
        ollama_models = response.json().get("models", []) if response.status_code == 200 else []
        
        return {
            "status": "healthy",
            "database": {"status": "connected", "embeddings_count": db_count},
            "ollama": {"status": "connected", "models": [m["name"] for m in ollama_models]},
            "embedding_model": "all-MiniLM-L6-v2",
            "current_ollama_model": rag_service.model_name,
            "features": ["advanced_people_search", "research_area_mapping", "lab_discovery"]
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting Advanced CDS RAG FastAPI Server...")
    print("📍 Access the web interface at: http://localhost:8000")
    print("📊 API documentation at: http://localhost:8000/api/docs")
    print("🔍 Health check at: http://localhost:8000/api/health")
    print("🎯 Features: Advanced People Search | Research Area Mapping | Lab Discovery")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )