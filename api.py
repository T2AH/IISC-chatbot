"""
FastAPI Server for IISc Research Chatbot
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv

# LOAD ENV FIRST
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

from src.rag.langgraph_chatbot import LangGraphChatbot
from src.database.chat_db import ChatDB
from loguru import logger

# Initialize FastAPI app
app = FastAPI(
    title="IISc Research Chatbot API",
    description="RAG-based chatbot for IISc research queries",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management
sessions: Dict[str, Dict] = {}  # session_id -> {history: [], last_access: datetime}
SESSION_TIMEOUT = timedelta(hours=2)

# Initialize chatbot and chat database
chatbot = None
chat_db = None

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot and database on startup"""
    global chatbot, chat_db
    try:
        chatbot = LangGraphChatbot(openai_api_key=OPENAI_KEY)
        logger.info("Chatbot initialized successfully")
        
        chat_db = ChatDB()
        logger.info("Chat database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise


# Request/Response models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    thread_id: Optional[str] = None  # For persistent threads
    conversation_history: Optional[List[Dict]] = []


class ChatResponse(BaseModel):
    answer: str
    query: str
    context_used: int
    messages: List[Dict]
    session_id: str
    thread_id: Optional[str] = None  # For persistent threads


@app.get("/")
async def root():
    """Serve the web UI"""
    return FileResponse("static/index.html")


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "IISc Research Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "/": "GET - Web UI",
            "/chat": "POST - Chat with the bot",
            "/health": "GET - Health check",
            "/stats": "GET - Database statistics",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    return {
        "status": "healthy",
        "chatbot": "ready",
        "databases": {
            "chromadb": "connected",
            "neo4j": "connected" if chatbot.retriever.neo4j.driver else "disconnected"
        }
    }


def cleanup_old_sessions():
    """Remove expired sessions"""
    now = datetime.now()
    expired = [sid for sid, data in sessions.items() 
               if now - data['last_access'] > SESSION_TIMEOUT]
    for sid in expired:
        del sessions[sid]
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions")


def resolve_query_context(query: str, history: List[Dict]) -> str:
    """Resolve pronouns and references using conversation history"""
    if not history:
        return query
    
    query_lower = query.lower()
    
    # Pronouns that indicate reference to previous context
    reference_patterns = [
        'his ', 'her ', 'their ', 'this ', 'that ', 'these ', 'those ',
        'the same ', 'it ', 'they ', 'he ', 'she ',
        'tell me more', 'more about', 'what about', 'more info',
        'continue', 'go on', 'elaborate', 'explain more'
    ]
    
    has_reference = any(pattern in query_lower for pattern in reference_patterns)
    
    if has_reference and len(history) >= 2:
        # Get last user query and assistant response
        last_user = None
        last_assistant = None
        
        for msg in reversed(history):
            if msg['role'] == 'user' and last_user is None:
                last_user = msg['content']
            elif msg['role'] == 'assistant' and last_assistant is None:
                last_assistant = msg['content']
            if last_user and last_assistant:
                break
        
        if last_user:
            # Add context from previous query
            return f"Previous question was: '{last_user}'. Now answering: {query}"
    
    return query


def expand_acronyms(query: str) -> str:
    """Expand common IISc acronyms"""
    acronyms = {
    ' biochem ': ' department of biochemistry ',
    ' caf ': ' central animal facility ',
    ' ces ': ' centre for ecological sciences ',
    ' cidr ': ' centre for infectious disease research ',
    ' cns ': ' centre for neuroscience ',
    ' mcb ': ' department of microbiology and cell biology ',
    ' mbu ': ' molecular biophysics unit ',
    ' dbg ': ' department of developmental biology and genetics ',
    ' ipc ': ' department of inorganic and physical chemistry ',
    ' mrc ': ' materials research centre ',
    ' orgchem ': ' department of organic chemistry ',
    ' sscu ': ' solid state and structural chemistry unit ',
    ' csa ': ' computer science and automation ',
    ' ece ': ' electrical communication engineering ',
    ' dese ': ' department of electronic systems engineering ',
    ' ee ': ' electrical engineering ',
    ' cistup ': ' centre for infrastructure, sustainable transportation and urban planning ',
    ' be ': ' bioengineering ',
    ' csp ': ' centre for sustainable technologies ',
    ' cense ': ' centre for nanoscience and engineering ',
    ' cds ': ' computational and data sciences ',
    ' mgmt ': ' management studies ',
    ' icer ': ' interdisciplinary centre for energy research ',
    ' icwar ': ' interdisciplinary centre for water research ',
    ' cps ': ' centre for contemporary studies ',
    ' msci ': ' department of materials science ',
    ' serc ': ' supercomputer education and research centre ',
    ' iqti ': ' international centre for quantum technology initiatives ',
    ' abcmc ': ' atomic, biomolecular and chemical sciences centre ',
    ' longevity ': ' centre for longevity research ',
    ' aero ': ' aerospace engineering ',
    ' caos ': ' centre for atmospheric and oceanic sciences ',
    ' ceas ': ' centre for earth sciences ',
    ' camm ': ' centre for advanced manufacturing and materials ',
    ' dm ': ' department of mathematics ',
    ' cst ': ' centre for scientific teaching ',
    ' chemeng ': ' chemical engineering ',
    ' civil ': ' civil engineering ',
    ' dccc ': ' digital campus and cloud computing centre ',
    ' materials ': ' materials engineering ',
    ' mecheng ': ' mechanical engineering ',
    ' physics_jap ': ' department of physics (japan group page) ',
    ' cct ': ' centre for catalysis and transition metal chemistry ',
    ' chep ': ' centre for high energy physics ',
    ' math ': ' mathematics department ',
    ' iap ': ' instrumentation and applied physics ',
    ' physics ': ' department of physics ',
    ' cbr ': ' centre for brain research ',
    ' fsid ': ' foundation for science, innovation and development ',
    ' diarcoe ': ' department of interdisciplinary and applied research in chemical engineering '
    }
    
    # Add spaces to handle word boundaries correctly
    query_padded = f" {query.lower()} "
    result = query_padded
    
    for acronym, full in acronyms.items():
        result = result.replace(acronym, full)
    
    # Remove padding spaces
    return result.strip()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - process user query and return answer with conversation memory
    
    Args:
        request: ChatRequest with query and optional session_id
    
    Returns:
        ChatResponse with answer and metadata
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        # Clean up old sessions periodically
        cleanup_old_sessions()
        
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        
        if session_id not in sessions:
            sessions[session_id] = {
                'history': [],
                'last_access': datetime.now()
            }
            logger.info(f"Created new session: {session_id}")
        
        # Update last access
        sessions[session_id]['last_access'] = datetime.now()
        
        # Get conversation history
        conversation_history = sessions[session_id]['history']
        
        # Expand acronyms in query
        expanded_query = expand_acronyms(request.query)
        if expanded_query != request.query:
            logger.debug(f"Expanded query: '{request.query}' -> '{expanded_query}'")
        
        # Resolve references using conversation history
        enriched_query = resolve_query_context(expanded_query, conversation_history)
        if enriched_query != expanded_query:
            logger.debug(f"Enriched query with context: '{expanded_query}' -> '{enriched_query}'")
        
        # Chat with history
        result = chatbot.chat(
            query=enriched_query,
            conversation_history=conversation_history
        )
        
        # Update session history
        sessions[session_id]['history'].append({
            "role": "user",
            "content": request.query
        })
        sessions[session_id]['history'].append({
            "role": "assistant",
            "content": result["answer"]
        })
        
        # Keep only last 10 exchanges (20 messages)
        if len(sessions[session_id]['history']) > 20:
            sessions[session_id]['history'] = sessions[session_id]['history'][-20:]
        
        # Add session_id to response
        result['session_id'] = session_id
        
        logger.info(f"Session {session_id}: Query processed, history length: {len(sessions[session_id]['history'])}")
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        # Get ChromaDB stats
        chroma_collection = chatbot.retriever.chromadb.collection
        chroma_count = chroma_collection.count() if chroma_collection else 0
        
        # Get Neo4j stats
        neo4j_stats = {}
        if chatbot.retriever.neo4j.driver:
            with chatbot.retriever.neo4j.driver.session() as session:
                pages = session.run("MATCH (p:Page) RETURN count(p)").single()[0]
                entities = session.run("MATCH (e:Entity) RETURN count(e)").single()[0]
                rels = session.run("MATCH ()-[r:MENTIONS]->() RETURN count(r)").single()[0]
                neo4j_stats = {
                    "pages": pages,
                    "entities": entities,
                    "relationships": rels
                }
        
        return {
            "chromadb": {
                "total_chunks": chroma_count
            },
            "neo4j": neo4j_stats
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# CHAT THREAD MANAGEMENT ENDPOINTS
# ============================================================

class ThreadResponse(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    context_used: int
    created_at: str


@app.get("/api/threads", response_model=List[ThreadResponse])
async def get_threads(user_id: str = "default", status: str = "active"):
    """
    Get all chat threads for a user
    
    Args:
        user_id: User identifier (default: 'default')
        status: Thread status ('active' or 'archived')
    
    Returns:
        List of thread summaries
    """
    if chat_db is None:
        raise HTTPException(status_code=503, detail="Chat database not initialized")
    
    try:
        threads = chat_db.get_all_threads(user_id=user_id, status=status)
        return threads
    except Exception as e:
        logger.error(f"Get threads error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/threads", response_model=Dict)
async def create_thread(initial_query: str, user_id: str = "default"):
    """
    Create a new chat thread
    
    Args:
        initial_query: First query to initialize thread
        user_id: User identifier
    
    Returns:
        Thread info and first response
    """
    if chat_db is None or chatbot is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Create thread
        thread_id = chat_db.create_thread(initial_query, user_id)
        
        # Process first query
        expanded_query = expand_acronyms(initial_query)
        result = chatbot.chat(query=expanded_query, conversation_history=[])
        
        # Save messages
        chat_db.add_message(thread_id, "user", initial_query)
        chat_db.add_message(
            thread_id, 
            "assistant", 
            result["answer"],
            context_used=result["context_used"]
        )
        
        return {
            "thread_id": thread_id,
            "answer": result["answer"],
            "context_used": result["context_used"]
        }
    except Exception as e:
        logger.error(f"Create thread error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/threads/{thread_id}", response_model=Dict)
async def get_thread_details(thread_id: str):
    """
    Get thread information and messages
    
    Args:
        thread_id: Thread identifier
    
    Returns:
        Thread info with all messages
    """
    if chat_db is None:
        raise HTTPException(status_code=503, detail="Chat database not initialized")
    
    try:
        # Get thread info
        thread_info = chat_db.get_thread_info(thread_id)
        if not thread_info:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Get messages
        messages = chat_db.get_thread_messages(thread_id)
        
        return {
            **thread_info,
            "messages": messages
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get thread details error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class MessageRequest(BaseModel):
    query: str


@app.post("/api/threads/{thread_id}/messages", response_model=Dict)
async def continue_thread(thread_id: str, request: MessageRequest):
    """
    Continue conversation in existing thread
    
    Args:
        thread_id: Thread identifier
        request: Message request with query
    
    Returns:
        Assistant response
    """
    if chat_db is None or chatbot is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Verify thread exists
        thread_info = chat_db.get_thread_info(thread_id)
        if not thread_info:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Get conversation history
        conversation_history = chat_db.get_conversation_history(thread_id, limit=20)
        
        # Process query with context
        expanded_query = expand_acronyms(request.query)
        enriched_query = resolve_query_context(expanded_query, conversation_history)
        
        result = chatbot.chat(
            query=enriched_query,
            conversation_history=conversation_history
        )
        
        # Save messages
        chat_db.add_message(thread_id, "user", request.query)
        chat_db.add_message(
            thread_id,
            "assistant",
            result["answer"],
            context_used=result["context_used"]
        )
        
        return {
            "answer": result["answer"],
            "context_used": result["context_used"],
            "thread_id": thread_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Continue thread error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """
    Delete a chat thread permanently
    
    Args:
        thread_id: Thread identifier
    
    Returns:
        Success message
    """
    if chat_db is None:
        raise HTTPException(status_code=503, detail="Chat database not initialized")
    
    try:
        deleted = chat_db.delete_thread(thread_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        return {"message": "Thread deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete thread error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/threads/{thread_id}/archive")
async def archive_thread(thread_id: str):
    """
    Archive a chat thread (soft delete)
    
    Args:
        thread_id: Thread identifier
    
    Returns:
        Success message
    """
    if chat_db is None:
        raise HTTPException(status_code=503, detail="Chat database not initialized")
    
    try:
        archived = chat_db.archive_thread(thread_id)
        if not archived:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        return {"message": "Thread archived successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Archive thread error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/threads/search/{query}", response_model=List[ThreadResponse])
async def search_threads(query: str, user_id: str = "default"):
    """
    Search threads by title or content
    
    Args:
        query: Search query
        user_id: User identifier
    
    Returns:
        List of matching threads
    """
    if chat_db is None:
        raise HTTPException(status_code=503, detail="Chat database not initialized")
    
    try:
        threads = chat_db.search_threads(query, user_id)
        return threads
    except Exception as e:
        logger.error(f"Search threads error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
