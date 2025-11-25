"""
Enhanced LangChain-based RAG chatbot with conversation memory
Integrates with existing RAGChatbot while adding persistent context
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from src.rag.chatbot import RAGChatbot
from src.config import config


class EnhancedRAGChatbot(RAGChatbot):
    """
    Enhanced RAG chatbot with LangChain conversation memory
    Extends existing RAGChatbot with context-aware conversations
    """
    
    def __init__(self, retriever=None, session_id: str = "default"):
        """
        Initialize enhanced chatbot with memory
        
        Args:
            retriever: Hybrid retriever instance
            session_id: Unique session identifier for multi-user support
        """
        super().__init__(retriever)
        
        self.session_id = session_id
        self.conversation_memory = {}  # Dict to store multiple sessions
        
        # Initialize LangChain memory if available
        self._initialize_memory()
        
        logger.info(f"Enhanced RAG Chatbot initialized for session: {session_id}")
    
    def _initialize_memory(self):
        """Initialize LangChain conversation memory"""
        try:
            from langchain.memory import ConversationBufferMemory
            
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="response"
            )
            logger.info("LangChain memory initialized")
            
        except ImportError:
            logger.warning("LangChain not installed. Install: pip install langchain")
            self.memory = None
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            self.memory = None
    
    def chat_with_memory(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process query with conversation memory
        
        Args:
            query: User query
            filters: Optional metadata filters
        
        Returns:
            Response dictionary with conversation context
        """
        try:
            logger.info(f"Processing query with memory: {query[:100]}...")
            
            # 1. Get conversation history from memory
            history = self._get_conversation_history()
            
            # 2. Use parent class chat method with history
            response = self.chat(query, filters, history)
            
            # 3. Update memory with new interaction
            if self.memory:
                self.memory.save_context(
                    {"input": query},
                    {"response": response.get('response', '')}
                )
            
            # 4. Add conversation context to response
            response['conversation_turns'] = len(history) // 2 if history else 0
            response['session_id'] = self.session_id
            
            return response
        
        except Exception as e:
            logger.error(f"Error in chat_with_memory: {e}")
            return {
                'query': query,
                'response': f"Error: {str(e)}",
                'error': str(e)
            }
    
    def _get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get formatted conversation history
        
        Returns:
            List of conversation messages
        """
        if not self.memory:
            return []
        
        try:
            # Get messages from LangChain memory
            memory_vars = self.memory.load_memory_variables({})
            messages = memory_vars.get('chat_history', [])
            
            # Convert to OpenAI format
            formatted = []
            for msg in messages:
                role = "user" if msg.type == "human" else "assistant"
                formatted.append({
                    "role": role,
                    "content": msg.content
                })
            
            return formatted
        
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def clear_memory(self):
        """Clear conversation memory for current session"""
        if self.memory:
            self.memory.clear()
            logger.info(f"Memory cleared for session: {self.session_id}")
    
    def get_conversation_summary(self) -> str:
        """
        Get summary of current conversation
        
        Returns:
            Summary string
        """
        history = self._get_conversation_history()
        if not history:
            return "No conversation history"
        
        turns = len(history) // 2
        return f"Session {self.session_id}: {turns} conversation turn(s)"


class SessionManager:
    """
    Manage multiple chatbot sessions for different users
    """
    
    def __init__(self):
        self.sessions: Dict[str, EnhancedRAGChatbot] = {}
        logger.info("SessionManager initialized")
    
    def get_session(self, session_id: str) -> EnhancedRAGChatbot:
        """
        Get or create chatbot session
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            EnhancedRAGChatbot instance
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = EnhancedRAGChatbot(session_id=session_id)
            logger.info(f"Created new session: {session_id}")
        
        return self.sessions[session_id]
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        if session_id in self.sessions:
            self.sessions[session_id].clear_memory()
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs"""
        return list(self.sessions.keys())
    
    def clear_all_sessions(self):
        """Clear all sessions"""
        for session_id in list(self.sessions.keys()):
            self.delete_session(session_id)
        logger.info("All sessions cleared")


# Example usage
if __name__ == "__main__":
    # Single session
    chatbot = EnhancedRAGChatbot(session_id="user123")
    
    # Multi-turn conversation
    response1 = chatbot.chat_with_memory("What is IISc?")
    print(f"Bot: {response1['response']}\n")
    
    response2 = chatbot.chat_with_memory("Tell me more about its research")
    print(f"Bot: {response2['response']}\n")
    
    # Check conversation turns
    print(chatbot.get_conversation_summary())
    
    # Multi-user sessions
    manager = SessionManager()
    
    user1_bot = manager.get_session("user_alice")
    user2_bot = manager.get_session("user_bob")
    
    # Each user has independent context
    user1_bot.chat_with_memory("What is machine learning?")
    user2_bot.chat_with_memory("What is quantum computing?")
