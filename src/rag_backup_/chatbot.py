"""
LangChain-based RAG chatbot with OpenAI integration
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from src.rag.retriever import HybridRetriever
from src.config import config


class RAGChatbot:
    """RAG chatbot using LangChain and OpenAI"""
    
    def __init__(self, retriever: HybridRetriever = None):
        """
        Initialize RAG chatbot
        
        Args:
            retriever: Hybrid retriever instance
        """
        self.retriever = retriever or HybridRetriever()
        
        # Load configuration
        self.temperature = config.get('rag', 'generation', 'temperature', default=0.7)
        self.max_tokens = config.get('rag', 'generation', 'max_tokens', default=500)
        self.system_prompt = config.get('rag', 'generation', 'system_prompt', default='')
        self.include_sources = config.get('rag', 'response', 'include_sources', default=True)
        self.max_sources = config.get('rag', 'response', 'max_sources', default=3)
        
        # Initialize OpenAI client
        self.openai_client = None
        self._initialize_openai()
        
        logger.info("RAG Chatbot initialized")
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            
            api_key = config.openai_api_key
            if not api_key:
                logger.warning("OpenAI API key not found. Chatbot will not function.")
                return
            
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
    
    def chat(self, query: str, filters: Dict[str, Any] = None, 
             conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a chat query and generate response
        
        Args:
            query: User query
            filters: Optional metadata filters for retrieval
            conversation_history: Optional conversation history
        
        Returns:
            Dictionary with response and metadata
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # 1. Retrieve relevant context
            retrieval_results = self.retriever.retrieve(query, filters)
            
            # 2. Format context
            context = self.retriever.get_context_for_generation(retrieval_results)
            
            # 3. Generate response
            response = self._generate_response(query, context, conversation_history)
            
            # 4. Format output
            output = {
                'query': query,
                'response': response,
                'context_used': context[:500] + '...' if len(context) > 500 else context,
                'num_sources': len(retrieval_results.get('vector_results', [])),
            }
            
            # Add sources if configured
            if self.include_sources:
                output['sources'] = self._format_sources(retrieval_results)
            
            logger.info("Query processed successfully")
            return output
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'query': query,
                'response': "I apologize, but I encountered an error processing your query. Please try again.",
                'error': str(e)
            }
    
    def _generate_response(self, query: str, context: str, 
                          conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Generate response using OpenAI with special handling for list queries
        
        Args:
            query: User query
            context: Retrieved context
            conversation_history: Optional conversation history
        
        Returns:
            Generated response text
        """
        if not self.openai_client:
            return "OpenAI client not initialized. Please check your API key."
        
        try:
            # Detect list queries that need comprehensive responses
            is_list_query = any(keyword in query.lower() for keyword in [
                'list', 'show me', 'show all', 'who are', 'all', 'faculty', 'members', 
                'labs', 'people', 'researchers', 'professors', 'staff'
            ])
            
            # Use enhanced system prompt for list queries
            if is_list_query:
                system_prompt = self.system_prompt + """

CRITICAL INSTRUCTION FOR THIS LIST QUERY:
- You MUST include EVERY item found in the context
- Do NOT summarize or say "and others" - list ALL items
- Use numbered list format for clarity
- Include complete details for each item (name, title, research areas, contact)
- If context has 15+ items, list ALL 15+ items
- Never truncate lists - completeness is more important than brevity"""
            else:
                system_prompt = self.system_prompt
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history[-5:])  # Last 5 exchanges
            
            # Add context and query with emphasis on completeness for lists
            if is_list_query:
                user_message = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

IMPORTANT: This is a LIST query. Provide a COMPLETE list with ALL items found in the context. Do not truncate or summarize. Include every single item with full details."""
            else:
                user_message = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Please provide a clear, accurate, and concise answer based on the context provided. If the context doesn't contain enough information to fully answer the question, acknowledge this and provide what information is available."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=config.get('rag', 'generation', 'model', default='gpt-4-turbo-preview'),
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"
    
    def _format_sources(self, retrieval_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Format sources for citation
        
        Args:
            retrieval_results: Results from retriever
        
        Returns:
            List of formatted source dictionaries
        """
        sources = []
        
        # Get top sources from vector results
        for doc in retrieval_results.get('vector_results', [])[:self.max_sources]:
            metadata = doc.get('metadata', {})
            source = {
                'title': metadata.get('title', 'Untitled'),
                'url': metadata.get('url', ''),
                'page_type': metadata.get('page_type', 'general'),
                'similarity': f"{doc.get('similarity', 0):.2f}"
            }
            sources.append(source)
        
        return sources
    
    def ask_about_faculty(self, faculty_name: str) -> Dict[str, Any]:
        """
        Ask specific question about a faculty member
        
        Args:
            faculty_name: Name of faculty member
        
        Returns:
            Response dictionary with faculty information
        """
        query = f"Tell me about {faculty_name}, including their research interests and affiliations."
        
        # Get faculty-specific information
        faculty_info = self.retriever.retrieve_faculty_info(faculty_name)
        
        if faculty_info:
            # Format faculty information
            context = f"""Faculty: {faculty_name}
Labs: {', '.join(faculty_info.get('labs', []))}
Projects: {', '.join(faculty_info.get('projects', []))}
Research Topics: {', '.join(faculty_info.get('topics', []))}"""
            
            response = self._generate_response(query, context)
            
            return {
                'query': query,
                'response': response,
                'faculty_info': faculty_info
            }
        else:
            return {
                'query': query,
                'response': f"I couldn't find information about {faculty_name} in the database.",
                'faculty_info': {}
            }
    
    def ask_about_research_topic(self, topic: str) -> Dict[str, Any]:
        """
        Ask about a specific research topic
        
        Args:
            topic: Research topic
        
        Returns:
            Response dictionary with topic information
        """
        query = f"What research is being done on {topic} at IISc?"
        
        # Use topic-specific filter
        filters = {'page_type': 'faculty'}  # Focus on faculty pages
        
        return self.chat(query, filters=filters)
    
    def ask_about_courses(self, course_query: str) -> Dict[str, Any]:
        """
        Ask about courses
        
        Args:
            course_query: Query about courses
        
        Returns:
            Response dictionary
        """
        filters = {'page_type': 'course'}
        return self.chat(course_query, filters=filters)
    
    def multi_turn_conversation(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Handle multi-turn conversation
        
        Args:
            queries: List of queries in order
        
        Returns:
            List of responses
        """
        conversation_history = []
        responses = []
        
        for query in queries:
            response = self.chat(query, conversation_history=conversation_history)
            responses.append(response)
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": response['response']})
        
        return responses
