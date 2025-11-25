"""
Enhanced RAG Chatbot with Improved Retriever and Reranker
Uses multi-strategy retrieval, 5-stage reranking, and query-aware prompting

FEATURES:
- Uses EnhancedHybridRetriever (4 retrieval strategies + RRF)
- Uses EnhancedReranker (5-stage reranking with LLM intelligence)
- Query-aware prompting (different for lists vs lookups)
- Better context building with metadata
- Debug mode for troubleshooting
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from src.rag.retriever import HybridRetriever
from src.rag.reranker import Reranker
from src.config import config


class RAGChatbot:
    """Enhanced RAG chatbot with improved retrieval and reranking"""
    
    def __init__(self, retriever: HybridRetriever = None, 
                 reranker: Reranker = None,
                 debug: bool = False):
        """
        Initialize enhanced RAG chatbot
        
        Args:
            retriever: Enhanced hybrid retriever instance
            reranker: Enhanced reranker instance
            debug: Enable debug mode for troubleshooting
        """
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or Reranker()
        self.debug = debug
        
        # Load configuration
        self.temperature = config.get('rag', 'generation', 'temperature', default=0.7)
        self.max_tokens = config.get('rag', 'generation', 'max_tokens', default=500)
        self.system_prompt = config.get('rag', 'generation', 'system_prompt', default='')
        self.include_sources = config.get('rag', 'response', 'include_sources', default=True)
        self.max_sources = config.get('rag', 'response', 'max_sources', default=3)
        
        # Initialize OpenAI client
        self.openai_client = None
        self._initialize_openai()
        
        logger.info("Enhanced RAG Chatbot initialized with improved retriever and reranker")
    
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
             conversation_history: List[Dict[str, str]] = None,
             debug: bool = None) -> Dict[str, Any]:
        """
        Process a chat query and generate response
        
        Args:
            query: User query
            filters: Optional metadata filters for retrieval
            conversation_history: Optional conversation history
            debug: Override instance debug setting
        
        Returns:
            Dictionary with response and metadata
        """
        debug_mode = debug if debug is not None else self.debug
        
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Step 1: Enhanced multi-strategy retrieval with RRF
            logger.info("[Step 1] Multi-strategy retrieval with RRF...")
            retrieval_results = self.retriever.retrieve(query, filters, top_k=100)
            
            if debug_mode:
                logger.debug(f"Retrieval strategies: {retrieval_results.get('strategy_results', {})}")
                logger.debug(f"Query analysis: {retrieval_results.get('query_analysis', {})}")
            
            # Step 2: Enhanced 5-stage reranking
            logger.info("[Step 2] 5-stage reranking...")
            vector_results = retrieval_results.get('vector_results', [])
            query_analysis = retrieval_results.get('query_analysis', {})
            matched_depts = retrieval_results.get('matched_depts', [])
            
            reranked_results = self.reranker.rerank(
                query=query,
                results=vector_results,
                query_analysis=query_analysis,
                top_k=50,
                matched_depts=matched_depts
            )
            
            if debug_mode:
                logger.debug(f"Reranking reduced {len(vector_results)} -> {len(reranked_results)} results")
                if reranked_results:
                    top = reranked_results[0]
                    logger.debug(f"Top result: {top.get('source', 'Unknown')[:80]}")
                    logger.debug(f"  Scores - Stage1: {top.get('stage1_score', 0):.4f}, "
                               f"Stage2: {top.get('stage2_score', 0):.4f}, "
                               f"Final: {top.get('final_rerank_score', 0):.4f}")
            
            # Step 3: Build context with enhanced metadata
            logger.info("[Step 3] Building context...")
            context = self._build_enhanced_context(reranked_results, query_analysis)
            
            # Step 4: Query-aware response generation
            logger.info("[Step 4] Generating response...")
            response = self._generate_response(query, context, query_analysis, conversation_history)
            
            # Step 5: Format output
            output = {
                'query': query,
                'response': response,
                'answer': response,  # API expects 'answer' field
                'context_used': len(reranked_results),  # API expects 'context_used' field
                'num_sources': len(reranked_results),
                'query_analysis': query_analysis,
            }
            
            # Add sources if configured
            if self.include_sources:
                output['sources'] = self._format_sources(reranked_results)
            
            # Add debug information if requested
            if debug_mode:
                output['debug'] = {
                    'retrieval_strategies': retrieval_results.get('strategy_results', {}),
                    'matched_departments': matched_depts,
                    'top_3_results': [
                        {
                            'source': r.get('source', 'Unknown')[:80],
                            'title': r.get('title', 'Unknown')[:80],
                            'page_type': r.get('page_type', 'Unknown'),
                            'final_score': r.get('final_rerank_score', 0),
                            'strategies': r.get('matched_strategies', []),
                            'boost_reasons': r.get('boost_reasons', [])
                        }
                        for r in reranked_results[:3]
                    ],
                    'context_length': len(context)
                }
            
            logger.info("Query processed successfully")
            return output
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'query': query,
                'response': "I apologize, but I encountered an error processing your query. Please try again.",
                'answer': "I apologize, but I encountered an error processing your query. Please try again.",
                'context_used': 0,
                'error': str(e)
            }
    
    def _build_enhanced_context(self, results: List[Dict[str, Any]], 
                               query_analysis: Dict[str, Any]) -> str:
        """
        Build context string with enhanced metadata
        
        Args:
            results: Reranked results
            query_analysis: Query analysis metadata
        
        Returns:
            Formatted context string
        """
        context_parts = []
        is_list_query = query_analysis.get('is_list_query', False)
        
        # For list queries, include more results
        max_results = 20 if is_list_query else 10
        
        for i, doc in enumerate(results[:max_results], 1):
            title = doc.get('title', 'Untitled')
            page_type = doc.get('page_type', 'general')
            source = doc.get('source', 'Unknown')
            text = doc.get('text', '')
            
            # For list queries, include more complete information
            if is_list_query:
                context_parts.append(f"[Source {i} - {page_type}]\nTitle: {title}\nContent: {text}\nURL: {source}\n")
            else:
                # For specific queries, truncate text if needed
                truncated_text = text if len(text) <= 500 else text[:500] + "..."
                context_parts.append(f"[Source {i}] {truncated_text}\n")
        
        return '\n---\n'.join(context_parts)
    
    def _generate_response(self, query: str, context: str, 
                          query_analysis: Dict[str, Any] = None,
                          conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Generate response with query-aware prompting
        
        Args:
            query: User query
            context: Retrieved context
            query_analysis: Query analysis metadata
            conversation_history: Optional conversation history
        
        Returns:
            Generated response text
        """
        if not self.openai_client:
            return "OpenAI client not initialized. Please check your API key."
        
        try:
            # Analyze query type for appropriate prompting
            is_list_query = False
            is_specific_query = False
            
            if query_analysis:
                is_list_query = query_analysis.get('is_list_query', False)
                is_specific_query = any(kw in query.lower() for kw in ['who is', 'about', 'tell me about', 'describe'])
            else:
                q_lower = query.lower()
                is_list_query = any(kw in q_lower for kw in ['list', 'all', 'show all', 'who are'])
                is_specific_query = any(kw in q_lower for kw in ['who is', 'about', 'tell me about'])
            
            # Build query-aware system prompt
            if is_list_query:
                system_prompt = self.system_prompt + """

CRITICAL INSTRUCTION FOR LIST QUERIES:
- This is a LIST query - user wants to see ALL items
- You MUST include EVERY person/item found in the context
- Do NOT summarize or say "and others" - list ALL items
- Use clear numbered or bulleted list format
- Include complete details for each item (name, title, research areas, contact if available)
- If context has 15+ items, list ALL 15+ items
- Never truncate lists - completeness is MORE important than brevity
- Order by relevance or alphabetically if no natural order exists"""
            
            elif is_specific_query:
                system_prompt = self.system_prompt + """

INSTRUCTION FOR SPECIFIC QUERIES:
- User is asking about a specific person or topic
- Provide detailed, comprehensive information
- Include background, expertise, research areas, affiliations
- Use clear paragraph format
- Be thorough but well-organized"""
            
            else:
                system_prompt = self.system_prompt
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history[-5:])  # Last 5 exchanges
            
            # Build user message with context
            if is_list_query:
                user_message = f"""Based on the following sources, please answer the question.

SOURCES:
{context}

QUESTION: {query}

IMPORTANT: This is a LIST query. Provide a COMPLETE list with ALL items found in the sources. Do not truncate or summarize. Include every single item with full details (name, position, research interests, contact info if available). Format as a numbered list."""
            
            elif is_specific_query:
                user_message = f"""Based on the following sources, please answer the question.

SOURCES:
{context}

QUESTION: {query}

Please provide detailed, comprehensive information about this specific person or topic. Include all relevant details from the sources."""
            
            else:
                user_message = f"""Based on the following sources, please answer the question.

SOURCES:
{context}

QUESTION: {query}

Please provide a clear, accurate, and helpful answer based on the sources provided. If the sources don't contain enough information, acknowledge this and provide what information is available."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Adjust max_tokens for list queries (they need more space)
            # But respect model limits (most models support max 4096)
            if is_list_query:
                max_tokens = min(4000, self.max_tokens * 3)  # Cap at 4000 to be safe
            else:
                max_tokens = self.max_tokens
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=config.get('rag', 'generation', 'model', default='gpt-4-turbo-preview'),
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"
    
    def _format_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Format sources for citation
        
        Args:
            results: Reranked results
        
        Returns:
            List of formatted source dictionaries
        """
        sources = []
        
        for doc in results[:self.max_sources]:
            metadata = doc.get('metadata', {})
            source = {
                'title': doc.get('title', metadata.get('title', 'Untitled')),
                'url': doc.get('source', metadata.get('url', '')),
                'page_type': doc.get('page_type', metadata.get('page_type', 'general')),
                'relevance_score': f"{doc.get('final_rerank_score', 0):.3f}"
            }
            sources.append(source)
        
        return sources
    
    def ask_about_faculty(self, faculty_name: str, debug: bool = None) -> Dict[str, Any]:
        """
        Ask specific question about a faculty member
        
        Args:
            faculty_name: Name of faculty member
            debug: Enable debug mode
        
        Returns:
            Response dictionary with faculty information
        """
        query = f"Tell me about {faculty_name}, including their research interests and affiliations."
        return self.chat(query, debug=debug)
    
    def list_faculty(self, department: str = None, debug: bool = None) -> Dict[str, Any]:
        """
        List faculty members, optionally filtered by department
        
        Args:
            department: Optional department name to filter by
            debug: Enable debug mode
        
        Returns:
            Response dictionary with faculty list
        """
        if department:
            query = f"List all faculty members in the {department} department"
        else:
            query = "List all faculty members"
        
        return self.chat(query, debug=debug)
    
    def ask_about_research_topic(self, topic: str, debug: bool = None) -> Dict[str, Any]:
        """
        Ask about a specific research topic
        
        Args:
            topic: Research topic
            debug: Enable debug mode
        
        Returns:
            Response dictionary with topic information
        """
        query = f"What research is being done on {topic} at IISc?"
        
        # Use topic-specific filter
        filters = {'page_type': 'faculty'}  # Focus on faculty pages
        
        return self.chat(query, filters=filters, debug=debug)
    
    def ask_about_courses(self, course_query: str, debug: bool = None) -> Dict[str, Any]:
        """
        Ask about courses
        
        Args:
            course_query: Query about courses
            debug: Enable debug mode
        
        Returns:
            Response dictionary
        """
        filters = {'page_type': 'course'}
        return self.chat(course_query, filters=filters, debug=debug)
    
    def multi_turn_conversation(self, queries: List[str], debug: bool = None) -> List[Dict[str, Any]]:
        """
        Handle multi-turn conversation
        
        Args:
            queries: List of queries in order
            debug: Enable debug mode
        
        Returns:
            List of responses
        """
        conversation_history = []
        responses = []
        
        for query in queries:
            response = self.chat(query, conversation_history=conversation_history, debug=debug)
            responses.append(response)
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": response['response']})
        
        return responses
