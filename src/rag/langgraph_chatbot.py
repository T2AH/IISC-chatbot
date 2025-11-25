"""
LangGraph-based RAG Chatbot with Context Awareness
Uses StateGraph for conversation flow management
"""

from typing import TypedDict, List, Dict, Annotated
from operator import add
import os

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.rag.retriever import HybridRetriever
from src.rag.reranker import Reranker
from loguru import logger


class ChatState(TypedDict):
    """State for conversation management"""
    messages: Annotated[List, add]  # Conversation history
    query: str  # Current user query
    retrieved_context: List[Dict]  # Retrieved documents
    reranked_context: List[Dict]  # Reranked documents
    final_answer: str  # Generated response
    conversation_id: str  # Session identifier


class LangGraphChatbot:
    """
    Context-aware RAG chatbot using LangGraph for state management
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize LangGraph chatbot
        
        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not set. Chatbot will not work without it.")
        
        # Initialize components
        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-4o",
            temperature=0.7,
            max_tokens=800
        ) if self.api_key else None
        
        # Build LangGraph workflow
        self.workflow = self._build_graph()
        
        logger.info("LangGraph Chatbot initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine for conversation flow"""
        
        graph = StateGraph(ChatState)
        
        # Add nodes (processing steps)
        graph.add_node("retrieve", self._retrieve_context)
        graph.add_node("rerank", self._rerank_context)
        graph.add_node("generate", self._generate_response)
        
        # Define edges (flow)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "generate")
        graph.add_edge("generate", END)
        
        return graph.compile()
    
    def _retrieve_context(self, state: ChatState) -> ChatState:
        """Step 1: Retrieve relevant context from databases"""
        query = state["query"]
        
        logger.info(f"Retrieving context for: {query[:100]}")
        
        # Hybrid retrieval (ChromaDB + Neo4j)
        results = self.retriever.retrieve(query)
        
        # Combine vector and graph results
        vector_results = results.get("vector_results", [])
        graph_results = results.get("graph_results", [])
        matched_depts = results.get('matched_depts', [])
        
        # Format graph results with richer context
        formatted_graph_results = []
        for r in graph_results:
            graph_text = f"Entity: {r['entity_name']} ({r['entity_type']}) - Mentioned in {r['page_count']} pages, {r['total_mentions']} total mentions"
            if r.get('sample_titles'):
                graph_text += f"\nRelated pages: {', '.join(r['sample_titles'][:3])}"
            if r.get('sample_urls'):
                graph_text += f"\nSources: {', '.join(r['sample_urls'][:2])}"
            
            formatted_graph_results.append({
                "text": graph_text,
                "score": 0.85,  # Slight boost for graph results
                "metadata": {
                    "source": "knowledge_graph", 
                    "entity_type": r['entity_type'],
                    "entity_name": r['entity_name']
                }
            })
        
        state["retrieved_context"] = vector_results + formatted_graph_results
        # expose detected departments for downstream reranking
        state['matched_depts'] = matched_depts
        
        logger.debug(f"Retrieved {len(state['retrieved_context'])} contexts")
        
        return state
    
    def _rerank_context(self, state: ChatState) -> ChatState:
        """Step 2: Rerank retrieved contexts for relevance"""
        query = state["query"]
        contexts = state["retrieved_context"]
        
        logger.info("Reranking contexts...")
        
        # Apply 3-stage reranking with higher top_k for comprehensive answers
        matched_depts = state.get('matched_depts')
        reranked = self.reranker.rerank(query, contexts, top_k=50, matched_depts=matched_depts)
        
        state["reranked_context"] = reranked
        
        logger.debug(f"Reranked to top {len(reranked)} contexts")
        
        return state
    
    def _generate_response(self, state: ChatState) -> ChatState:
        """Step 3: Generate response using LLM with context"""
        query = state["query"]
        contexts = state["reranked_context"]
        messages = state.get("messages", [])
        
        if not self.llm:
            state["final_answer"] = "ERROR: OpenAI API key not configured"
            return state
        
        logger.info("Generating response...")
        
        # Build context string with URLs
        context_parts = []
        source_urls = []
        for i, ctx in enumerate(contexts):
            source_url = ctx.get('source', 'Unknown')
            source_title = ctx.get('title', 'Unknown')
            text = ctx.get('text', '')[:500]
            
            context_parts.append(f"[Source {i+1}] Title: {source_title}\nURL: {source_url}\n{text}")
            if source_url != 'Unknown' and source_url not in source_urls:
                source_urls.append(source_url)
        
        context_str = "\n\n".join(context_parts)
        
        # Build conversation history
        history_str = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages[-6:]  # Last 3 turns
        ]) if messages else ""
        
        # System prompt - fully generalized and data-driven
        system_prompt = f"""You are an expert research assistant AI that helps users find information from an institutional knowledge base.

🚨 CRITICAL RULE: NEVER say "not specified", "not mentioned", or "not provided" if ANY relevant information exists in the context. YOU MUST extract and cite ALL names, details, and facts from the context.

CONVERSATION HISTORY:
{history_str}

RETRIEVED CONTEXT:
{context_str}

MANDATORY INSTRUCTIONS:

1. **EXTRACT NAMES FIRST**: Scan context for ALL person names, lab names, organization names, departments. ALWAYS include them.

2. **DIRECT ANSWERS**: For "who" questions, START your answer with the person's name immediately.
   ✅ CORRECT: "**Prof. [Name]** leads the [Lab Name]..."
   ❌ WRONG: "The faculty member is not specified..." (NEVER do this)

3. **CITE EVERYTHING**: If context mentions a name, number, date, affiliation, or fact - INCLUDE IT in your response.

4. **BE COMPLETE**: Extract ALL details from context:
   - Full names (with titles: Prof./Dr./etc.)
   - Organizational affiliations (labs, departments, centers)
   - Research areas and focus topics
   - Contact information (if available)
   - Web links/URLs (if available)

5. **STRUCTURE WELL**: Use bullet points, bold text, and clear formatting for readability

6. **EXPAND ACRONYMS**: When you encounter acronyms in the context, provide their full form along with the acronym.

7. **CONVERSATION AWARE**: Reference previous questions in the conversation history when answering follow-up questions.

8. **ENTITY FOCUS**: Pay special attention to entities mentioned in the context (people, organizations, locations, research topics) and highlight them.

9. **CITE SOURCES**: At the end of your response, include a "**References:**" section listing the relevant source URLs from the context for users to explore further.

Response Style: Professional, direct, informative - like a knowledgeable institutional assistant who ALWAYS provides specific names and details when available.

If the context truly does not contain the answer, suggest where the user might find more information (official website, specific department, etc.).
"""
        
        # Generate response
        prompt_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(prompt_messages)
        
        state["final_answer"] = response.content
        
        # Update conversation history
        state["messages"].extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response.content}
        ])
        
        logger.debug("Response generated successfully")
        
        return state
    
    def chat(self, query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Process a chat query with context awareness
        
        Args:
            query: User's question
            conversation_history: Previous conversation messages
        
        Returns:
            Dictionary with answer and metadata
        """
        # Initialize state
        initial_state: ChatState = {
            "messages": conversation_history or [],
            "query": query,
            "retrieved_context": [],
            "reranked_context": [],
            "final_answer": "",
            "conversation_id": ""
        }
        
        # Run through LangGraph workflow
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "answer": final_state["final_answer"],
            "query": query,
            "context_used": len(final_state["reranked_context"]),
            "messages": final_state["messages"]
        }
    
    def interactive_chat(self):
        """
        Start interactive chat session
        """
        if not self.llm:
            print("\n[ERROR] OpenAI API key required for chatbot")
            print("Please set OPENAI_API_KEY environment variable")
            print("Or pass api_key to LangGraphChatbot()")
            return
        
        print("\n" + "="*80)
        print("IISc RAG CHATBOT (LangGraph + Context Awareness)")
        print("="*80)
        print("Ask questions about IISc faculty, research, courses, etc.")
        print("Type 'quit' or 'exit' to end conversation\n")
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!\n")
                    break
                
                if not user_input:
                    continue
                
                # Get response
                result = self.chat(user_input, conversation_history)
                
                print(f"\nAssistant: {result['answer']}")
                print(f"\n[Used {result['context_used']} sources]")
                
                # Update history
                conversation_history = result["messages"]
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!\n")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                logger.error(f"Chat error: {e}")


if __name__ == "__main__":
    # Example usage (requires OPENAI_API_KEY)
    chatbot = LangGraphChatbot()
    chatbot.interactive_chat()
