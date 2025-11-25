"""
Simple Example: Using the Enhanced RAG System

This script demonstrates how to use the enhanced retriever, reranker, and chatbot
for improved results on faculty queries and other questions.
"""

from src.rag.retriever_enhanced import EnhancedHybridRetriever
from src.rag.reranker_enhanced import EnhancedReranker
from src.rag.chatbot_enhanced import EnhancedRAGChatbot


def example_1_basic_usage():
    """Example 1: Basic usage with enhanced chatbot"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    # Initialize enhanced chatbot (simplest way)
    chatbot = EnhancedRAGChatbot()
    
    # Ask a question
    response = chatbot.chat("Who are the CDS faculty?")
    
    print(f"\nQuery: Who are the CDS faculty?")
    print(f"\nResponse:")
    print(response['response'])
    print(f"\nSources used: {response['num_sources']}")


def example_2_with_debug():
    """Example 2: Using debug mode to understand results"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Debug Mode")
    print("="*80)
    
    # Initialize with debug enabled
    chatbot = EnhancedRAGChatbot(debug=True)
    
    # Ask a question
    response = chatbot.chat("Who are the CDS faculty?", debug=True)
    
    print(f"\nQuery: Who are the CDS faculty?")
    print(f"\nQuery Analysis:")
    print(f"  Type: {response['query_analysis'].get('query_type', 'Unknown')}")
    print(f"  Is Faculty Query: {response['query_analysis'].get('is_faculty_query', False)}")
    print(f"  Is List Query: {response['query_analysis'].get('is_list_query', False)}")
    print(f"  Departments: {response['query_analysis'].get('departments', [])}")
    
    if 'debug' in response:
        print(f"\nRetrieval Strategies Used:")
        for strategy, count in response['debug']['retrieval_strategies'].items():
            print(f"  - {strategy}: {count} results")
        
        print(f"\nTop 3 Results:")
        for i, result in enumerate(response['debug']['top_3_results'][:3], 1):
            print(f"\n  {i}. {result['title'][:70]}")
            print(f"     URL: {result['source'][:70]}")
            print(f"     Type: {result['page_type']}")
            print(f"     Score: {result['final_score']:.4f}")
            print(f"     Strategies: {', '.join(result['strategies'])}")
            print(f"     Boosts: {', '.join(result['boost_reasons'][:2])}")
    
    print(f"\nResponse:")
    print(response['response'][:500])  # First 500 chars


def example_3_retriever_only():
    """Example 3: Using just the enhanced retriever"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Enhanced Retriever Only")
    print("="*80)
    
    # Initialize retriever
    retriever = EnhancedHybridRetriever()
    
    # Retrieve documents
    results = retriever.retrieve("Who are the CDS faculty?", top_k=10)
    
    print(f"\nRetrieval Results:")
    print(f"  Strategies: {results['strategy_results']}")
    print(f"  Departments detected: {results['matched_depts']}")
    print(f"  Total results: {results['num_results']}")
    
    print(f"\nTop 5 Retrieved Documents:")
    for i, doc in enumerate(results['vector_results'][:5], 1):
        print(f"\n  {i}. {doc.get('title', 'Unknown')[:70]}")
        print(f"     URL: {doc.get('source', 'Unknown')[:70]}")
        print(f"     Type: {doc.get('page_type', 'Unknown')}")
        print(f"     RRF Score: {doc.get('rrf_score', 0):.4f}")
        print(f"     Final Score: {doc.get('final_score', 0):.4f}")
        print(f"     Matched Strategies: {doc.get('matched_strategies', [])}")


def example_4_with_reranker():
    """Example 4: Using retriever + reranker"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Retriever + Reranker")
    print("="*80)
    
    # Initialize components
    retriever = EnhancedHybridRetriever()
    reranker = EnhancedReranker(use_llm=True)
    
    # Retrieve
    query = "Who are the CDS faculty?"
    retrieval_results = retriever.retrieve(query, top_k=50)
    
    print(f"\nAfter Retrieval: {len(retrieval_results['vector_results'])} results")
    
    # Rerank
    reranked = reranker.rerank(
        query=query,
        results=retrieval_results['vector_results'],
        query_analysis=retrieval_results['query_analysis'],
        top_k=10
    )
    
    print(f"After Reranking: {len(reranked)} results")
    
    print(f"\nTop 3 After 5-Stage Reranking:")
    for i, doc in enumerate(reranked[:3], 1):
        print(f"\n  {i}. {doc.get('title', 'Unknown')[:70]}")
        print(f"     URL: {doc.get('source', 'Unknown')[:70]}")
        print(f"     Stage 1 (Cross-Encoder): {doc.get('stage1_score', 0):.4f}")
        print(f"     Stage 2 (LLM): {doc.get('stage2_score', 0):.4f}")
        print(f"     Stage 4 (Multi-Signal): {doc.get('stage4_score', 0):.4f}")
        print(f"     Final Score: {doc.get('final_rerank_score', 0):.4f}")
        print(f"     LLM Boost: {doc.get('llm_boost', 1.0):.2f}x")
        print(f"     Boost Reasons: {', '.join(doc.get('llm_reasons', [])[:2])}")


def example_5_helper_methods():
    """Example 5: Using helper methods"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Helper Methods")
    print("="*80)
    
    chatbot = EnhancedRAGChatbot()
    
    # List all faculty
    print("\n--- Listing All Faculty ---")
    response = chatbot.list_faculty()
    print(response['response'][:300])
    
    # List faculty by department
    print("\n\n--- Listing CDS Faculty ---")
    response = chatbot.list_faculty(department="CDS")
    print(response['response'][:300])
    
    # Ask about specific faculty
    print("\n\n--- Ask About Specific Faculty ---")
    response = chatbot.ask_about_faculty("Venkatesh Murthy")
    print(response['response'][:300])
    
    # Ask about research topic
    print("\n\n--- Ask About Research Topic ---")
    response = chatbot.ask_about_research_topic("machine learning")
    print(response['response'][:300])


def example_6_different_queries():
    """Example 6: Testing different query types"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Different Query Types")
    print("="*80)
    
    chatbot = EnhancedRAGChatbot(debug=False)
    
    queries = [
        "Who are the CDS faculty?",  # List query
        "Tell me about Dr. Venkatesh Murthy",  # Specific query
        "What research is done in CDS?",  # General query
        "List all machine learning researchers at IISc",  # List query with topic
    ]
    
    for query in queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        
        response = chatbot.chat(query)
        
        print(f"Query Type: {response['query_analysis'].get('query_type', 'Unknown')}")
        print(f"Is List Query: {response['query_analysis'].get('is_list_query', False)}")
        print(f"Sources: {response['num_sources']}")
        print(f"\nResponse:")
        print(response['response'][:400] + "..." if len(response['response']) > 400 else response['response'])


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("ENHANCED RAG SYSTEM - USAGE EXAMPLES")
    print("="*80)
    
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Debug Mode", example_2_with_debug),
        ("Retriever Only", example_3_retriever_only),
        ("Retriever + Reranker", example_4_with_reranker),
        ("Helper Methods", example_5_helper_methods),
        ("Different Queries", example_6_different_queries),
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...")
    print("(Press Ctrl+C to skip to next example)\n")
    
    for name, func in examples:
        try:
            func()
            input("\n[Press Enter to continue to next example...]")
        except KeyboardInterrupt:
            print("\n\nSkipping to next example...")
            continue
        except Exception as e:
            print(f"\n\nError in example '{name}': {e}")
            print("Continuing to next example...")
            continue
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    # Check if specific example requested
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_1_basic_usage,
            example_2_with_debug,
            example_3_retriever_only,
            example_4_with_reranker,
            example_5_helper_methods,
            example_6_different_queries,
        ]
        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        main()
