"""
Example usage scripts for the IISc Research Chatbot
"""

from src.rag.chatbot import RAGChatbot
from src.nlp.pipeline import NLPPipeline
from src.database.manager import DatabaseManager
from src.utils.logger import setup_logging
from loguru import logger


def example_chatbot_usage():
    """Example: Using the RAG chatbot"""
    print("\n" + "="*80)
    print("Example: RAG Chatbot Usage")
    print("="*80 + "\n")
    
    setup_logging("INFO")
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Example queries
    queries = [
        "Who are the faculty members working on machine learning?",
        "Tell me about the Computer Systems Lab",
        "What courses are offered in data science?",
        "What research is being done on artificial intelligence?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        response = chatbot.chat(query)
        print(f"Response: {response['response']}")
        
        if 'sources' in response:
            print("\nSources:")
            for i, source in enumerate(response['sources'], 1):
                print(f"  {i}. {source['title']}")
        print()


def example_nlp_processing():
    """Example: NLP processing pipeline"""
    print("\n" + "="*80)
    print("Example: NLP Processing")
    print("="*80 + "\n")
    
    setup_logging("INFO")
    
    # Initialize pipeline
    pipeline = NLPPipeline()
    
    # Example page data (simulated)
    page_data = {
        'page_id': 'example_001',
        'url': 'https://example.com/faculty/john-doe',
        'domain': 'example.com',
        'page_type': 'faculty',
        'title': 'Dr. John Doe - Faculty Profile',
        'cleaned_text': """
        Dr. John Doe is a Professor in the Department of Computer Science.
        His research interests include machine learning, deep learning, and
        computer vision. He leads the AI Research Lab and has published over
        100 papers in top-tier conferences. Current projects focus on neural
        networks for image recognition and natural language processing.
        """
    }
    
    # Process page
    processed_page = pipeline.process_page(page_data)
    
    print(f"Processed Page ID: {processed_page['page_id']}")
    print(f"Entities found: {len(processed_page.get('entities', {}))}")
    print(f"Keywords extracted: {len(processed_page.get('keywords', []))}")
    print(f"Chunks created: {len(processed_page.get('chunks', []))}")
    print()


def example_database_operations():
    """Example: Database operations"""
    print("\n" + "="*80)
    print("Example: Database Operations")
    print("="*80 + "\n")
    
    setup_logging("INFO")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Get statistics
    stats = db_manager.get_stats()
    print(f"Database Statistics:")
    print(f"  ChromaDB Documents: {stats['chromadb'].get('count', 0)}")
    print(f"  Neo4j Connected: {stats['neo4j'].get('connected', False)}")
    print()
    
    # Example search
    query = "machine learning research"
    results = db_manager.search_similar(query, n_results=3)
    
    print(f"Search Results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['metadata'].get('title', 'Untitled')}")
        print(f"   URL: {result['metadata'].get('url', 'N/A')}")
        print(f"   Similarity: {1 - result['distance']:.2f}")
    print()
    
    db_manager.close()


def example_full_workflow():
    """Example: Complete workflow from crawling to querying"""
    print("\n" + "="*80)
    print("Example: Full Workflow")
    print("="*80 + "\n")
    
    setup_logging("INFO")
    
    print("Step 1: Crawling")
    print("  - Use: python main.py crawl --spider iisc")
    print()
    
    print("Step 2: NLP Processing")
    print("  - Use: python main.py process --input data/crawled_pages/pages_*.jsonl")
    print()
    
    print("Step 3: Database Import")
    print("  - Use: python main.py import --input data/crawled_pages/processed_*.jsonl")
    print()
    
    print("Step 4: Query Chatbot")
    print("  - Use: python main.py chat --interactive")
    print()
    
    print("Or run entire pipeline:")
    print("  - Use: python main.py pipeline")
    print()


if __name__ == '__main__':
    # Run examples
    print("\n" + "="*80)
    print("IISc Research Chatbot - Usage Examples")
    print("="*80)
    
    # Uncomment the example you want to run:
    
    # example_chatbot_usage()
    # example_nlp_processing()
    # example_database_operations()
    example_full_workflow()
