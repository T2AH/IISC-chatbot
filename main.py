"""
Main CLI entry point for the IISc Research Chatbot
"""

import argparse
import sys
from pathlib import Path

from src.utils.logger import setup_logging
from src.config import config
from loguru import logger


def run_crawler(args):
    """Run the web crawler"""
    logger.info("Starting crawler...")
    from src.utils.logger import setup_logging as _setup_logging
    from datetime import datetime

    # Create a per-crawl log file with timestamp for better traceability
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    crawl_log_file = f"logs/crawl_{args.spider}_{timestamp}.log"
    # Re-configure logging to add a file handler for this crawl
    _setup_logging(config.log_level, crawl_log_file)
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings
    
    # Get spider class
    if args.spider == 'iisc':
        from src.crawler.spiders.iisc_spider import IIScSpider
        spider = IIScSpider
    elif args.spider == 'generic':
        from src.crawler.spiders.generic_spider import GenericSpider
        spider = GenericSpider
    else:
        logger.error(f"Unknown spider: {args.spider}")
        return
    
    # Ensure logs directory exists
    from pathlib import Path
    Path('logs').mkdir(exist_ok=True)
    
    # Set up crawler
    settings = get_project_settings()
    settings.update({
        'SPIDER_MODULES': ['src.crawler.spiders'],
        'NEWSPIDER_MODULE': 'src.crawler.spiders',
        'LOG_FILE': crawl_log_file,
        'LOG_ENABLED': True,
        'LOG_LEVEL': config.log_level,
        'LOG_STDOUT': True,
    })
    
    # Add checkpoint settings (resume support)
    if hasattr(args, 'resume') and args.resume:
        logger.info("✓ RESUME MODE: Loading checkpoint from previous crawl")
        # Checkpoint system is automatic - just inform user
    else:
        logger.info("✓ FRESH CRAWL: Starting from beginning (checkpoints will be saved)")
        # Optional: Clear old checkpoints if fresh start is explicitly requested
        if hasattr(args, 'fresh') and args.fresh:
            import shutil
            checkpoint_dir = Path('data/checkpoints')
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
                logger.info("Cleared old checkpoints for fresh start")
    
    process = CrawlerProcess(settings)
    
    # Start crawler
    if args.url:
        process.crawl(spider, start_url=args.url)
    else:
        process.crawl(spider)
    
    process.start()
    logger.info("Crawler finished")


def run_nlp_processing(args):
    """Run NLP processing on crawled data"""
    logger.info("Starting NLP processing...")
    
    from src.nlp.pipeline import NLPPipeline
    
    pipeline = NLPPipeline()
    
    if args.input:
        # If no output specified, let pipeline auto-generate path in data/ dir
        # If output is specified, use it as-is
        output = args.output if args.output else None
        
        # Check if parallel processing is requested
        use_parallel = config.get('nlp', 'processing', 'use_parallel', default=True)
        
        if hasattr(args, 'no_parallel') and args.no_parallel:
            use_parallel = False
        
        # Check if resume mode is requested
        resume = getattr(args, 'resume', False)
        
        if use_parallel and not (hasattr(args, 'no_parallel') and args.no_parallel):
            # Use parallel processing
            workers = getattr(args, 'workers', None)
            if workers is None:
                workers_config = config.get('nlp', 'processing', 'workers', default='auto')
                if workers_config == 'auto':
                    from multiprocessing import cpu_count
                    workers = max(1, cpu_count() - 2)
                else:
                    workers = int(workers_config)
            
            logger.info(f"Using parallel processing with {workers} workers")
            if resume:
                logger.info("Resume mode enabled - will skip already processed pages")
            pipeline.process_from_file_parallel(args.input, output, num_workers=workers, resume=resume)
        else:
            # Use single-process
            logger.info("Using single-process mode")
            if resume:
                logger.info("Resume mode enabled - will skip already processed pages")
            pipeline.process_from_file(args.input, output, resume=resume)
    else:
        logger.error("Input file required for NLP processing")


def run_database_import(args):
    """Import processed data into databases using parallel processing"""
    logger.info("Starting parallel database import...")
    
    from src.database.import_parallel import ParallelDatabaseImporter
    
    # Configuration
    chromadb_config = {
        'persist_directory': config.chromadb_persist_dir,
        'collection_name': config.get('database', 'chromadb', 'collection_name', default='iisc_research_docs')
    }
    
    neo4j_config = {
        'uri': config.neo4j_uri,
        'username': config.neo4j_username,
        'password': config.neo4j_password
    }
    
    # Create importer
    importer = ParallelDatabaseImporter(chromadb_config, neo4j_config)
    
    if args.input:
        # Get batch size and workers from args or config
        batch_size = getattr(args, 'batch_size', 200)
        workers = getattr(args, 'workers', None)
        
        # Import to both databases by default
        import_chromadb = not getattr(args, 'skip_chromadb', False)
        import_neo4j = not getattr(args, 'skip_neo4j', False)
        
        importer.import_from_file(
            args.input,
            batch_size=batch_size,
            num_workers=workers,
            import_chromadb=import_chromadb,
            import_neo4j=import_neo4j
        )
        
        logger.info("Database import complete!")
    else:
        logger.error("Input file required for database import")


def run_chatbot(args):
    """Run the RAG chatbot"""
    logger.info("Starting RAG chatbot...")
    
    from src.rag.chatbot import RAGChatbot
    
    chatbot = RAGChatbot()
    
    if args.query:
        # Single query mode
        response = chatbot.chat(args.query)
        print("\n" + "="*80)
        print(f"Query: {response['query']}")
        print("-"*80)
        print(f"Response: {response['response']}")
        
        if 'sources' in response:
            print("-"*80)
            print("Sources:")
            for i, source in enumerate(response['sources'], 1):
                print(f"{i}. {source['title']} ({source['url']})")
        print("="*80 + "\n")
    
    elif args.interactive:
        # Interactive mode
        print("\n" + "="*80)
        print("IISc Research Chatbot - Interactive Mode")
        print("Type 'exit' or 'quit' to end the conversation")
        print("="*80 + "\n")
        
        conversation_history = []
        
        while True:
            try:
                query = input("You: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break
                
                response = chatbot.chat(query, conversation_history=conversation_history)
                
                print(f"\nBot: {response['response']}\n")
                
                if 'sources' in response and response['sources']:
                    print("Sources:")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"  {i}. {source['title']} - {source['url']}")
                    print()
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": query})
                conversation_history.append({"role": "assistant", "content": response['response']})
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}\n")
    
    else:
        logger.error("Either --query or --interactive must be specified")


def run_improved_chatbot(args):
    """Run the improved RAG chatbot with enhanced retrieval and reranking"""
    logger.info("Starting Improved RAG chatbot...")
    
    from src.rag.chatbot import ImprovedRAGChatbot
    
    chatbot = ImprovedRAGChatbot()
    
    if args.query:
        # Single query mode
        response = chatbot.chat(args.query, return_debug_info=args.debug)
        
        print("\n" + "="*80)
        print(f"Query: {response['query']}")
        print("-"*80)
        print(f"Response: {response['response']}")
        
        if args.debug and 'debug' in response:
            print("-"*80)
            print("Debug Info:")
            debug = response['debug']
            print(f"  Retrieved: {debug['raw_retrieval_count']} documents")
            print(f"  Reranked: {debug['reranked_count']} documents")
            print(f"  Context chunks used: {debug['context_chunks']}")
            if debug['top_result']['url']:
                print(f"  Top result: {debug['top_result']['url']}")
                print(f"  Score: {debug['top_result']['score']:.3f}")
                print(f"  Found by: {', '.join(debug['top_result']['methods'])}")
        
        if 'sources' in response and response['sources']:
            print("-"*80)
            print("Sources:")
            for i, source in enumerate(response['sources'], 1):
                print(f"  {i}. [{source['page_type']}] {source['title']}")
                print(f"     {source['url']}")
                if 'ranking_reason' in source:
                    print(f"     Why: {source['ranking_reason']}")
        
        print("="*80 + "\n")
    
    elif args.interactive:
        # Interactive mode
        chatbot.interactive_chat()
    
    else:
        logger.error("Either --query or --interactive must be specified")


def run_full_pipeline(args):
    """Run the complete pipeline: crawl -> process -> import"""
    logger.info("Starting full pipeline...")
    
    # Step 1: Crawl
    if not args.skip_crawl:
        logger.info("Step 1/3: Crawling...")
        run_crawler(args)
    
    # Step 2: NLP Processing
    if not args.skip_nlp:
        logger.info("Step 2/3: NLP Processing...")
        # Find latest crawled file
        data_dir = Path("data/crawled_pages")
        if data_dir.exists():
            files = sorted(data_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
            if files:
                args.input = str(files[0])
                args.output = str(data_dir / f"processed_{files[0].name}")
                run_nlp_processing(args)
    
    # Step 3: Database Import
    if not args.skip_import:
        logger.info("Step 3/3: Database Import...")
        if hasattr(args, 'output') and args.output:
            args.input = args.output
            run_database_import(args)
    
    logger.info("Full pipeline complete!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="IISc Research Chatbot - AI-powered research assistant for IISc"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Crawler command
    crawler_parser = subparsers.add_parser('crawl', help='Run web crawler')
    crawler_parser.add_argument('--spider', choices=['iisc', 'generic'], default='iisc',
                               help='Spider to use')
    crawler_parser.add_argument('--url', help='Start URL for crawling')
    crawler_parser.add_argument('--resume', action='store_true',
                               help='Resume from checkpoint (auto-detects saved state)')
    crawler_parser.add_argument('--fresh', action='store_true',
                               help='Force fresh start (clears checkpoints)')
    
    # NLP processing command
    nlp_parser = subparsers.add_parser('process', help='Run NLP processing')
    nlp_parser.add_argument('--input', required=True, help='Input JSONL file')
    nlp_parser.add_argument('--output', help='Output JSONL file')
    nlp_parser.add_argument('--workers', type=int, help='Number of parallel workers (default: auto-detect)')
    nlp_parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    nlp_parser.add_argument('--resume', action='store_true', help='Resume from where processing stopped (skips already processed pages)')
    
    # Database import command
    db_parser = subparsers.add_parser('import', help='Import data to databases (parallel)')
    db_parser.add_argument('--input', required=True, help='Input processed JSONL file')
    db_parser.add_argument('--batch-size', type=int, default=200, help='Batch size for import (default: 200)')
    db_parser.add_argument('--workers', type=int, help='Number of parallel workers (default: auto-detect)')
    db_parser.add_argument('--skip-chromadb', action='store_true', help='Skip ChromaDB import')
    db_parser.add_argument('--skip-neo4j', action='store_true', help='Skip Neo4j import')
    
    # Chatbot command
    chat_parser = subparsers.add_parser('chat', help='Run chatbot (original)')
    chat_parser.add_argument('--query', help='Single query to process')
    chat_parser.add_argument('--interactive', action='store_true',
                            help='Run in interactive mode')
    
    # Improved Chatbot command (NEW - Enhanced retrieval & reranking)
    improved_chat_parser = subparsers.add_parser('chat-improved', 
                                                  help='Run improved chatbot (enhanced retrieval & reranking)')
    improved_chat_parser.add_argument('--query', '-q', help='Single query to process')
    improved_chat_parser.add_argument('--interactive', '-i', action='store_true',
                                     help='Run in interactive mode')
    improved_chat_parser.add_argument('--debug', '-d', action='store_true',
                                     help='Show debug info (retrieval stats, sources, etc.)')
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--spider', choices=['iisc', 'generic'], default='iisc')
    pipeline_parser.add_argument('--url', help='Start URL for crawling')
    pipeline_parser.add_argument('--skip-crawl', action='store_true',
                                help='Skip crawling step')
    pipeline_parser.add_argument('--skip-nlp', action='store_true',
                               help='Skip NLP processing step')
    pipeline_parser.add_argument('--skip-import', action='store_true',
                                help='Skip database import step')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = config.log_level
    log_file = "logs/iisc_chatbot.log"
    setup_logging(log_level, log_file)
    
    # Run command
    if args.command == 'crawl':
        run_crawler(args)
    elif args.command == 'process':
        run_nlp_processing(args)
    elif args.command == 'import':
        run_database_import(args)
    elif args.command == 'chat':
        run_chatbot(args)
    elif args.command == 'chat-improved':
        run_improved_chatbot(args)
    elif args.command == 'pipeline':
        run_full_pipeline(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
