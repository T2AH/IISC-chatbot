"""
NLP Pipeline orchestration
Combines all NLP components for end-to-end processing
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
from loguru import logger

from src.nlp.text_processing import TextProcessor
from src.nlp.entity_extraction import EntityExtractor
from src.nlp.keyword_extraction import KeywordExtractor
from src.nlp.embedding_generation import EmbeddingGenerator
from src.config import config


# Global worker pipeline - initialized once per worker process
worker_pipeline = None


def init_worker():
    """Initialize models in worker process - must be at module level for Windows"""
    global worker_pipeline
    try:
        worker_pipeline = NLPPipeline()
        logger.info("Worker initialized successfully")
    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")
        raise


def process_page_worker(page_data):
    """Process a single page in worker - must be at module level for Windows"""
    import time
    
    start_time = time.time()
    
    try:
        # Limit text size to prevent memory issues (reduced for faster processing)
        if 'cleaned_text' in page_data:
            original_len = len(page_data['cleaned_text'])
            if original_len > 100000:  # 100K chars max (academic pages rarely exceed this)
                page_data['cleaned_text'] = page_data['cleaned_text'][:100000]
                page_data['text_truncated'] = True
                page_data['original_length'] = original_len
        
        # Process the page
        result = worker_pipeline.process_page(page_data)
        
        elapsed = time.time() - start_time
        
        # Return result with timing
        result['processing_time'] = elapsed
        return result
    
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Worker error processing page (after {elapsed:.1f}s): {e}")
        page_data['nlp_error'] = str(e)
        page_data['processing_time'] = elapsed
        return page_data


class NLPPipeline:
    """End-to-end NLP pipeline for processing crawled pages"""
    
    def __init__(self):
        """Initialize NLP pipeline with all components"""
        logger.info("Initializing NLP Pipeline...")
        
        # Load configuration
        self.config = config
        
        # Initialize components
        chunk_size = self.config.get('nlp', 'text_processing', 'chunk_size', default=250)
        chunk_overlap = self.config.get('nlp', 'text_processing', 'chunk_overlap', default=50)
        min_chunk_size = self.config.get('nlp', 'text_processing', 'min_chunk_size', default=50)
        
        self.text_processor = TextProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )
        
        spacy_model = self.config.get('nlp', 'models', 'spacy_model', default='en_core_web_lg')
        self.entity_extractor = EntityExtractor(spacy_model=spacy_model)
        
        keybert_model = self.config.get('nlp', 'models', 'keybert_model', default='all-MiniLM-L6-v2')
        top_n = self.config.get('nlp', 'keyword_extraction', 'top_n', default=10)
        diversity = self.config.get('nlp', 'keyword_extraction', 'diversity', default=0.5)
        
        self.keyword_extractor = KeywordExtractor(
            model_name=keybert_model,
            top_n=top_n,
            diversity=diversity
        )
        
        embedding_model = self.config.get('nlp', 'models', 'embedding_model', default='sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        
        # Optional: Enhanced models for better results
        self.use_enhanced_models = self.config.get('nlp', 'enhanced_models', 'enabled', default=False)
        self.sci_ner = None
        self.yake_extractor = None
        
        if self.use_enhanced_models:
            self._load_enhanced_models()
        
        logger.info("NLP Pipeline initialized successfully")
    
    def _load_enhanced_models(self):
        """Load additional NLP models for enhanced processing"""
        logger.info("Loading enhanced NLP models...")
        
        # 1. Scientific NER for better academic entity extraction
        try:
            from transformers import pipeline
            self.sci_ner = pipeline(
                "ner",
                model="allenai/scibert_scivocab_uncased",
                aggregation_strategy="simple"
            )
            logger.info("✓ Loaded SciBERT for scientific entity extraction")
        except Exception as e:
            logger.warning(f"Failed to load SciBERT: {e}")
        
        # 2. YAKE for unsupervised keyword extraction
        try:
            import yake
            self.yake_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,  # max ngram size
                dedupLim=0.9,
                top=10,
                features=None
            )
            logger.info("✓ Loaded YAKE for enhanced keyword extraction")
        except Exception as e:
            logger.warning(f"Failed to load YAKE: {e}")
    
    def process_page(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single page through the NLP pipeline
        
        Args:
            page_data: Dictionary with page information (must include 'cleaned_text')
        
        Returns:
            Processed page data with entities, keywords, chunks, and embeddings
        """
        try:
            page_id = page_data.get('page_id', 'unknown')
            logger.info(f"Processing page: {page_id}")
            
            # Get cleaned text
            text = page_data.get('cleaned_text', '')
            if not text:
                logger.warning(f"No cleaned text found for page {page_id}")
                return page_data
            
            # 1. Get text statistics
            text_stats = self.text_processor.get_text_stats(text)
            page_data['text_stats'] = text_stats
            
            # 2. Extract entities
            entities = self.entity_extractor.extract_entities(text)
            page_data['entities'] = entities
            
            # 2b. Enhanced: Extract scientific entities if enabled
            if self.sci_ner:
                try:
                    sci_entities = self.sci_ner(text[:512])  # Limit to 512 tokens
                    scientific_terms = [
                        {'term': ent['word'], 'type': ent['entity_group'], 'score': ent['score']}
                        for ent in sci_entities if ent['score'] > 0.7
                    ]
                    page_data['scientific_entities'] = scientific_terms
                except Exception as e:
                    logger.warning(f"Scientific NER failed: {e}")
            
            # 3. Extract keywords
            keywords = self.keyword_extractor.extract_keywords(text)
            page_data['keywords'] = [{'keyword': kw, 'score': score} for kw, score in keywords]
            
            # 3b. Enhanced: Extract additional keywords with YAKE
            if self.yake_extractor:
                try:
                    yake_keywords = self.yake_extractor.extract_keywords(text)
                    page_data['yake_keywords'] = [
                        {'keyword': kw, 'score': score} 
                        for kw, score in yake_keywords[:10]
                    ]
                except Exception as e:
                    logger.warning(f"YAKE extraction failed: {e}")
            
            # 4. Chunk text
            metadata = {
                'page_id': page_data.get('page_id'),
                'url': page_data.get('url'),
                'domain': page_data.get('domain'),
                'page_type': page_data.get('page_type'),
                'title': page_data.get('title')
            }
            
            chunks = self.text_processor.chunk_text(text, metadata=metadata)
            
            # 5. Generate embeddings for chunks
            if chunks:
                chunks = self.embedding_generator.generate_chunk_embeddings(chunks)
            
            page_data['chunks'] = chunks
            
            # 6. Add processing metadata
            page_data['nlp_processed_date'] = datetime.now().isoformat()
            page_data['embedding_dimension'] = self.embedding_generator.get_embedding_dimension()
            
            logger.info(f"Successfully processed page {page_id}: "
                       f"{len(entities)} entity types, "
                       f"{len(keywords)} keywords, "
                       f"{len(chunks)} chunks")
            
            return page_data
        
        except Exception as e:
            logger.error(f"Error processing page {page_data.get('page_id', 'unknown')}: {e}")
            return page_data
    
    def process_batch(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple pages
        
        Args:
            pages: List of page dictionaries
        
        Returns:
            List of processed page dictionaries
        """
        logger.info(f"Processing batch of {len(pages)} pages")
        
        processed_pages = []
        for i, page in enumerate(pages, 1):
            logger.info(f"Processing page {i}/{len(pages)}")
            processed_page = self.process_page(page)
            processed_pages.append(processed_page)
        
        logger.info(f"Batch processing complete: {len(processed_pages)} pages processed")
        return processed_pages
    
    def process_from_file(self, input_file: str, output_file: str = None, resume: bool = False):
        """
        Process pages from a JSONL file
        
        Args:
            input_file: Path to input JSONL file with crawled pages
            output_file: Path to output JSONL file (optional)
            resume: If True, skip already processed pages (default: False)
        """
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_file}")
            return
        
        logger.info(f"Processing pages from file: {input_file}")
        print(f"\n✓ Starting NLP processing: {input_file}\n", flush=True)
        
        # Count total lines for progress
        total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
        print(f"Total pages to process: {total_lines}\n", flush=True)
        
        # Prepare output file - store in data/ directory
        if output_file is None:
            # Get project root and create data/ subdirectory
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(data_dir / f"processed_{input_path.name}")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output file: {output_file}\n", flush=True)
        
        # Load already processed page IDs if resuming
        processed_page_ids = set()
        if resume and output_path.exists():
            logger.info("Resume mode: Loading already processed pages...")
            print(f"✓ Resume mode enabled - checking existing output file\n", flush=True)
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            page = json.loads(line)
                            page_id = page.get('page_id') or page.get('url')
                            if page_id:
                                processed_page_ids.add(page_id)
                        except:
                            continue
                logger.info(f"Found {len(processed_page_ids)} already processed pages")
                print(f"✓ Found {len(processed_page_ids)} already processed pages - will skip these\n", flush=True)
            except Exception as e:
                logger.warning(f"Could not load processed pages: {e}")
                print(f"⚠ Warning: Could not load processed pages: {e}\n", flush=True)
        
        # Process pages
        processed_count = 0
        skipped_count = 0
        file_mode = 'a' if (resume and output_path.exists()) else 'w'
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, file_mode, encoding='utf-8', buffering=1) as outfile:
            
            for line in infile:
                try:
                    page_data = json.loads(line)
                    page_id = page_data.get('page_id') or page_data.get('url')
                    
                    # Skip if already processed
                    if resume and page_id in processed_page_ids:
                        skipped_count += 1
                        if skipped_count % 100 == 0:
                            print(f"Skipped {skipped_count} already processed pages...", flush=True)
                        continue
                    
                    processed_page = self.process_page(page_data)
                    
                    # Write to output file and flush immediately
                    outfile.write(json.dumps(processed_page, ensure_ascii=False) + '\n')
                    outfile.flush()
                    processed_count += 1
                    
                    # Real-time progress output
                    url = page_data.get('url', 'unknown')[:70]
                    entities_count = len(processed_page.get('entities', {}))
                    keywords_count = len(processed_page.get('keywords', []))
                    chunks_count = len(processed_page.get('chunks', []))
                    
                    remaining = total_lines - processed_count - skipped_count
                    print(f"[{processed_count + skipped_count}/{total_lines}] {url}... | "
                          f"E:{entities_count} K:{keywords_count} C:{chunks_count} | Remaining: {remaining}", 
                          flush=True)
                
                except Exception as e:
                    logger.error(f"Error processing line: {e}")
                    print(f"✗ Error processing page: {e}", flush=True)
        
        logger.info(f"Processed {processed_count} pages. Output saved to: {output_file}")
        print(f"\n✓ Processing complete!", flush=True)
        if resume:
            print(f"  - Skipped: {skipped_count} already processed pages", flush=True)
            print(f"  - Newly processed: {processed_count} pages", flush=True)
            print(f"  - Total in output: {processed_count + skipped_count} pages", flush=True)
        else:
            print(f"  - Processed: {processed_count} pages", flush=True)
        print(f"✓ Output saved to: {output_file}\n", flush=True)
    
    def process_from_file_parallel(self, input_file: str, output_file: str = None, num_workers: int = None, resume: bool = False):
        """
        Process pages from a JSONL file using multiple worker processes
        
        Args:
            input_file: Path to input JSONL file with crawled pages
            output_file: Path to output JSONL file (optional)
            num_workers: Number of worker processes (default: cpu_count - 2, min 1)
            resume: If True, skip already processed pages (default: False)
        """
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_file}")
            return
        
        # Determine worker count - optimized for 10-core CPU with 16GB RAM
        if num_workers is None:
            num_workers = max(1, cpu_count() - 2)
        # Cap at 14 workers for maximum throughput (allows hyperthreading utilization)
        num_workers = max(1, min(num_workers, 14))
        
        logger.info(f"Processing pages from file: {input_file}")
        logger.info(f"Using {num_workers} worker processes")
        print(f"\n{'='*60}")
        print(f"Starting Parallel NLP Processing")
        print(f"{'='*60}")
        print(f"Input: {input_file}")
        print(f"Workers: {num_workers}")
        
        # Prepare output file - store in data/ directory
        if output_file is None:
            # Get project root and create data/ subdirectory
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(data_dir / f"processed_{input_path.name}")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_file}")
        
        # Load already processed page IDs if resuming (fast, only IDs)
        processed_page_ids = set()
        if resume and output_path.exists():
            logger.info("Resume mode: Loading processed page IDs...")
            print(f"Resume mode: Enabled")
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            page = json.loads(line)
                            page_id = page.get('page_id') or page.get('url')
                            if page_id:
                                processed_page_ids.add(page_id)
                        except:
                            continue
                logger.info(f"Found {len(processed_page_ids)} already processed pages")
                print(f"Already processed: {len(processed_page_ids)} pages")
            except Exception as e:
                logger.warning(f"Could not load processed pages: {e}")
                print(f"Warning: Could not load processed pages: {e}")
        
        # Load pages with streaming filter (memory efficient)
        pages = []
        skipped_count = 0
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    page = json.loads(line)
                    page_id = page.get('page_id') or page.get('url')
                    
                    # Skip if already processed
                    if resume and page_id in processed_page_ids:
                        skipped_count += 1
                        continue
                    
                    pages.append(page)
                except Exception as e:
                    logger.error(f"Error loading page: {e}")
        
        total_pages = len(pages)
        print(f"Total pages to process: {total_pages}")
        if resume and skipped_count > 0:
            print(f"Skipped {skipped_count} already processed pages")
        
        print(f"{'='*60}\n", flush=True)
        
        # Process in parallel with real-time output and optimized batching
        manager = Manager()
        progress_dict = manager.dict()
        progress_dict['count'] = 0
        progress_dict['total'] = len(pages)
        progress_dict['already_processed'] = len(processed_page_ids)
        
        # Progress callback for real-time updates
        def update_progress(result):
            """Update progress counter and print status"""
            progress_dict['count'] += 1
            count = progress_dict['count']
            total = progress_dict['total']
            
            url = result.get('url', 'unknown')[:60]
            entities_count = len(result.get('entities', {}))
            keywords_count = len(result.get('keywords', []))
            chunks_count = len(result.get('chunks', []))
            elapsed = result.get('processing_time', 0)
            
            print(f"[{count}/{total}] {url}... | "
                  f"E:{entities_count} K:{keywords_count} C:{chunks_count} | {elapsed:.1f}s", 
                  flush=True)
        
        # Process in parallel
        try:
            processed_count = 0
            
            # Create and open output file BEFORE starting processing
            file_mode = 'a' if (resume and output_path.exists()) else 'w'
            action = "Appending to" if file_mode == 'a' else "Creating"
            print(f"\n✓ {action} output file: {output_file}")
            outfile = open(output_path, file_mode, encoding='utf-8', buffering=1)
            print(f"✓ Output file ready for writing\n", flush=True)
            
            # Use optimal chunksize for better performance (4 pages per task for efficiency)
            optimal_chunksize = max(2, min(4, total_pages // (num_workers * 8)))
            
            with Pool(processes=num_workers, initializer=init_worker, maxtasksperchild=100) as pool:
                # Use imap_unordered for real-time results as they complete
                # maxtasksperchild=100 balances memory management and worker restart overhead
                for processed_page in pool.imap_unordered(process_page_worker, pages, chunksize=optimal_chunksize):
                    # Update progress display
                    update_progress(processed_page)
                    
                    # Write immediately to file
                    outfile.write(json.dumps(processed_page, ensure_ascii=False) + '\n')
                    outfile.flush()
                    os.fsync(outfile.fileno())  # Always fsync for crash safety
                    
                    processed_count += 1
            
            # Close output file after all processing
            outfile.close()
            
            logger.info(f"Processed {processed_count} pages. Output saved to: {output_file}")
            print(f"\n{'='*60}")
            print(f"✓ Processing Complete!")
            print(f"{'='*60}")
            if resume and len(processed_page_ids) > 0:
                print(f"Previously processed: {len(processed_page_ids)} pages")
                print(f"Newly processed: {processed_count} pages")
                print(f"Total in output: {processed_count + len(processed_page_ids)} pages")
            else:
                print(f"Pages processed: {processed_count}")
            print(f"Output saved to: {output_file}")
            print(f"{'='*60}\n", flush=True)
        
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            print(f"\n✗ Parallel processing failed: {e}")
            print("Falling back to single-process mode...\n", flush=True)
            # Fallback to single-process
            self.process_from_file(input_file, output_file, resume=resume)
