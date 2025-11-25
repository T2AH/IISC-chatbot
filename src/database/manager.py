"""
Database manager for orchestrating Neo4j and ChromaDB operations
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from src.database.neo4j_client import Neo4jClient
from src.database.chromadb_client import ChromaDBClient


class DatabaseManager:
    """Manager for coordinating Neo4j and ChromaDB operations"""
    
    def __init__(self):
        """Initialize database manager with both clients"""
        logger.info("Initializing Database Manager...")
        
        self.neo4j = Neo4jClient()
        self.chromadb = ChromaDBClient()
        
        # Initialize Neo4j schema
        if self.neo4j.driver:
            self.neo4j.create_indices()
            self.neo4j.create_constraints()
        
        logger.info("Database Manager initialized")
    
    def store_processed_page(self, page_data: Dict[str, Any]) -> bool:
        """
        Store processed page in both databases
        
        Args:
            page_data: Processed page data with entities, keywords, and chunks
        
        Returns:
            True if successful
        """
        try:
            page_id = page_data.get('page_id', 'unknown')
            logger.info(f"Storing page {page_id} in databases")
            
            # 1. Store entities in Neo4j knowledge graph
            if self.neo4j.driver:
                success_neo4j = self.neo4j.store_page_entities(page_data)
                if not success_neo4j:
                    logger.warning(f"Failed to store entities in Neo4j for page {page_id}")
            
            # 2. Store chunks and embeddings in ChromaDB
            chunks = page_data.get('chunks', [])
            if chunks and self.chromadb.collection:
                success_chroma = self.chromadb.add_documents(chunks)
                if not success_chroma:
                    logger.warning(f"Failed to store chunks in ChromaDB for page {page_id}")
            
            logger.info(f"Successfully stored page {page_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error storing page: {e}")
            return False
    
    def store_batch(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store multiple processed pages
        
        Args:
            pages: List of processed page dictionaries
        
        Returns:
            Dictionary with success statistics
        """
        logger.info(f"Storing batch of {len(pages)} pages")
        
        success_count = 0
        failed_count = 0
        
        for page in pages:
            if self.store_processed_page(page):
                success_count += 1
            else:
                failed_count += 1
        
        stats = {
            'total': len(pages),
            'success': success_count,
            'failed': failed_count
        }
        
        logger.info(f"Batch storage complete: {success_count} succeeded, {failed_count} failed")
        return stats
    
    def store_from_file(self, input_file: str) -> Dict[str, Any]:
        """
        Store processed pages from a JSONL file
        
        Args:
            input_file: Path to JSONL file with processed pages
        
        Returns:
            Dictionary with storage statistics
        """
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_file}")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        logger.info(f"Storing pages from file: {input_file}")

        # Batch parameters
        batch_pages: List[Dict[str, Any]] = []
        batch_chunk_threshold = 1024  # flush when accumulated chunks reach this
        batch_page_threshold = 200    # or flush when this many pages accumulated

        accumulated_chunks = 0
        total_pages = 0
        success_count = 0
        failed_count = 0

        def flush_batch(pages_batch: List[Dict[str, Any]]) -> Dict[str, int]:
            """Flush a batch of pages to ChromaDB and Neo4j in bulk."""
            nonlocal success_count, failed_count

            if not pages_batch:
                return {'success': 0, 'failed': 0}

            # 1) Aggregate chunks for ChromaDB bulk add
            all_chunks: List[Dict[str, Any]] = []
            for p in pages_batch:
                chunks = p.get('chunks', [])
                # annotate chunk-level metadata if missing
                for c in chunks:
                    if 'page_id' not in c:
                        c['page_id'] = p.get('page_id', '')
                    if 'url' not in c:
                        c['url'] = p.get('url', '')
                    if 'domain' not in c:
                        c['domain'] = p.get('domain', '')
                    if 'page_type' not in c:
                        c['page_type'] = p.get('page_type', '')
                all_chunks.extend(chunks)

            chroma_ok = True
            try:
                if all_chunks and self.chromadb.collection:
                    # Single bulk add for the entire batch
                    chroma_ok = self.chromadb.add_documents(all_chunks)
            except Exception as e:
                logger.error(f"ChromaDB bulk add failed: {e}")
                chroma_ok = False

            # 2) Bulk store entities in Neo4j using a batched UNWIND approach
            neo4j_ok = True
            try:
                if self.neo4j.driver:
                    self.neo4j.store_pages_entities_batch(pages_batch)
            except Exception as e:
                logger.error(f"Neo4j bulk store failed: {e}")
                neo4j_ok = False

            # Update counts based on success
            if chroma_ok and neo4j_ok:
                success_count += len(pages_batch)
            else:
                failed_count += len(pages_batch)

            return {'success': success_count, 'failed': failed_count}

        # Stream and batch
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_pages += 1
                try:
                    page_data = json.loads(line)
                    batch_pages.append(page_data)
                    accumulated_chunks += len(page_data.get('chunks', []))

                    # Flush conditions
                    if accumulated_chunks >= batch_chunk_threshold or len(batch_pages) >= batch_page_threshold:
                        flush_batch(batch_pages)
                        batch_pages = []
                        accumulated_chunks = 0

                except Exception as e:
                    logger.error(f"Error processing line: {e}")
                    failed_count += 1

        # Flush remaining pages
        if batch_pages:
            flush_batch(batch_pages)

        stats = {
            'total': total_pages,
            'success': success_count,
            'failed': failed_count
        }

        logger.info(f"File storage complete: {success_count} succeeded, {failed_count} failed")
        return stats
    
    def search_similar(self, query: str, n_results: int = 50, 
                      filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using ChromaDB
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Optional metadata filters
        
        Returns:
            List of similar documents with metadata
        """
        if not self.chromadb.collection:
            logger.error("ChromaDB not available")
            return []
        
        try:
            results = self.chromadb.query(
                query_text=query,
                n_results=n_results,
                where=filters
            )
            
            # Format results
            documents = []
            for i in range(len(results['documents'][0])):
                doc = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                }
                documents.append(doc)
            
            return documents
        
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def search_knowledge_graph(self, query_type: str, query_value: str) -> Any:
        """
        Search knowledge graph in Neo4j
        
        Args:
            query_type: Type of query ('faculty', 'topic', 'lab')
            query_value: Value to search for
        
        Returns:
            Query results
        """
        if not self.neo4j.driver:
            logger.error("Neo4j not available")
            return None
        
        try:
            if query_type == 'faculty':
                return self.neo4j.query_faculty(query_value)
            elif query_type == 'topic':
                return self.neo4j.search_by_topic(query_value)
            else:
                logger.warning(f"Unknown query type: {query_type}")
                return None
        
        except Exception as e:
            logger.error(f"Error searching knowledge graph: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from both databases
        
        Returns:
            Dictionary with database statistics
        """
        stats = {
            'chromadb': self.chromadb.get_collection_stats() if self.chromadb.collection else {},
            'neo4j': {
                'connected': self.neo4j.driver is not None
            }
        }
        
        return stats
    
    def close(self):
        """Close all database connections"""
        if self.neo4j:
            self.neo4j.close()
        
        logger.info("All database connections closed")
