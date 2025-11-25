"""
ChromaDB client for vector database storage
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from loguru import logger

from src.config import config


class ChromaDBClient:
    """Client for interacting with ChromaDB vector database"""
    
    def __init__(self, persist_directory: str = None, collection_name: str = None):
        """
        Initialize ChromaDB client
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory or config.chromadb_persist_dir
        self.collection_name = collection_name or config.get(
            'database', 'chromadb', 'collection_name', 
            default='iisc_research_docs'
        )
        
        self.client = None
        self.collection = None
        self._connect()
    
    def _connect(self):
        """Connect to ChromaDB"""
        try:
            # Create persistent client with settings
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=settings
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "IISc research documents and embeddings"}
            )
            
            logger.info(f"Connected to ChromaDB. Collection: {self.collection_name}")
            logger.info(f"Collection size: {self.collection.count()} documents")
        
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def add_documents(self, documents: List[str] = None, embeddings: List[List[float]] = None,
                     metadatas: List[Dict[str, Any]] = None, ids: List[str] = None,
                     chunks: List[Dict[str, Any]] = None) -> bool:
        """
        Add document chunks with embeddings to the collection
        Supports both direct parameters and chunks format for flexibility
        
        Args:
            documents: List of document texts (direct format)
            embeddings: List of embedding vectors (direct format)
            metadatas: List of metadata dicts (direct format)
            ids: List of document IDs (direct format)
            chunks: List of chunk dictionaries with embeddings (legacy format)
        
        Returns:
            True if successful
        """
        if not self.collection:
            logger.error("No ChromaDB collection available")
            return False
        
        try:
            # Handle legacy chunks format
            if chunks:
                ids = []
                embeddings = []
                documents = []
                metadatas = []
                
                for chunk in chunks:
                    # Generate unique ID
                    chunk_id = f"{chunk.get('page_id', 'unknown')}_{chunk.get('chunk_id', 0)}"
                    ids.append(chunk_id)
                    
                    # Get embedding
                    embedding = chunk.get('embedding', [])
                    if not embedding:
                        logger.warning(f"No embedding found for chunk {chunk_id}")
                        continue
                    
                    embeddings.append(embedding)
                    
                    # Get text
                    documents.append(chunk.get('text', ''))
                    
                    # Prepare metadata
                    metadata = {
                        'page_id': chunk.get('page_id', ''),
                        'url': chunk.get('url', ''),
                        'domain': chunk.get('domain', ''),
                        'page_type': chunk.get('page_type', ''),
                        'title': chunk.get('title', ''),
                        'chunk_id': chunk.get('chunk_id', 0),
                        'word_count': chunk.get('word_count', 0)
                    }
                    metadatas.append(metadata)
            
            # Add to collection (batch insert)
            if ids and documents and embeddings:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                
                logger.info(f"Added {len(ids)} chunks to ChromaDB")
                return True
            else:
                logger.warning("No valid chunks to add")
                return False
        
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            return False
    
    def query(self, query_text: str = None, query_embedding: List[float] = None, 
              n_results: int = 50, where: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Query the collection for similar documents
        
        Args:
            query_text: Query text (will be embedded automatically)
            query_embedding: Pre-computed query embedding
            n_results: Number of results to return
            where: Optional metadata filters
        
        Returns:
            Query results dictionary
        """
        if not self.collection:
            logger.error("No ChromaDB collection available")
            return {'documents': [], 'metadatas': [], 'distances': []}
        
        try:
            # Query the collection
            if query_embedding:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where
                )
            elif query_text:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where
                )
            else:
                logger.error("Either query_text or query_embedding must be provided")
                return {'documents': [], 'metadatas': [], 'distances': []}
            
            logger.debug(f"Query returned {len(results['documents'][0])} results")
            return results
        
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return {'documents': [], 'metadatas': [], 'distances': []}
    
    def query_by_metadata(self, filters: Dict[str, Any], n_results: int = 50) -> List[Dict[str, Any]]:
        """
        Query documents by metadata filters
        
        Args:
            filters: Metadata filters (e.g., {'page_type': 'faculty'})
            n_results: Maximum number of results
        
        Returns:
            List of matching documents with metadata
        """
        if not self.collection:
            logger.error("No ChromaDB collection available")
            return []
        
        try:
            # Get all documents matching filter
            results = self.collection.get(
                where=filters,
                limit=n_results
            )
            
            # Format results
            documents = []
            for i in range(len(results['ids'])):
                doc = {
                    'id': results['ids'][i],
                    'document': results['documents'][i] if 'documents' in results else None,
                    'metadata': results['metadatas'][i] if 'metadatas' in results else {}
                }
                documents.append(doc)
            
            logger.debug(f"Retrieved {len(documents)} documents by metadata")
            return documents
        
        except Exception as e:
            logger.error(f"Error querying by metadata: {e}")
            return []
    
    def delete_by_page_id(self, page_id: str) -> bool:
        """
        Delete all chunks for a specific page
        
        Args:
            page_id: Page ID to delete
        
        Returns:
            True if successful
        """
        if not self.collection:
            return False
        
        try:
            self.collection.delete(
                where={"page_id": page_id}
            )
            
            logger.info(f"Deleted chunks for page {page_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting page {page_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.collection:
            return {}
        
        try:
            stats = {
                'name': self.collection.name,
                'count': self.collection.count(),
                'metadata': self.collection.metadata
            }
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def reset_collection(self) -> bool:
        """
        Delete and recreate the collection (WARNING: deletes all data)
        
        Returns:
            True if successful
        """
        if not self.client:
            return False
        
        try:
            # Delete collection
            self.client.delete_collection(name=self.collection_name)
            logger.warning(f"Deleted collection: {self.collection_name}")
            
            # Recreate collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "IISc research documents and embeddings"}
            )
            
            logger.info(f"Recreated collection: {self.collection_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
