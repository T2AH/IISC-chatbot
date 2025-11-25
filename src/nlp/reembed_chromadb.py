"""
Re-embed ChromaDB documents with OpenAI text-embedding-3-large
This script upgrades embeddings for better semantic matching with structured data
"""

import os
import time
from typing import List, Dict, Any
from loguru import logger
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CHROMA_PERSIST_DIR = "./data/chromadb"  # Correct path from config
OLD_COLLECTION_NAME = "iisc_research_docs"
BACKUP_COLLECTION_NAME = "iisc_research_docs_backup"
NEW_EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072  # Can be 1536 or 3072 (higher = better quality)
BATCH_SIZE = 100  # Reduced to avoid token limit (was 500)
CHROMA_BATCH_SIZE = 1000  # ChromaDB insert batch size

def backup_collection(client: chromadb.Client):
    """Create backup of existing collection"""
    logger.info("Creating backup of existing collection...")
    
    try:
        # Delete old backup if exists
        try:
            client.delete_collection(name=BACKUP_COLLECTION_NAME)
            logger.info(f"Deleted old backup collection: {BACKUP_COLLECTION_NAME}")
        except:
            pass
        
        # Get existing collection
        old_collection = client.get_collection(name=OLD_COLLECTION_NAME)
        
        # Get all documents
        result = old_collection.get(include=["documents", "metadatas", "embeddings"])
        total_docs = len(result['ids'])
        
        logger.info(f"Backing up {total_docs} documents...")
        
        # Create backup collection with old embeddings
        backup_collection = client.create_collection(
            name=BACKUP_COLLECTION_NAME,
            metadata={"description": "Backup before re-embedding"}
        )
        
        # Copy in batches
        for i in range(0, total_docs, CHROMA_BATCH_SIZE):
            end_idx = min(i + CHROMA_BATCH_SIZE, total_docs)
            backup_collection.add(
                ids=result['ids'][i:end_idx],
                documents=result['documents'][i:end_idx],
                metadatas=result['metadatas'][i:end_idx],
                embeddings=result['embeddings'][i:end_idx]
            )
            logger.info(f"Backed up {end_idx}/{total_docs} documents")
        
        logger.success(f"✅ Backup complete: {total_docs} documents saved to '{BACKUP_COLLECTION_NAME}'")
        return total_docs
        
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        raise

def fetch_all_documents(client: chromadb.Client) -> Dict[str, Any]:
    """Fetch all documents from existing collection"""
    logger.info("Fetching all documents from ChromaDB...")
    
    collection = client.get_collection(name=OLD_COLLECTION_NAME)
    result = collection.get(include=["documents", "metadatas"])
    
    logger.info(f"Fetched {len(result['ids'])} documents")
    return result

def generate_embeddings_batch(texts: List[str], openai_client: OpenAI) -> List[List[float]]:
    """Generate embeddings using OpenAI API with error handling"""
    try:
        # Filter out empty texts and truncate very long ones
        processed_texts = []
        for text in texts:
            if not text or len(text.strip()) == 0:
                text = "empty"
            # Truncate if too long (8192 token limit, ~6000 chars to be safe)
            if len(text) > 6000:
                text = text[:6000]
            processed_texts.append(text)
        
        response = openai_client.embeddings.create(
            model=NEW_EMBEDDING_MODEL,
            input=processed_texts,
            dimensions=EMBEDDING_DIMENSION
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        # If batch fails, try one at a time
        if len(texts) > 1:
            logger.warning(f"Batch failed, retrying {len(texts)} texts individually...")
            embeddings = []
            for text in texts:
                try:
                    single_embed = generate_embeddings_batch([text], openai_client)
                    embeddings.extend(single_embed)
                except:
                    # Use zero vector as fallback
                    logger.error(f"Failed to embed text (length: {len(text)}), using zero vector")
                    embeddings.append([0.0] * EMBEDDING_DIMENSION)
            return embeddings
        raise

def reembed_documents(client: chromadb.Client, openai_client: OpenAI):
    """Re-embed all documents with new model"""
    
    # Fetch existing documents
    documents_data = fetch_all_documents(client)
    
    ids = documents_data['ids']
    documents = documents_data['documents']
    metadatas = documents_data['metadatas']
    total_docs = len(ids)
    
    logger.info(f"Starting re-embedding of {total_docs} documents...")
    logger.info(f"Model: {NEW_EMBEDDING_MODEL}, Dimensions: {EMBEDDING_DIMENSION}")
    
    # Generate new embeddings in batches
    all_embeddings = []
    start_time = time.time()
    
    with tqdm(total=total_docs, desc="Generating embeddings") as pbar:
        for i in range(0, total_docs, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, total_docs)
            batch_texts = documents[i:end_idx]
            
            # Generate embeddings for batch
            batch_embeddings = generate_embeddings_batch(batch_texts, openai_client)
            all_embeddings.extend(batch_embeddings)
            
            pbar.update(end_idx - i)
            
            # Log progress
            elapsed = time.time() - start_time
            docs_per_sec = (end_idx) / elapsed if elapsed > 0 else 0
            logger.info(f"Processed {end_idx}/{total_docs} docs ({docs_per_sec:.1f} docs/sec)")
    
    elapsed_time = time.time() - start_time
    logger.success(f"✅ Generated {len(all_embeddings)} embeddings in {elapsed_time:.1f} seconds")
    logger.info(f"Average speed: {total_docs/elapsed_time:.1f} documents/second")
    
    # Delete old collection and create new one
    logger.info(f"Deleting old collection: {OLD_COLLECTION_NAME}")
    client.delete_collection(name=OLD_COLLECTION_NAME)
    
    logger.info(f"Creating new collection with {EMBEDDING_DIMENSION}-dimensional embeddings...")
    new_collection = client.create_collection(
        name=OLD_COLLECTION_NAME,
        metadata={
            "description": "IISc research documents",
            "embedding_model": NEW_EMBEDDING_MODEL,
            "embedding_dimension": str(EMBEDDING_DIMENSION),
            "reembedded_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )
    
    # Insert documents with new embeddings in batches
    logger.info("Inserting documents with new embeddings...")
    
    with tqdm(total=total_docs, desc="Inserting documents") as pbar:
        for i in range(0, total_docs, CHROMA_BATCH_SIZE):
            end_idx = min(i + CHROMA_BATCH_SIZE, total_docs)
            
            new_collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                embeddings=all_embeddings[i:end_idx]
            )
            
            pbar.update(end_idx - i)
            logger.info(f"Inserted {end_idx}/{total_docs} documents")
    
    logger.success(f"✅ Re-embedding complete! {total_docs} documents now use {NEW_EMBEDDING_MODEL}")
    
    # Verify
    verify_collection = client.get_collection(name=OLD_COLLECTION_NAME)
    final_count = verify_collection.count()
    logger.info(f"Verification: Collection contains {final_count} documents")
    
    return final_count

def main():
    logger.info("="*60)
    logger.info("ChromaDB Re-embedding Script")
    logger.info("="*60)
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ OPENAI_API_KEY not found in environment")
        logger.info("Please set it in your .env file or environment variables")
        return
    
    logger.info(f"✅ OpenAI API key found")
    logger.info(f"Model: {NEW_EMBEDDING_MODEL}")
    logger.info(f"Dimensions: {EMBEDDING_DIMENSION}")
    logger.info(f"ChromaDB path: {CHROMA_PERSIST_DIR}")
    
    # Initialize clients
    openai_client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Confirm before proceeding
    logger.warning(f"⚠️  This will replace embeddings in '{OLD_COLLECTION_NAME}'")
    logger.warning(f"⚠️  Estimated cost: $7-10 for ~200k documents")
    logger.warning(f"⚠️  Estimated time: 10-20 minutes")
    
    response = input("\nProceed with re-embedding? (yes/no): ").strip().lower()
    if response != 'yes':
        logger.info("Re-embedding cancelled")
        return
    
    start_total = time.time()
    
    try:
        # Step 1: Backup
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Creating backup")
        logger.info("="*60)
        backup_collection(chroma_client)
        
        # Step 2: Re-embed
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Re-embedding documents")
        logger.info("="*60)
        final_count = reembed_documents(chroma_client, openai_client)
        
        total_time = time.time() - start_total
        
        # Summary
        logger.info("\n" + "="*60)
        logger.success("✅ RE-EMBEDDING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Total documents: {final_count}")
        logger.info(f"New model: {NEW_EMBEDDING_MODEL}")
        logger.info(f"Dimensions: {EMBEDDING_DIMENSION}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Backup saved as: '{BACKUP_COLLECTION_NAME}'")
        logger.info("="*60)
        logger.info("You can now restart your chatbot to use the new embeddings!")
        
    except Exception as e:
        logger.error(f"❌ Re-embedding failed: {e}")
        logger.info("Your original data is safe in the backup collection")
        raise

if __name__ == "__main__":
    main()
