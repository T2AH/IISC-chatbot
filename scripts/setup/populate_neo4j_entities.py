"""
Populate Neo4j with Entity nodes from ChromaDB data
Extracts Person, Lab, ResearchTopic entities and creates MENTIONS relationships
"""

import os
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from loguru import logger
from tqdm import tqdm
from collections import defaultdict

load_dotenv()

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'neo4j://127.0.0.1:7687')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
CHROMA_PERSIST_DIR = "./data/chromadb"
COLLECTION_NAME = "iisc_research_docs"


def extract_entities_from_text(text: str, metadata: dict) -> dict:
    """Extract entities from text and metadata using pattern matching"""
    entities = {
        'persons': set(),
        'labs': set(),
        'topics': set(),
        'organizations': set()
    }
    
    text_lower = text.lower()
    
    # Extract labs - look for "lab" or "laboratory" mentions
    lab_patterns = [
        r'(?:the\s+)?(\w+(?:\s+\w+){0,3})\s+lab(?:oratory)?',
        r'lab(?:oratory)?\s+(?:of|for|on)\s+(\w+(?:\s+\w+){0,3})',
        r'(\w+)\s+research\s+(?:group|lab|laboratory)'
    ]
    
    for pattern in lab_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            lab_name = match.group(1).strip()
            if len(lab_name) > 3 and lab_name not in ['the', 'this', 'that', 'their']:
                entities['labs'].add(lab_name.title())
    
    # Extract research topics from keywords and common patterns
    topic_patterns = [
        r'research\s+(?:in|on|area|focus)[\s:]+(\w+(?:\s+\w+){0,4})',
        r'(?:studies?|work(?:ing)?\s+on)\s+(\w+(?:\s+\w+){0,3})',
        r'(?:machine\s+learning|artificial\s+intelligence|deep\s+learning|'
        r'computer\s+vision|natural\s+language\s+processing|cloud\s+computing|'
        r'data\s+science|quantum\s+computing|bioinformatics|neuroscience|'
        r'robotics|cybersecurity|blockchain|IoT|edge\s+computing)'
    ]
    
    for pattern in topic_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            topic = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if len(topic) > 3:
                entities['topics'].add(topic.title())
    
    # Extract from URL patterns
    url = metadata.get('url', '')
    
    # Faculty pages often contain person names
    if '/faculty/' in url or '/people/' in url or '/staff/' in url:
        # Extract from URL path
        path_parts = url.split('/')
        for part in path_parts:
            if part and len(part) > 2 and not part.startswith('http'):
                # Clean up common URL patterns
                clean_name = part.replace('-', ' ').replace('_', ' ').strip()
                if clean_name and not any(x in clean_name.lower() for x in ['faculty', 'people', 'staff', 'www', 'http', 'iisc', 'html', 'php']):
                    if len(clean_name.split()) <= 3:  # Names typically 1-3 words
                        entities['persons'].add(clean_name.title())
    
    # Extract from page title
    title = metadata.get('title', '')
    if title:
        # Lab names in titles
        if 'lab' in title.lower():
            lab_match = re.search(r'(\w+(?:\s+\w+){0,3})\s+lab', title, re.IGNORECASE)
            if lab_match:
                entities['labs'].add(lab_match.group(1).strip().title())
        
        # Person names in titles (Prof, Dr patterns)
        name_pattern = r'(?:Prof(?:essor)?|Dr)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        name_matches = re.finditer(name_pattern, title)
        for match in name_matches:
            entities['persons'].add(match.group(1).strip())
    
    # Extract organizations/departments
    org_patterns = [
        r'department\s+of\s+(\w+(?:\s+\w+){0,4})',
        r'centre\s+for\s+(\w+(?:\s+\w+){0,4})',
        r'school\s+of\s+(\w+(?:\s+\w+){0,4})'
    ]
    
    for pattern in org_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            org_name = match.group(1).strip().title()
            if len(org_name) > 3:
                entities['organizations'].add(org_name)
    
    # Clean up entities
    for key in entities:
        entities[key] = {e for e in entities[key] if e and len(e) > 2}
    
    return entities


def create_entity_schema(driver):
    """Create Entity node type and MENTIONS relationship"""
    logger.info("Creating Entity schema...")
    
    with driver.session() as session:
        # Create Entity constraint
        try:
            session.run("""
                CREATE CONSTRAINT entity_name IF NOT EXISTS 
                FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE
            """)
            logger.success("✓ Created Entity constraint")
        except Exception as e:
            logger.warning(f"Entity constraint issue: {e}")
        
        # Create index for faster lookups
        try:
            session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            logger.success("✓ Created Entity type index")
        except Exception as e:
            logger.warning(f"Index issue: {e}")


def populate_entities(driver, chroma_data):
    """Extract entities from ChromaDB documents and populate Neo4j"""
    logger.info("Extracting and populating entities...")
    
    ids = chroma_data['ids']
    documents = chroma_data['documents']
    metadatas = chroma_data['metadatas']
    total = len(ids)
    
    entity_stats = defaultdict(int)
    entity_types = defaultdict(set)
    
    batch_size = 100
    batches = [range(i, min(i + batch_size, total)) for i in range(0, total, batch_size)]
    
    with driver.session() as session:
        for batch_range in tqdm(batches, desc="Processing documents"):
            batch_entities = []
            
            for i in batch_range:
                page_id = ids[i]
                text = documents[i] if i < len(documents) else ""
                metadata = metadatas[i] if i < len(metadatas) else {}
                
                # Extract entities
                entities = extract_entities_from_text(text, metadata)
                
                # Collect for batch insert
                for person in entities['persons']:
                    batch_entities.append({
                        'page_id': page_id,
                        'entity_name': person,
                        'entity_type': 'Person'
                    })
                    entity_types['Person'].add(person)
                
                for lab in entities['labs']:
                    batch_entities.append({
                        'page_id': page_id,
                        'entity_name': lab,
                        'entity_type': 'Lab'
                    })
                    entity_types['Lab'].add(lab)
                
                for topic in entities['topics']:
                    batch_entities.append({
                        'page_id': page_id,
                        'entity_name': topic,
                        'entity_type': 'ResearchTopic'
                    })
                    entity_types['ResearchTopic'].add(topic)
                
                for org in entities['organizations']:
                    batch_entities.append({
                        'page_id': page_id,
                        'entity_name': org,
                        'entity_type': 'Organization'
                    })
                    entity_types['Organization'].add(org)
            
            # Batch insert entities
            if batch_entities:
                session.run("""
                    UNWIND $entities AS entity
                    MATCH (p:Page {page_id: entity.page_id})
                    MERGE (e:Entity {name: entity.entity_name, type: entity.entity_type})
                    MERGE (p)-[r:MENTIONS]->(e)
                    ON CREATE SET r.count = 1
                    ON MATCH SET r.count = r.count + 1
                """, entities=batch_entities)
                
                entity_stats['total_mentions'] += len(batch_entities)
    
    # Log statistics
    logger.info("\n" + "="*60)
    logger.info("Entity Extraction Summary:")
    logger.info("="*60)
    for entity_type, names in entity_types.items():
        logger.info(f"  {entity_type}: {len(names)} unique entities")
        entity_stats[entity_type] = len(names)
    
    logger.info(f"\n  Total MENTIONS relationships: {entity_stats['total_mentions']}")
    
    # Show sample entities
    with driver.session() as session:
        logger.info("\nSample Entities:")
        result = session.run("""
            MATCH (e:Entity)
            RETURN e.type as type, e.name as name
            LIMIT 10
        """)
        for record in result:
            logger.info(f"  [{record['type']}] {record['name']}")
    
    return entity_stats


def verify_graph_structure(driver):
    """Verify the created graph structure"""
    logger.info("\n" + "="*60)
    logger.info("Graph Structure Verification:")
    logger.info("="*60)
    
    with driver.session() as session:
        # Count nodes by type
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(*) as count
            ORDER BY count DESC
        """)
        logger.info("\nNode Counts:")
        for record in result:
            logger.info(f"  {record['label']}: {record['count']:,}")
        
        # Count relationships
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(*) as count
            ORDER BY count DESC
        """)
        logger.info("\nRelationship Counts:")
        for record in result:
            logger.info(f"  {record['rel_type']}: {record['count']:,}")
        
        # Sample query test
        result = session.run("""
            MATCH (p:Page)-[r:MENTIONS]->(e:Entity)
            WHERE e.type = 'Lab'
            RETURN e.name as lab, count(DISTINCT p) as pages
            ORDER BY pages DESC
            LIMIT 5
        """)
        logger.info("\nTop 5 Most Mentioned Labs:")
        for record in result:
            logger.info(f"  {record['lab']}: mentioned in {record['pages']} pages")


def main():
    logger.info("="*60)
    logger.info("Neo4j Entity Population from ChromaDB")
    logger.info("="*60)
    
    # Check Neo4j connection
    logger.info("Connecting to Neo4j...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logger.success("✓ Connected to Neo4j")
    except Exception as e:
        logger.error(f"✗ Cannot connect to Neo4j: {e}")
        return
    
    try:
        # Step 1: Create Entity schema
        create_entity_schema(driver)
        
        # Step 2: Load ChromaDB data
        logger.info(f"\nLoading data from ChromaDB: {CHROMA_PERSIST_DIR}")
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection = client.get_collection(name=COLLECTION_NAME)
        total_docs = collection.count()
        logger.info(f"Found {total_docs:,} documents in ChromaDB")
        
        # Get all documents
        result = collection.get(include=["documents", "metadatas"])
        logger.success(f"✓ Loaded {len(result['ids']):,} documents")
        
        # Step 3: Extract and populate entities
        stats = populate_entities(driver, result)
        
        # Step 4: Verify structure
        verify_graph_structure(driver)
        
        logger.info("\n" + "="*60)
        logger.success("✓ Entity population complete!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Restart your web server: python start_web_server.py")
        logger.info("2. Test queries like: 'which labs work on machine learning?'")
        logger.info("3. Try follow-up questions: 'tell me more about them'")
        
    except Exception as e:
        logger.error(f"Population failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()


if __name__ == "__main__":
    main()
