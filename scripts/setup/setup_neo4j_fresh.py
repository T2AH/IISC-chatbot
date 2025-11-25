"""
Setup Fresh Neo4j Database with ChromaDB Data
Creates a new Neo4j knowledge graph from ChromaDB embeddings
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from loguru import logger
from tqdm import tqdm

load_dotenv()

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '')
CHROMA_PERSIST_DIR = "./data/chromadb"
COLLECTION_NAME = "iisc_research_docs"


def check_neo4j_connection():
    """Check if Neo4j is running and accessible"""
    logger.info("Checking Neo4j connection...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logger.success("✓ Neo4j is running and accessible")
        driver.close()
        return True
    except Exception as e:
        logger.error(f"✗ Cannot connect to Neo4j: {e}")
        logger.info("\nPlease start Neo4j:")
        logger.info("1. Open Neo4j Desktop")
        logger.info("2. Start your database")
        logger.info("3. Make sure it's running on bolt://localhost:7687")
        return False


def clear_neo4j_database(driver):
    """Clear all nodes and relationships from Neo4j"""
    logger.info("Clearing existing Neo4j database...")
    
    with driver.session() as session:
        # Check current count
        result = session.run("MATCH (n) RETURN count(n) as count")
        old_count = result.single()['count']
        logger.info(f"Current nodes: {old_count}")
        
        if old_count > 0:
            response = input(f"\nDelete all {old_count} nodes? (yes/no): ").strip().lower()
            if response != 'yes':
                logger.info("Database clear cancelled")
                return False
            
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            logger.success("✓ Database cleared")
        else:
            logger.info("Database is already empty")
    
    return True


def create_neo4j_schema(driver):
    """Create Neo4j schema with constraints and indexes"""
    logger.info("Creating Neo4j schema...")
    
    with driver.session() as session:
        # Create constraints (also creates indexes)
        constraints = [
            "CREATE CONSTRAINT page_id IF NOT EXISTS FOR (p:Page) REQUIRE p.page_id IS UNIQUE",
            "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT lab_name IF NOT EXISTS FOR (l:Lab) REQUIRE l.name IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:ResearchTopic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT course_code IF NOT EXISTS FOR (c:Course) REQUIRE c.code IS UNIQUE",
        ]
        
        for constraint in constraints:
            try:
                session.run(constraint)
                logger.info(f"  Created: {constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"  Constraint issue: {e}")
        
        logger.success("✓ Neo4j schema created")


def load_data_from_chromadb():
    """Load documents from ChromaDB"""
    logger.info(f"Loading data from ChromaDB: {CHROMA_PERSIST_DIR}")
    
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection = client.get_collection(name=COLLECTION_NAME)
        total_docs = collection.count()
        logger.info(f"Found {total_docs} documents in ChromaDB")
        
        # Get all documents with metadata
        result = collection.get(include=["documents", "metadatas"])
        
        logger.success(f"✓ Loaded {len(result['ids'])} documents")
        return result
        
    except Exception as e:
        logger.error(f"Failed to load ChromaDB data: {e}")
        return None


def populate_neo4j(driver, chroma_data):
    """Populate Neo4j with data from ChromaDB"""
    logger.info("Populating Neo4j knowledge graph...")
    
    ids = chroma_data['ids']
    documents = chroma_data['documents']
    metadatas = chroma_data['metadatas']
    total = len(ids)
    
    logger.info(f"Processing {total} documents...")
    
    with driver.session() as session:
        created_pages = 0
        created_entities = 0
        
        for i in tqdm(range(total), desc="Creating nodes"):
            page_id = ids[i]
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # Create Page node
            url = metadata.get('url', 'unknown')
            title = metadata.get('title', 'Unknown')
            page_type = metadata.get('page_type', 'general')
            domain = metadata.get('domain', 'unknown')
            
            session.run("""
                MERGE (p:Page {page_id: $page_id})
                SET p.url = $url,
                    p.title = $title,
                    p.page_type = $page_type,
                    p.domain = $domain
            """, page_id=page_id, url=url, title=title, page_type=page_type, domain=domain)
            created_pages += 1
            
            # Extract and create entity nodes from metadata if available
            entities = metadata.get('entities', {})
            
            # Create Person nodes
            for person in entities.get('PERSON', []):
                if isinstance(person, dict):
                    name = person.get('text', '').strip()
                else:
                    name = str(person).strip()
                
                if name and len(name) > 2:
                    session.run("""
                        MERGE (per:Person {name: $name})
                        MERGE (p:Page {page_id: $page_id})
                        MERGE (per)-[:MENTIONED_IN]->(p)
                    """, name=name, page_id=page_id)
                    created_entities += 1
            
            # Create ResearchTopic nodes
            for topic in entities.get('RESEARCH_TOPIC', []):
                if isinstance(topic, dict):
                    name = topic.get('text', '').strip()
                else:
                    name = str(topic).strip()
                
                if name and len(name) > 2:
                    session.run("""
                        MERGE (t:ResearchTopic {name: $name})
                        MERGE (p:Page {page_id: $page_id})
                        MERGE (t)-[:DISCUSSED_IN]->(p)
                    """, name=name, page_id=page_id)
                    created_entities += 1
        
        logger.success(f"✓ Created {created_pages} pages and {created_entities} entity relationships")
        
        # Show summary
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(*) as count
            ORDER BY count DESC
        """)
        
        logger.info("\nDatabase Summary:")
        for record in result:
            logger.info(f"  {record['label']}: {record['count']}")


def main():
    logger.info("="*60)
    logger.info("Neo4j Fresh Setup from ChromaDB")
    logger.info("="*60)
    
    # Step 1: Check Neo4j connection
    if not check_neo4j_connection():
        return
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Step 2: Clear existing database
        if not clear_neo4j_database(driver):
            return
        
        # Step 3: Create schema
        create_neo4j_schema(driver)
        
        # Step 4: Load ChromaDB data
        chroma_data = load_data_from_chromadb()
        if not chroma_data:
            return
        
        # Step 5: Populate Neo4j
        populate_neo4j(driver, chroma_data)
        
        logger.info("\n" + "="*60)
        logger.success("✓ Neo4j setup complete!")
        logger.info("="*60)
        logger.info("You can now use the chatbot with the new knowledge graph")
        logger.info("Run: python main.py chat --interactive")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise
    finally:
        driver.close()


if __name__ == "__main__":
    main()
