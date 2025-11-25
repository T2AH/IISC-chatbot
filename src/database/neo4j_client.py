"""
Neo4j database client for knowledge graph storage
"""

from typing import List, Dict, Any, Optional
import time
from neo4j import GraphDatabase
from neo4j.exceptions import TransientError
from loguru import logger

from src.config import config


class Neo4jClient:
    """Client for interacting with Neo4j knowledge graph database"""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        """
        Initialize Neo4j client
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
        """
        self.uri = uri or config.neo4j_uri
        self.username = username or config.neo4j_username
        self.password = password or config.neo4j_password
        
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            
            logger.info(f"Connected to Neo4j at {self.uri}")
        
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def create_indices(self):
        """Create indices for better query performance"""
        if not self.driver:
            logger.error("No Neo4j connection available")
            return
        
        indices = [
            "CREATE INDEX faculty_name IF NOT EXISTS FOR (f:Faculty) ON (f.name)",
            "CREATE INDEX lab_name IF NOT EXISTS FOR (l:Lab) ON (l.name)",
            "CREATE INDEX project_name IF NOT EXISTS FOR (p:Project) ON (p.name)",
            "CREATE INDEX topic_name IF NOT EXISTS FOR (t:ResearchTopic) ON (t.name)",
            "CREATE INDEX course_code IF NOT EXISTS FOR (c:Course) ON (c.code)",
            "CREATE INDEX page_id IF NOT EXISTS FOR (p:Page) ON (p.page_id)",
            "CREATE INDEX entity_name_type IF NOT EXISTS FOR (e:Entity) ON (e.name, e.type)",
        ]
        
        with self.driver.session() as session:
            for index_query in indices:
                try:
                    session.run(index_query)
                    logger.debug(f"Created index: {index_query[:50]}...")
                except Exception as e:
                    logger.warning(f"Index creation failed or already exists: {e}")
        
        logger.info("Indices created successfully")
    
    def create_constraints(self):
        """Create uniqueness constraints"""
        if not self.driver:
            logger.error("No Neo4j connection available")
            return
        
        constraints = [
            "CREATE CONSTRAINT faculty_unique IF NOT EXISTS FOR (f:Faculty) REQUIRE f.name IS UNIQUE",
            "CREATE CONSTRAINT page_unique IF NOT EXISTS FOR (p:Page) REQUIRE p.page_id IS UNIQUE",
        ]
        
        with self.driver.session() as session:
            for constraint_query in constraints:
                try:
                    session.run(constraint_query)
                    logger.debug(f"Created constraint: {constraint_query[:50]}...")
                except Exception as e:
                    logger.warning(f"Constraint creation failed or already exists: {e}")
        
        logger.info("Constraints created successfully")
    
    def bulk_create_pages(self, pages_data: List[Dict[str, Any]]) -> bool:
        """
        Bulk create Page nodes (optimized for parallel import)
        
        Args:
            pages_data: List of page dictionaries
        
        Returns:
            True if successful
        """
        if not self.driver or not pages_data:
            return False
        
        try:
            query = """
            UNWIND $pages AS page
            MERGE (p:Page {page_id: page.page_id})
            SET p.url = page.url,
                p.title = page.title,
                p.page_type = page.page_type,
                p.domain = page.domain
            """
            
            with self.driver.session() as session:
                session.run(query, pages=pages_data)
            
            logger.debug(f"Bulk created {len(pages_data)} pages")
            return True
            
        except Exception as e:
            logger.error(f"Bulk page creation failed: {e}")
            return False
    
    def bulk_create_relationships(self, relationships_data: List[Dict[str, Any]], max_retries: int = 5) -> bool:
        """
        Bulk create entity nodes and relationships with retry logic for deadlocks
        
        Optimized 2-stage approach:
        1. MERGE all entities first (indexed lookup)
        2. CREATE relationships in bulk (no MERGE overhead)
        
        Args:
            relationships_data: List of relationship dictionaries
            max_retries: Maximum retry attempts for deadlock errors
        
        Returns:
            True if successful
        """
        if not self.driver or not relationships_data:
            return False
        
        # Stage 1: MERGE entities (fast with index)
        entity_query = """
        UNWIND $relationships AS rel
        MERGE (e:Entity {name: rel.entity_name, type: rel.entity_type})
        """
        
        # Stage 2: MERGE relationships (preserves quality, avoids duplicates)
        relationship_query = """
        UNWIND $relationships AS rel
        MATCH (p:Page {page_id: rel.page_id})
        MATCH (e:Entity {name: rel.entity_name, type: rel.entity_type})
        MERGE (p)-[r:MENTIONS]->(e)
        ON CREATE SET r.count = rel.count
        ON MATCH SET r.count = r.count + rel.count
        """
        
        for attempt in range(max_retries):
            try:
                with self.driver.session() as session:
                    # Stage 1: Create entities
                    session.run(entity_query, relationships=relationships_data)
                    # Stage 2: Create relationships (much faster)
                    session.run(relationship_query, relationships=relationships_data)
                
                logger.debug(f"Bulk created {len(relationships_data)} relationships")
                return True
                
            except TransientError as e:
                if "DeadlockDetected" in str(e):
                    if attempt < max_retries - 1:
                        # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s
                        wait_time = 0.5 * (2 ** attempt)
                        logger.warning(f"Deadlock detected, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Bulk relationship creation failed after {max_retries} retries: {e}")
                        return False
                else:
                    logger.error(f"Transient error in bulk relationship creation: {e}")
                    return False
                    
            except Exception as e:
                logger.error(f"Bulk relationship creation failed: {e}")
                return False
        
        return False
    
    def create_faculty_node(self, name: str, properties: Dict[str, Any] = None) -> bool:
        """
        Create or update a Faculty node
        
        Args:
            name: Faculty member name
            properties: Additional properties
        
        Returns:
            True if successful
        """
        if not self.driver:
            return False
        
        try:
            query = """
            MERGE (f:Faculty {name: $name})
            SET f += $properties
            RETURN f
            """
            
            props = properties or {}
            
            with self.driver.session() as session:
                session.run(query, name=name, properties=props)
            
            logger.debug(f"Created/updated Faculty node: {name}")
            return True
        
        except Exception as e:
            logger.error(f"Error creating Faculty node: {e}")
            return False
    
    def create_lab_node(self, name: str, properties: Dict[str, Any] = None) -> bool:
        """Create or update a Lab node"""
        if not self.driver:
            return False
        
        try:
            query = """
            MERGE (l:Lab {name: $name})
            SET l += $properties
            RETURN l
            """
            
            props = properties or {}
            
            with self.driver.session() as session:
                session.run(query, name=name, properties=props)
            
            logger.debug(f"Created/updated Lab node: {name}")
            return True
        
        except Exception as e:
            logger.error(f"Error creating Lab node: {e}")
            return False
    
    def create_relationship(self, from_node: Dict[str, Any], rel_type: str, 
                          to_node: Dict[str, Any], properties: Dict[str, Any] = None) -> bool:
        """
        Create a relationship between two nodes
        
        Args:
            from_node: Dictionary with 'label' and 'name' keys
            rel_type: Relationship type (e.g., 'WORKS_IN', 'CONDUCTS')
            to_node: Dictionary with 'label' and 'name' keys
            properties: Optional relationship properties
        
        Returns:
            True if successful
        """
        if not self.driver:
            return False
        
        try:
            query = f"""
            MATCH (a:{from_node['label']} {{name: $from_name}})
            MATCH (b:{to_node['label']} {{name: $to_name}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $properties
            RETURN r
            """
            
            props = properties or {}
            
            with self.driver.session() as session:
                session.run(
                    query,
                    from_name=from_node['name'],
                    to_name=to_node['name'],
                    properties=props
                )
            
            logger.debug(f"Created relationship: {from_node['name']} -{rel_type}-> {to_node['name']}")
            return True
        
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            return False
    
    def store_page_entities(self, page_data: Dict[str, Any]) -> bool:
        """
        Store entities from a processed page in the knowledge graph
        
        Args:
            page_data: Processed page data with entities
        
        Returns:
            True if successful
        """
        if not self.driver:
            return False
        
        try:
            page_id = page_data.get('page_id')
            url = page_data.get('url')
            entities = page_data.get('entities', {})
            
            # Create Page node
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (p:Page {page_id: $page_id})
                    SET p.url = $url,
                        p.title = $title,
                        p.domain = $domain,
                        p.page_type = $page_type
                    """,
                    page_id=page_id,
                    url=url,
                    title=page_data.get('title', ''),
                    domain=page_data.get('domain', ''),
                    page_type=page_data.get('page_type', '')
                )
            
            # Create entity nodes and relationships
            # Faculty entities
            for person in entities.get('PERSON', []):
                name = person['text']
                self.create_faculty_node(name, {'source_page': page_id})
                
                # Link to page
                self.create_relationship(
                    {'label': 'Faculty', 'name': name},
                    'MENTIONED_IN',
                    {'label': 'Page', 'name': page_id}
                )
            
            # Lab entities
            for lab in entities.get('LAB_NAME', []):
                name = lab['text']
                self.create_lab_node(name, {'source_page': page_id})
                
                # Link to page
                self.create_relationship(
                    {'label': 'Lab', 'name': name},
                    'MENTIONED_IN',
                    {'label': 'Page', 'name': page_id}
                )
            
            # Research topics
            for topic in entities.get('RESEARCH_TOPIC', []):
                name = topic['text']
                
                with self.driver.session() as session:
                    session.run(
                        "MERGE (t:ResearchTopic {name: $name})",
                        name=name
                    )
                
                # Link to page
                self.create_relationship(
                    {'label': 'ResearchTopic', 'name': name},
                    'MENTIONED_IN',
                    {'label': 'Page', 'name': page_id}
                )
            
            logger.info(f"Stored entities for page {page_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error storing page entities: {e}")
            return False

    def store_pages_entities_batch(self, pages: List[Dict[str, Any]]) -> bool:
        """
        Store entities and relationships for multiple pages in a single batched transaction.

        Args:
            pages: List of processed page dicts

        Returns:
            True if successful
        """
        if not self.driver:
            return False

        try:
            # Prepare rows for UNWIND
            rows = []
            for p in pages:
                rows.append({
                    'page_id': p.get('page_id'),
                    'url': p.get('url'),
                    'title': p.get('title', ''),
                    'domain': p.get('domain', ''),
                    'page_type': p.get('page_type', ''),
                    'persons': [e['text'] for e in p.get('entities', {}).get('PERSON', [])],
                    'labs': [e['text'] for e in p.get('entities', {}).get('LAB_NAME', [])],
                    'topics': [e['text'] for e in p.get('entities', {}).get('RESEARCH_TOPIC', [])]
                })

            # Use UNWIND to create pages, entities and relationships in a single transaction
            query = """
            UNWIND $rows AS r
            MERGE (p:Page {page_id: r.page_id})
            SET p.url = r.url, p.title = r.title, p.domain = r.domain, p.page_type = r.page_type

            WITH r, p
            UNWIND (CASE WHEN r.persons IS NULL THEN [] ELSE r.persons END) AS personName
            MERGE (f:Faculty {name: personName})
            SET f.source_page = r.page_id
            MERGE (f)-[:MENTIONED_IN]->(p)

            WITH r, p
            UNWIND (CASE WHEN r.labs IS NULL THEN [] ELSE r.labs END) AS labName
            MERGE (l:Lab {name: labName})
            SET l.source_page = r.page_id
            MERGE (l)-[:MENTIONED_IN]->(p)

            WITH r, p
            UNWIND (CASE WHEN r.topics IS NULL THEN [] ELSE r.topics END) AS topicName
            MERGE (t:ResearchTopic {name: topicName})
            MERGE (t)-[:MENTIONED_IN]->(p)
            """

            with self.driver.session() as session:
                session.run(query, rows=rows)

            logger.info(f"Batched stored entities for {len(rows)} pages")
            return True

        except Exception as e:
            logger.error(f"Error storing pages entities batch: {e}")
            return False
    
    def query_faculty(self, name: str) -> Optional[Dict[str, Any]]:
        """Query faculty information"""
        if not self.driver:
            return None
        
        try:
            query = """
            MATCH (f:Faculty {name: $name})
            OPTIONAL MATCH (f)-[:WORKS_IN]->(l:Lab)
            OPTIONAL MATCH (f)-[:CONDUCTS]->(p:Project)
            OPTIONAL MATCH (f)-[r:RESEARCHES]->(t:ResearchTopic)
            RETURN f, collect(DISTINCT l.name) as labs, 
                   collect(DISTINCT p.name) as projects,
                   collect(DISTINCT t.name) as topics
            """
            
            with self.driver.session() as session:
                result = session.run(query, name=name)
                record = result.single()
                
                if record:
                    return {
                        'faculty': dict(record['f']),
                        'labs': record['labs'],
                        'projects': record['projects'],
                        'topics': record['topics']
                    }
            
            return None
        
        except Exception as e:
            logger.error(f"Error querying faculty: {e}")
            return None
    
    def search_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Search for faculty and labs by research topic"""
        if not self.driver:
            return []
        
        try:
            query = """
            MATCH (t:ResearchTopic)
            WHERE toLower(t.name) CONTAINS toLower($topic)
            OPTIONAL MATCH (t)<-[:RESEARCHES]-(f:Faculty)
            OPTIONAL MATCH (t)<-[:COVERS]-(l:Lab)
            RETURN t.name as topic, 
                   collect(DISTINCT f.name) as faculty,
                   collect(DISTINCT l.name) as labs
            """
            
            with self.driver.session() as session:
                result = session.run(query, topic=topic)
                
                results = []
                for record in result:
                    results.append({
                        'topic': record['topic'],
                        'faculty': record['faculty'],
                        'labs': record['labs']
                    })
                
                return results
        
        except Exception as e:
            logger.error(f"Error searching by topic: {e}")
            return []
