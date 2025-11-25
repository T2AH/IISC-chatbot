# Neo4j Setup Guide

## Quick Start

### 1. Start Neo4j Desktop

You have Neo4j Desktop installed at: `C:\Users\cdsmt\.Neo4jDesktop2`

**To start Neo4j:**

1. Open Neo4j Desktop from Start Menu
2. Select your database project (or create a new one)
3. Click the **Start** button
4. Wait for the database status to show **"Active"**

### 2. Verify Connection

Your `.env` file should have:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

### 3. Setup Fresh Database

Once Neo4j is running, execute:

```powershell
python setup_neo4j_fresh.py
```

This will:
- Clear existing Neo4j data (after confirmation)
- Create fresh schema with constraints and indexes
- Populate knowledge graph from ChromaDB (with text-embedding-3-large embeddings)
- Create entity nodes (Person, ResearchTopic, etc.) and relationships

### 4. Test the Chatbot

```powershell
python main.py chat --interactive
```

## Troubleshooting

### Neo4j Won't Start

**Issue:** "Connection refused" error

**Solutions:**
1. Make sure Neo4j Desktop is open
2. Click "Start" on your database
3. Check the database status shows "Active" (green)
4. Verify port 7687 is not blocked by firewall

### Wrong Password

**Issue:** Authentication failed

**Solutions:**
1. Check `.env` file has correct password
2. In Neo4j Desktop, you can reset password:
   - Click on database
   - Go to "Manage" → "Open Terminal"
   - Run: `neo4j-admin set-initial-password <newpassword>`
3. Update `.env` with new password

### Database Not Found

**Issue:** No databases available

**Solutions:**
1. In Neo4j Desktop, create a new database:
   - Click "Add" → "Create a Local DBMS"
   - Set name and password
   - Remember password for `.env` file
2. Start the new database
3. Update `.env` if needed

## Creating a New Neo4j Database

If you want to start completely fresh:

### Option 1: Using Neo4j Desktop (Recommended)

1. Open Neo4j Desktop
2. Click "Add" → "Create a Local DBMS"
3. Set details:
   - **Name:** IISc Chatbot DB
   - **Password:** (choose a secure password)
   - **Version:** 5.x (latest)
4. Click "Create"
5. Click "Start"
6. Update `.env` with your password
7. Run `python setup_neo4j_fresh.py`

### Option 2: Using Docker

```powershell
docker run `
  --name neo4j-iisc `
  -p 7474:7474 -p 7687:7687 `
  -d `
  -e NEO4J_AUTH=neo4j/your_password `
  neo4j:5-community
```

Then update `.env` and run `python setup_neo4j_fresh.py`

## Benefits of Fresh Setup

With your new **text-embedding-3-large** embeddings (3072 dimensions), the knowledge graph will provide:

✅ **Better semantic understanding** - Higher quality embeddings capture nuances  
✅ **Improved entity extraction** - More accurate entity relationships  
✅ **Enhanced context retrieval** - Better hybrid search results  
✅ **Cleaner data structure** - Fresh schema without old inconsistencies  

## What Gets Created

The setup script creates:

### Nodes
- `Page` - Web pages from crawled data
- `Person` - Faculty, researchers, staff
- `ResearchTopic` - Research areas and topics
- `Organization` - Departments, labs, institutions
- `Lab` - Research labs and groups
- `Course` - Academic courses

### Relationships
- `MENTIONED_IN` - Person mentioned in page
- `DISCUSSED_IN` - Topic discussed in page
- `WORKS_AT` - Person works at organization
- `RESEARCHES` - Person researches topic
- `TEACHES` - Person teaches course

### Constraints & Indexes
- Unique constraints on all primary keys
- Automatic indexes for fast lookups
- Optimized for hybrid retrieval queries

## Next Steps

After setup:

1. **Test retrieval**: `python main.py chat --query "Who works on machine learning?"`
2. **Browse graph**: Open Neo4j Browser at http://localhost:7474
3. **Run queries**: 
   ```cypher
   MATCH (p:Person)-[:RESEARCHES]->(t:ResearchTopic)
   RETURN p.name, t.name
   LIMIT 10
   ```

## Database Maintenance

### Backup Database

Neo4j Desktop → Database → Manage → Dump

### Clear Database

```cypher
MATCH (n) DETACH DELETE n
```

Or run `python setup_neo4j_fresh.py` again

### Check Database Stats

```cypher
CALL db.schema.visualization()
```

```cypher
MATCH (n)
RETURN labels(n)[0] as NodeType, count(*) as Count
ORDER BY Count DESC
```
