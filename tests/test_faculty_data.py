#!/usr/bin/env python3.8
"""Test script to check faculty data in database"""
import psycopg2

def test_faculty_data():
    db_config = {
        'host': 'localhost',
        'database': 'cds_rag_db',
        'user': 'rag_user',
        'password': 'secure_rag_password_123',
        'port': 5432
    }
    
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    # Test 1: Check faculty data exists
    cur.execute("""
        SELECT COUNT(*) FROM cds_embeddings 
        WHERE faculty_names IS NOT NULL 
        AND array_length(faculty_names, 1) > 0;
    """)
    faculty_count = cur.fetchone()[0]
    print(f"📊 Chunks with faculty data: {faculty_count}")
    
    # Test 2: Sample faculty data (fixed syntax)
    cur.execute("""
        SELECT chunk_id, faculty_names, research_areas, LEFT(chunk_text, 100)
        FROM cds_embeddings 
        WHERE faculty_names IS NOT NULL 
        AND array_length(faculty_names, 1) > 0
        LIMIT 5;
    """)
    
    results = cur.fetchall()
    print(f"\n👥 Sample faculty data:")
    for chunk_id, faculty, research, text in results:
        print(f"  {chunk_id}: {faculty} | {research}")
        print(f"    Text: {text}...")
        print()
    
    # Test 3: Search for faculty query
    print("\n🔍 Testing faculty search:")
    cur.execute("""
        SELECT chunk_id, faculty_names, research_areas, chunk_type
        FROM cds_embeddings 
        WHERE chunk_text ILIKE '%faculty%' 
        OR chunk_text ILIKE '%professor%'
        OR chunk_text ILIKE '%sathish%'
        OR chunk_text ILIKE '%vadhiyar%'
        LIMIT 5;
    """)
    
    results = cur.fetchall()
    for chunk_id, faculty, research, chunk_type in results:
        print(f"  {chunk_id} ({chunk_type}): Faculty={faculty}, Research={research}")
    
    # Test 4: Check what the similarity search actually finds
    print("\n🔍 Testing similarity search for 'faculty names':")
    cur.execute("""
        SELECT chunk_id, faculty_names, research_areas, chunk_type,
               LEFT(chunk_text, 150) as sample_text
        FROM cds_embeddings 
        WHERE faculty_names IS NOT NULL 
        AND array_length(faculty_names, 1) > 0
        ORDER BY RANDOM()
        LIMIT 3;
    """)
    
    results = cur.fetchall()
    print("Sample chunks with faculty data:")
    for chunk_id, faculty, research, chunk_type, text in results:
        print(f"\n  ID: {chunk_id}")
        print(f"  Type: {chunk_type}")
        print(f"  Faculty: {faculty}")
        print(f"  Research: {research}")
        print(f"  Sample text: {text}...")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    test_faculty_data()