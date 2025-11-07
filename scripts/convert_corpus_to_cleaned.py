#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert hierarchical corpus to cleaned JSONL format for chunker
"""
import json
import os

def convert_corpus_to_cleaned(input_file, output_file):
    """Convert hierarchical corpus JSON to cleaned JSONL"""
    
    print(f"Converting {input_file} to {output_file}...")
    
    # Load hierarchical corpus
    with open(input_file, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert nodes to cleaned format
    nodes_processed = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for node in corpus.get('nodes', []):
            # Skip nodes without meaningful content
            content = node.get('content', '').strip()
            if not content or len(content) < 50:
                continue
            
            # Create cleaned document
            cleaned_doc = {
                'url': node.get('url', ''),
                'title': node.get('title', ''),
                'cleaned_text': content,
                'text': content,
                'domain': 'cds.iisc.ac.in',
                'content_length': len(content),
                'word_count': len(content.split()),
                # Include hierarchy metadata
                'hierarchy_level': node.get('level'),
                'parent_id': node.get('parent_id'),
                'children_ids': node.get('children_ids', []),
                'node_type': node.get('node_type', 'general'),
                'node_id': node.get('node_id')
            }
            
            # Write as JSONL
            f.write(json.dumps(cleaned_doc, ensure_ascii=False) + '\n')
            nodes_processed += 1
    
    print(f"✓ Converted {nodes_processed} nodes to cleaned format")
    print(f"✓ Output saved to: {output_file}")
    return nodes_processed

if __name__ == "__main__":
    input_file = "cds_hierarchical_corpus.json"
    output_file = "data/processed/cds_cleaned.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        print("Run the crawler first: python crawler/cds_hierarchical_crawler.py")
        exit(1)
    
    convert_corpus_to_cleaned(input_file, output_file)
    print("\nNow you can run the chunker:")
    print("python src/chunking/clean_chunker.py")
