# -*- coding: utf-8 -*-
import json
import re
import os
import codecs

class CleanChunker:
    def __init__(self, min_tokens=200, max_tokens=500, overlap_tokens=30, hierarchy_file=None):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.hierarchy_map = {}
        
        # Load hierarchical structure if provided
        if hierarchy_file and os.path.exists(hierarchy_file):
            self.load_hierarchy(hierarchy_file)
    
    def load_hierarchy(self, hierarchy_file):
        """Load hierarchical corpus and create URL mapping"""
        print("Loading hierarchical structure from: {}".format(hierarchy_file))
        try:
            with codecs.open(hierarchy_file, 'r', 'utf-8') as f:
                corpus = json.load(f)
            
            # Create URL to hierarchy mapping
            for node in corpus.get('nodes', []):
                url = node.get('url')
                if url:
                    self.hierarchy_map[url] = {
                        'level': node.get('level'),
                        'parent_id': node.get('parent_id'),
                        'children_ids': node.get('children_ids', []),
                        'node_type': node.get('node_type')
                    }
            
            print("Loaded hierarchy for {} nodes".format(len(self.hierarchy_map)))
        except Exception as e:
            print("Warning: Could not load hierarchy: {}".format(e))
            self.hierarchy_map = {}
    
    def estimate_tokens(self, text):
        """Estimate token count (1 word = 1.3 tokens approximately)"""
        if not text:
            return 0
        words = re.findall(r'\b\w+\b', text)
        return int(len(words) * 1.3)
    
    def extract_entities(self, text):
        """Extract faculty names, departments, research areas from text"""
        entities = {
            'faculty_names': [],
            'departments': [],
            'research_areas': [],
            'student_names': [],
            'positions': []
        }
        
        # Common department patterns
        dept_patterns = [
            r'Department of ([A-Z][a-zA-Z\s&]+)',
            r'([A-Z][a-zA-Z\s&]+) Department',
            r'(Computer Science|Computational Science|Data Science|Engineering|Mathematics|Physics|Chemistry|Biology)',
            r'(CDS|CSA|ECE|EE|ME|CE|SERC|IISc)'
        ]
        
        # Faculty/Person name patterns
        name_patterns = [
            r'Dr\.?\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
            r'Prof\.?\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
            r'([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+)(?:\s+(?:receives|awarded|honored))',
            r'([A-Z][a-zA-Z]+\s+[A-Z]\.?\s*[A-Z][a-zA-Z]+)'
        ]
        
        # Research area patterns
        research_patterns = [
            r'(machine learning|artificial intelligence|deep learning|neural networks)',
            r'(computational biology|bioinformatics|genomics)',
            r'(quantum computing|high performance computing|HPC)',
            r'(data science|data analytics|big data)',
            r'(computer vision|natural language processing|NLP)',
            r'(algorithms|optimization|modeling)'
        ]
        
        # Position patterns
        position_patterns = [
            r'(PhD student|Ph\.D\.?\s+student|doctoral student)',
            r'(M\.Tech\.?\s+student|MTech student|masters student)',
            r'(faculty|professor|assistant professor|associate professor)',
            r'(research scholar|research associate|postdoc)'
        ]
        
        text_lower = text.lower()
        
        # Extract departments
        for pattern in dept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 2:
                    entities['departments'].append(match)
        
        # Extract names
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match.split()) >= 2:
                    entities['faculty_names'].append(match)
        
        # Extract research areas
        for pattern in research_patterns:
            matches = re.findall(pattern, text_lower)
            entities['research_areas'].extend(matches)
        
        # Extract positions
        for pattern in position_patterns:
            matches = re.findall(pattern, text_lower)
            entities['positions'].extend(matches)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def create_semantic_chunks(self, text, base_metadata):
        """Create chunks that preserve semantic coherence"""
        if not text or len(text.strip()) < 50:
            return []
        
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        current_entities = {
            'faculty_names': set(),
            'departments': set(),
            'research_areas': set(),
            'positions': set()
        }
        
        for paragraph in paragraphs:
            paragraph_tokens = self.estimate_tokens(paragraph)
            paragraph_entities = self.extract_entities(paragraph)
            
            # If this paragraph alone is too big, split it further
            if paragraph_tokens > self.max_tokens:
                sentences = re.split(r'[.!?]+\s+', paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence_tokens = self.estimate_tokens(sentence)
                    
                    # Check if adding sentence exceeds max_tokens
                    if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                        # Save current chunk
                        chunk_metadata = self.create_chunk_metadata(
                            base_metadata, current_entities, current_chunk
                        )
                        chunks.append({
                            'text': current_chunk.strip(),
                            'token_count': current_tokens,
                            'metadata': chunk_metadata
                        })
                        
                        # Start new chunk with overlap
                        overlap_text = self.create_overlap(current_chunk)
                        current_chunk = overlap_text + sentence
                        current_tokens = self.estimate_tokens(current_chunk)
                        current_entities = {k: set() for k in current_entities.keys()}
                    else:
                        # Add sentence to current chunk
                        if current_chunk:
                            current_chunk += '. ' + sentence
                        else:
                            current_chunk = sentence
                        current_tokens += sentence_tokens
                    
                    # Update entities
                    sent_entities = self.extract_entities(sentence)
                    for key in current_entities:
                        current_entities[key].update(sent_entities.get(key, []))
            else:
                # Check if adding paragraph exceeds max_tokens
                if current_tokens + paragraph_tokens > self.max_tokens and current_chunk:
                    # Save current chunk
                    chunk_metadata = self.create_chunk_metadata(
                        base_metadata, current_entities, current_chunk
                    )
                    chunks.append({
                        'text': current_chunk.strip(),
                        'token_count': current_tokens,
                        'metadata': chunk_metadata
                    })
                    
                    # Start new chunk
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
                    current_entities = {k: set() for k in current_entities.keys()}
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += '\n\n' + paragraph
                    else:
                        current_chunk = paragraph
                    current_tokens += paragraph_tokens
                
                # Update entities from paragraph
                for key in current_entities:
                    current_entities[key].update(paragraph_entities.get(key, []))
        
        # Add final chunk if it meets minimum requirements
        if current_chunk.strip() and current_tokens >= self.min_tokens:
            chunk_metadata = self.create_chunk_metadata(
                base_metadata, current_entities, current_chunk
            )
            chunks.append({
                'text': current_chunk.strip(),
                'token_count': current_tokens,
                'metadata': chunk_metadata
            })
        
        return chunks
    
    def create_overlap(self, text):
        """Create overlap text from end of previous chunk"""
        words = text.split()
        if len(words) <= self.overlap_tokens:
            return text + '. '
        
        overlap_words = words[-self.overlap_tokens:]
        return ' '.join(overlap_words) + '. '
    
    def create_chunk_metadata(self, base_metadata, entities, chunk_text):
        """Create enriched metadata for chunk"""
        metadata = base_metadata.copy()
        
        # Add extracted entities
        metadata.update({
            'chunk_faculty_names': list(entities['faculty_names']),
            'chunk_departments': list(entities['departments']),
            'chunk_research_areas': list(entities['research_areas']),
            'chunk_positions': list(entities['positions']),
            'has_faculty_info': len(entities['faculty_names']) > 0,
            'has_research_info': len(entities['research_areas']) > 0,
            'chunk_type': self.classify_chunk(chunk_text, entities)
        })
        
        # Add hierarchical information if available
        url = base_metadata.get('url')
        if url and url in self.hierarchy_map:
            hierarchy_info = self.hierarchy_map[url]
            metadata.update({
                'hierarchy_level': hierarchy_info.get('level'),
                'parent_id': hierarchy_info.get('parent_id'),
                'children_ids': hierarchy_info.get('children_ids', []),
                'node_type': hierarchy_info.get('node_type')
            })
        
        return metadata
    
    def classify_chunk(self, text, entities):
        """Classify chunk type based on content"""
        text_lower = text.lower()
        
        if any(pos in text_lower for pos in ['phd', 'ph.d', 'doctoral', 'thesis']):
            return 'student_research'
        elif any(word in text_lower for word in ['faculty', 'professor', 'dr.', 'prof.']):
            return 'faculty_info'
        elif any(word in text_lower for word in ['research', 'project', 'publication']):
            return 'research_activity'
        elif any(word in text_lower for word in ['course', 'program', 'curriculum']):
            return 'academic_program'
        elif any(word in text_lower for word in ['news', 'event', 'seminar', 'conference']):
            return 'news_events'
        else:
            return 'general'
    
    def process_dataset(self, input_file, output_file):
        """Process entire dataset with smart chunking"""
        print("Starting SMART chunking process...")
        print("Input: {}".format(input_file))
        print("Output: {}".format(output_file))
        
        if not os.path.exists(input_file):
            print("Error: Input file '{}' not found!".format(input_file))
            return []
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        all_chunks = []
        total_docs = 0
        processed_docs = 0
        chunk_types = {}
        
        # Use codecs for Python 3.6 compatibility
        with codecs.open(input_file, 'r', 'utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    doc = json.loads(line.strip())
                    total_docs += 1
                    
                    # Get text content
                    text_content = doc.get('cleaned_text', '') or doc.get('text', '')
                    
                    if text_content and len(text_content.strip()) > 100:
                        # Prepare base metadata
                        base_metadata = {
                            'url': doc.get('url', ''),
                            'title': doc.get('title', ''),
                            'domain': doc.get('domain', ''),
                            'original_content_length': doc.get('content_length', 0),
                            'original_word_count': doc.get('word_count', 0),
                            'source_doc_id': line_num
                        }
                        
                        # Create semantic chunks
                        text_chunks = self.create_semantic_chunks(text_content, base_metadata)
                        
                        # Create formatted chunks
                        for chunk_idx, chunk in enumerate(text_chunks):
                            chunk_data = {
                                'chunk_id': "{}_{}".format(line_num, chunk_idx),
                                'doc_id': line_num,
                                'chunk_index': chunk_idx,
                                'total_chunks': len(text_chunks),
                                'chunk_text': chunk['text'],
                                'token_count': chunk['token_count'],
                                'metadata': chunk['metadata']
                            }
                            all_chunks.append(chunk_data)
                            
                            # Track chunk types
                            chunk_type = chunk['metadata']['chunk_type']
                            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                        
                        processed_docs += 1
                    
                    if total_docs % 50 == 0:
                        print("Processed {} docs, created {} chunks...".format(total_docs, len(all_chunks)))
                
                except Exception as e:
                    print("Error processing line {}: {}".format(line_num + 1, e))
                    continue
        
        # Save chunks using codecs
        with codecs.open(output_file, 'w', 'utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        # Print detailed statistics
        self.print_detailed_stats(total_docs, processed_docs, all_chunks, chunk_types)
        
        return all_chunks
    
    def print_detailed_stats(self, total_docs, processed_docs, chunks, chunk_types):
        """Print comprehensive statistics"""
        if not chunks:
            print("No chunks created!")
            return
        
        token_counts = [chunk['token_count'] for chunk in chunks]
        avg_tokens = sum(token_counts) / len(token_counts)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        avg_chunks_per_doc = float(len(chunks)) / processed_docs if processed_docs > 0 else 0
        
        # Count chunks with entities
        chunks_with_faculty = sum(1 for c in chunks if c['metadata'].get('has_faculty_info', False))
        chunks_with_research = sum(1 for c in chunks if c['metadata'].get('has_research_info', False))
        
        print("\nSMART Chunking completed!")
        print("Statistics:")
        print("  Total documents: {}".format(total_docs))
        print("  Processed documents: {}".format(processed_docs))
        print("  Total chunks created: {}".format(len(chunks)))
        print("  Average chunks per document: {:.1f}".format(avg_chunks_per_doc))
        print("  Average tokens per chunk: {:.0f}".format(avg_tokens))
        print("  Token range: {} - {}".format(min_tokens, max_tokens))
        print("  Chunks with faculty info: {} ({:.1f}%)".format(
            chunks_with_faculty, chunks_with_faculty * 100.0 / len(chunks)))
        print("  Chunks with research info: {} ({:.1f}%)".format(
            chunks_with_research, chunks_with_research * 100.0 / len(chunks)))
        
        print("\nChunk Types:")
        for chunk_type, count in sorted(chunk_types.items()):
            percentage = count * 100.0 / len(chunks)
            print("  {}: {} ({:.1f}%)".format(chunk_type, count, percentage))
        
        # Show sample chunks with metadata
        print("\nSample chunks with metadata:")
        for i in range(min(3, len(chunks))):
            chunk = chunks[i]
            metadata = chunk['metadata']
            print("\n--- Chunk {} ---".format(i+1))
            print("ID: {}".format(chunk['chunk_id']))
            print("Type: {}".format(metadata.get('chunk_type', 'unknown')))
            print("Tokens: {}".format(chunk['token_count']))
            
            if metadata.get('chunk_faculty_names'):
                print("Faculty: {}".format(', '.join(metadata['chunk_faculty_names'])))
            if metadata.get('chunk_departments'):
                print("Departments: {}".format(', '.join(metadata['chunk_departments'])))
            if metadata.get('chunk_research_areas'):
                print("Research: {}".format(', '.join(metadata['chunk_research_areas'])))
            
            print("Text preview: {}...".format(chunk['chunk_text'][:200]))

def main():
    # Configuration for coherent, metadata-rich chunks
    MIN_TOKENS = 200
    MAX_TOKENS = 500
    OVERLAP_TOKENS = 30
    
    # File paths
    input_file = 'data/processed/cds_cleaned.jsonl'
    output_file = 'data/processed/cds_smart_chunks.jsonl'
    hierarchy_file = 'cds_hierarchical_corpus.json'  # Hierarchical structure
    
    print("Starting SMART data chunking...")
    print("Configuration:")
    print("   Token range per chunk: {} - {}".format(MIN_TOKENS, MAX_TOKENS))
    print("   Overlap tokens: {}".format(OVERLAP_TOKENS))
    print("   Focus: Faculty names, departments, research areas")
    print("   Hierarchy preservation: ENABLED")
    
    # Initialize smart chunker with hierarchy
    chunker = CleanChunker(
        min_tokens=MIN_TOKENS, 
        max_tokens=MAX_TOKENS, 
        overlap_tokens=OVERLAP_TOKENS,
        hierarchy_file=hierarchy_file
    )
    
    # Process dataset
    chunks = chunker.process_dataset(input_file, output_file)
    
    print("\nSMART chunking process completed!")
    print("Chunks saved to: {}".format(output_file))
    print("Each chunk contains coherent content with faculty/department metadata!")
    print("Hierarchical relationships preserved!")

if __name__ == "__main__":
    main()