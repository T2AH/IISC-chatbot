"""
Scrapy pipelines for processing crawled items
"""

import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Dict, Any
from loguru import logger
import os


class ValidationPipeline:
    """Validate crawled items"""
    
    def process_item(self, item, spider):
        """Validate required fields"""
        required_fields = ['url', 'title', 'raw_html']
        
        for field in required_fields:
            if field not in item or not item[field]:
                logger.warning(f"Missing required field '{field}' in item: {item.get('url', 'unknown')}")
                raise ValueError(f"Missing required field: {field}")
        
        return item


class TextCleaningPipeline:
    """Clean and extract text from HTML - optimized for RAG/embedding pipeline"""
    
    def __init__(self):
        self.unwanted_tags = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']
    
    def process_item(self, item, spider):
        """Extract and clean text from HTML for optimal RAG performance"""
        try:
            soup = BeautifulSoup(item['raw_html'], 'lxml')
            
            # Remove unwanted tags
            for tag in self.unwanted_tags:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Extract main content with structure preservation
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main'))
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            # Clean whitespace while preserving sentence boundaries
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Fix punctuation spacing
            text = text.strip()
            
            # Store cleaned text
            item['cleaned_text'] = text
            
            # Calculate word count
            words = text.split()
            item['word_count'] = len(words)
            
            # Extract metadata for RAG (helps with context)
            item['metadata'] = self._extract_metadata(soup, item)
            
            # Calculate text quality score for RAG
            item['text_quality_score'] = self._calculate_text_quality(text, words)
            
            logger.debug(f"Cleaned text for {item['url']}: {item['word_count']} words (quality: {item['text_quality_score']:.2f})")
            
        except Exception as e:
            logger.error(f"Error cleaning text for {item['url']}: {e}")
            item['cleaned_text'] = ""
            item['word_count'] = 0
            item['metadata'] = {}
            item['text_quality_score'] = 0.0
        
        return item
    
    def _extract_metadata(self, soup, item):
        """Extract metadata useful for RAG context"""
        metadata = {}
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
        
        # Extract keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            metadata['keywords'] = meta_keywords.get('content', '')
        
        # Extract headings structure (helps with chunking)
        headings = []
        for i in range(1, 4):  # h1, h2, h3
            for heading in soup.find_all(f'h{i}'):
                headings.append({
                    'level': i,
                    'text': heading.get_text(strip=True)
                })
        metadata['headings'] = headings[:10]  # Limit to top 10
        
        # Extract author if available
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            metadata['author'] = author_meta.get('content', '')
        
        return metadata
    
    def _calculate_text_quality(self, text, words):
        """Calculate text quality score for RAG suitability (0-1)"""
        if not words:
            return 0.0
        
        score = 0.5  # Base score
        
        # Longer text is better (up to a point)
        word_count = len(words)
        if word_count >= 200:
            score += 0.2
        elif word_count >= 100:
            score += 0.1
        
        # Check for research indicators
        research_terms = ['research', 'publication', 'paper', 'study', 'analysis', 'journal', 
                         'conference', 'proceedings', 'phd', 'thesis', 'professor']
        research_count = sum(1 for term in research_terms if term in text.lower())
        if research_count >= 3:
            score += 0.2
        elif research_count >= 1:
            score += 0.1
        
        # Check vocabulary diversity (unique words / total words)
        unique_words = len(set(words))
        diversity = unique_words / len(words)
        if diversity > 0.5:
            score += 0.1
        
        return min(1.0, score)


class PageTypeClassificationPipeline:
    """Classify page type based on URL patterns and keywords"""
    
    def __init__(self, config):
        self.config = config
        self.page_types = config.get('crawler', 'page_types', default={})
    
    @classmethod
    def from_crawler(cls, crawler):
        from src.config import config
        return cls(config)
    
    def process_item(self, item, spider):
        """Classify page type"""
        url = item['url'].lower()
        text = item.get('cleaned_text', '').lower()
        
        # Check each page type
        for page_type, patterns in self.page_types.items():
            # Check URL patterns
            url_patterns = patterns.get('url_patterns', [])
            if any(pattern in url for pattern in url_patterns):
                item['page_type'] = page_type
                logger.debug(f"Classified {item['url']} as {page_type} (URL pattern)")
                return item
            
            # Check keywords in text
            keywords = patterns.get('keywords', [])
            keyword_matches = sum(1 for keyword in keywords if keyword in text)
            
            if keyword_matches >= 2:  # At least 2 keyword matches
                item['page_type'] = page_type
                logger.debug(f"Classified {item['url']} as {page_type} (keywords)")
                return item
        
        # Default to 'general' if no specific type found
        item['page_type'] = 'general'
        return item


class JsonExportPipeline:
    """Export items to JSON files"""
    
    def __init__(self):
        self.output_dir = Path('data/crawled_pages')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.files = {}
        self.item_count = 0
    
    def open_spider(self, spider):
        """Initialize file handles"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.file_path = self.output_dir / f'pages_{spider.name}_{timestamp}.jsonl'
        # Open with line buffering (1) for real-time writes
        self.file = open(self.file_path, 'w', encoding='utf-8', buffering=1)
        self.item_count = 0
        logger.info(f"✓ Exporting crawled items to {self.file_path}")
        print(f"\n✓ Export file created: {self.file_path}\n", flush=True)
    
    def close_spider(self, spider):
        """Close file handles"""
        if hasattr(self, 'file') and self.file:
            try:
                self.file.flush()
                os.fsync(self.file.fileno())
            except Exception:
                # If fsync isn't available or fails, ignore but ensure close
                pass
            self.file.close()
            logger.info(f"✓ Closed export file: {self.file_path} ({self.item_count} items)")
            print(f"\n✓ Export complete: {self.file_path} ({self.item_count} items)\n")
    
    def process_item(self, item, spider):
        """Write item to JSON file with RAG-optimized structure"""
        try:
            # Generate page ID
            if 'page_id' not in item:
                item['page_id'] = self._generate_page_id(item['url'], item['domain'])
            
            # Add crawl date
            if 'crawl_date' not in item:
                item['crawl_date'] = datetime.now().isoformat()
            
            # Add chunking hints for RAG pipeline
            item['chunking_hints'] = self._generate_chunking_hints(item)
            
            # Prepare item for export (exclude raw HTML to save space)
            export_item = dict(item)
            if 'raw_html' in export_item:
                del export_item['raw_html']
            
            # Write to file and flush immediately for real-time visibility
            line = json.dumps(export_item, ensure_ascii=False) + '\n'
            self.file.write(line)
            try:
                # Flush Python layer and force OS to persist to disk
                self.file.flush()
                os.fsync(self.file.fileno())
            except Exception:
                # If fsync fails (platform dependent), we still flushed the Python buffer
                pass
            
            self.item_count += 1
            
            # Real-time progress indicator for each item
            print(f"[SAVED #{self.item_count}] {export_item.get('url', 'unknown')[:80]}", flush=True)
            
            # Summary logging every 10 items
            if self.item_count % 10 == 0:
                logger.info(f"✓ Exported {self.item_count} items so far...")
                print(f"\n=== Progress: {self.item_count} items exported ===\n", flush=True)
            
        except Exception as e:
            logger.error(f"Error exporting item {item.get('url', 'unknown')}: {e}")
        
        return item
    
    def _generate_chunking_hints(self, item):
        """Generate hints for optimal chunking in RAG pipeline"""
        hints = {
            'suggested_chunk_size': 512,  # Default for embeddings
            'overlap': 50,
            'strategy': 'sentence'  # or 'paragraph'
        }
        
        # Adjust based on page type
        page_type = item.get('page_type', 'general')
        if page_type == 'faculty':
            hints['strategy'] = 'section'  # Faculty pages have clear sections
            hints['suggested_chunk_size'] = 768  # Larger chunks for bio/research
        elif page_type == 'publication':
            hints['strategy'] = 'paragraph'
            hints['suggested_chunk_size'] = 1024  # Abstracts can be longer
        elif page_type == 'lab':
            hints['strategy'] = 'section'
            hints['suggested_chunk_size'] = 512
        
        # Add heading-based boundaries if available
        metadata = item.get('metadata', {})
        if metadata.get('headings'):
            hints['has_headings'] = True
            hints['heading_count'] = len(metadata['headings'])
        
        return hints
    
    def _generate_page_id(self, url: str, domain: str) -> str:
        """Generate unique page ID"""
        # Create ID from domain and URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        domain_short = domain.replace('.', '_')
        return f"{domain_short}_{url_hash}"


class ContentQualityFilterPipeline:
    """Filter low-quality pages to ensure clean dataset"""
    
    def __init__(self):
        self.filtered_count = 0
        self.total_count = 0
    
    def process_item(self, item, spider):
        """Filter out low-quality content"""
        from scrapy.exceptions import DropItem
        
        self.total_count += 1
        url = item.get('url', 'unknown')
        
        # Filter 1: Skip very short pages (< 50 words)
        word_count = item.get('word_count', 0)
        if word_count < 50:
            self.filtered_count += 1
            logger.debug(f"Filtered short page ({word_count} words): {url}")
            raise DropItem(f"Page too short: {word_count} words")
        
        # Filter 2: Skip mostly duplicated content (boilerplate)
        text = item.get('cleaned_text', '')
        if text:
            words = text.split()
            if len(words) > 0:
                unique_words = len(set(words))
                repetition_ratio = 1 - (unique_words / len(words))
                
                if repetition_ratio > 0.7:  # 70% repeated words = boilerplate
                    self.filtered_count += 1
                    logger.debug(f"Filtered boilerplate page (repetition: {repetition_ratio:.2f}): {url}")
                    raise DropItem(f"Boilerplate content (repetition: {repetition_ratio:.2f})")
        
        # Filter 3: Skip redirect landing pages
        response_status = item.get('response_status', 200)
        if response_status in [301, 302, 303, 307, 308]:
            self.filtered_count += 1
            logger.debug(f"Filtered redirect page ({response_status}): {url}")
            raise DropItem(f"Redirect page ({response_status})")
        
        return item
    
    def close_spider(self, spider):
        """Log filtering statistics when spider closes"""
        if self.total_count > 0:
            filter_rate = (self.filtered_count / self.total_count) * 100
            logger.info(f"Quality Filter: Filtered {self.filtered_count}/{self.total_count} pages ({filter_rate:.1f}%)")
            logger.info(f"Quality pages exported: {self.total_count - self.filtered_count}")


class CrawlProgressPipeline:
    """Track crawl progress per domain for better monitoring"""
    
    def open_spider(self, spider):
        """Initialize progress tracking"""
        self.progress_file = Path('data/crawl_progress.json')
        self.progress_file.parent.mkdir(exist_ok=True, parents=True)
        self.progress = self._load_progress()
        self.save_counter = 0
    
    def _load_progress(self):
        """Load existing progress from file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def process_item(self, item, spider):
        """Update progress for each domain"""
        domain = item.get('domain', 'unknown')
        
        if domain not in self.progress:
            self.progress[domain] = {
                'total_pages': 0,
                'started_at': datetime.now().isoformat(),
                'last_crawl': None,
                'page_types': {}
            }
        
        self.progress[domain]['total_pages'] += 1
        self.progress[domain]['last_crawl'] = datetime.now().isoformat()
        
        # Track page types
        page_type = item.get('page_type', 'general')
        if page_type not in self.progress[domain]['page_types']:
            self.progress[domain]['page_types'][page_type] = 0
        self.progress[domain]['page_types'][page_type] += 1
        
        # Save periodically (every 50 items)
        self.save_counter += 1
        if self.save_counter % 50 == 0:
            self._save_progress()
        
        return item
    
    def _save_progress(self):
        """Save progress to file"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving progress: {e}")
    
    def close_spider(self, spider):
        """Final save and summary when spider closes"""
        self._save_progress()
        
        # Log summary
        total_domains = len(self.progress)
        total_pages = sum(d['total_pages'] for d in self.progress.values())
        
        logger.info(f"Crawl Progress Summary:")
        logger.info(f"  Total domains crawled: {total_domains}")
        logger.info(f"  Total pages crawled: {total_pages}")
        logger.info(f"  Progress saved to: {self.progress_file}")
        
        # Show top 5 domains by page count
        top_domains = sorted(self.progress.items(), 
                           key=lambda x: x[1]['total_pages'], 
                           reverse=True)[:5]
        logger.info(f"  Top 5 domains:")
        for domain, stats in top_domains:
            logger.info(f"    - {domain}: {stats['total_pages']} pages")
