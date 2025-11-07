import asyncio
import aiohttp
import json
import hashlib
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Set
import time
from collections import deque
import re

@dataclass
class PageNode:
    """Represents a page in the hierarchical structure"""
    url: str
    title: str
    content: str
    level: int
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    node_type: str = "general"
    metadata: Dict = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def node_id(self):
        """Generate unique ID based on URL"""
        return hashlib.md5(self.url.encode()).hexdigest()

class HierarchicalCDSCrawler:
    def __init__(self, base_url: str = "https://cds.iisc.ac.in", max_concurrent: int = 10):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.visited_urls: Set[str] = set()
        self.nodes: Dict[str, PageNode] = {}
        self.faculty_websites: Dict[str, Dict[str, Optional[str]]] = {}  # faculty_url -> {faculty_site, lab_site}
        
        # Rate limiting
        self.request_delay = 0.1
        self.last_request_time = 0
        
    async def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch a single page with error handling"""
        await self._rate_limit()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30), 
                                   headers={'User-Agent': 'Mozilla/5.0'}) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    print(f"  ⚠ Error {response.status} fetching {url}")
                    return None
        except asyncio.TimeoutError:
            print(f"  ⚠ Timeout fetching {url}")
            return None
        except Exception as e:
            print(f"  ⚠ Exception fetching {url}: {str(e)[:100]}")
            return None
    
    def clean_content(self, soup: BeautifulSoup, url: str = "") -> str:
        """Remove boilerplate and extract main content - IMPROVED with link text preservation"""
        # Clone soup to avoid modifying original
        soup_copy = BeautifulSoup(str(soup), 'html.parser')
        
        # IMPORTANT: Extract link texts BEFORE removing elements
        # This preserves student names and other important linked content
        link_texts = []
        for a in soup_copy.find_all('a'):
            text = a.get_text(strip=True)
            if text and len(text) > 2 and len(text) < 100:  # Reasonable name/title length
                link_texts.append(text)
        
        # Remove scripts, styles, navigation, headers, footers
        for element in soup_copy.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
            element.decompose()
        
        # Remove common boilerplate classes
        for element in soup_copy.find_all(class_=re.compile(r'menu|navigation|sidebar|footer|header|cookie|popup', re.I)):
            element.decompose()
        
        # Try to find main content area - multiple strategies
        main_content = None
        
        # Strategy 1: Look for main tag or content div
        main_content = (
            soup_copy.find('main') or 
            soup_copy.find('article') or
            soup_copy.find('div', id=re.compile(r'content|main', re.I)) or
            soup_copy.find('div', class_=re.compile(r'content|main|body', re.I))
        )
        
        # Strategy 2: If no main content found, use body but be more careful
        if not main_content or len(main_content.get_text(strip=True)) < 100:
            main_content = soup_copy.body
        
        if main_content:
            # Extract text with better formatting
            text_parts = []
            
            # Get all text-containing elements including links
            for elem in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'div', 'span', 'a']):
                text = elem.get_text(separator=' ', strip=True)
                if text and len(text) > 2:  # Ignore very short fragments
                    text_parts.append(text)
            
            # Combine and clean
            full_text = ' '.join(text_parts)
            full_text = re.sub(r'\s+', ' ', full_text)  # Normalize whitespace
            full_text = re.sub(r'(\w)\s+([.,!?;:])', r'\1\2', full_text)  # Fix punctuation
            
            # Add extracted link texts that might have been missed
            for link_text in link_texts:
                if link_text not in full_text:
                    full_text += f" {link_text}"
            
            return full_text.strip()
        
        return soup_copy.get_text(separator=' ', strip=True)
    
    def extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract page title - IMPROVED"""
        # Try multiple title sources in order of preference
        title = None
        
        # 1. h1 tag
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)
        
        # 2. meta og:title
        if not title or len(title) < 3:
            og_title = soup.find('meta', property='og:title')
            if og_title:
                title = og_title.get('content', '')
        
        # 3. title tag
        if not title or len(title) < 3:
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
        
        # 4. Fallback to URL
        if not title or len(title) < 3:
            path = urlparse(url).path
            title = path.split('/')[-1] or path.split('/')[-2] or 'Page'
            title = title.replace('-', ' ').replace('_', ' ').title()
        
        # Clean title
        title = re.sub(r'\s+', ' ', title)
        return title[:200]  # Limit length
    
    def classify_node_type(self, url: str, title: str) -> str:
        """Classify the node type based on URL patterns - IMPROVED"""
        path = urlparse(url).path.lower()
        
        if path == '/' or path == '':
            return 'root'
        
        # Faculty detection - multiple patterns
        if (re.search(r'/faculty/[^/]+/?$', path) or 
            re.search(r'/people/[^/]+/?$', path) or
            'faculty' in path.split('/')[-1]):
            return 'faculty'
        
        # People section
        if path == '/people' or path == '/people/':
            return 'section'
        
        # Lab/Research group
        if re.search(r'/(lab|research|group)/[^/]+/?$', path):
            return 'lab'
        
        # Main sections
        if any(section in path for section in ['/about', '/research', '/academics', '/news', '/resources']):
            if path.count('/') <= 2:
                return 'section'
            else:
                return 'subsection'
        
        # Subsections
        if path.count('/') == 2:
            return 'subsection'
        
        return 'page'
    
    def extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Extract all relevant links from a page"""
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            absolute_url = urljoin(current_url, href)
            
            # Include links within CDS domain
            if absolute_url.startswith(self.base_url):
                normalized = absolute_url.split('#')[0].rstrip('/')
                # Avoid duplicates and certain file types
                if not re.search(r'\.(pdf|jpg|png|gif|doc|docx|zip)$', normalized.lower()):
                    links.append(normalized)
        
        return list(set(links))
    
    def detect_faculty_and_lab_websites(self, soup: BeautifulSoup, base_url: str) -> Dict[str, Optional[str]]:
        """
        Detect Faculty Website and Lab Website links from a faculty profile page.
        Returns dict with 'faculty_website' and 'lab_website' keys.
        IMPROVED: Better filtering to avoid false positives like /category/openings
        """
        result = {
            'faculty_website': None,
            'lab_website': None
        }
        
        # Look for all links with their text and context
        all_links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True).lower()
            absolute_url = urljoin(base_url, href)
            
            # Get parent context for better matching
            parent_text = ""
            if a.parent:
                parent_text = a.parent.get_text(strip=True).lower()
            
            all_links.append((text, absolute_url, parent_text, a))
        
        # Exact keyword matching for faculty website
        faculty_exact_keywords = [
            'faculty website', 'personal website', 'faculty page', 'personal page', 'homepage'
        ]
        
        # Exact keyword matching for lab website  
        lab_exact_keywords = [
            'lab website', 'laboratory website', 'lab page', 'laboratory page',
            'research group', 'group website', 'research lab'
        ]
        
        # Phase 1: Look for exact keyword matches (most reliable)
        for text, url, parent_text, anchor in all_links:
            # Skip certain URLs that are definitely not what we want
            skip_patterns = ['/category/', '/tag/', '/archive/', '/feed/', '/wp-', '/openings']
            if any(pattern in url.lower() for pattern in skip_patterns):
                continue
            
            # Faculty website detection
            if any(keyword in text for keyword in faculty_exact_keywords):
                if result['faculty_website'] is None:
                    result['faculty_website'] = url.split('#')[0].rstrip('/')
                    continue
            
            # Lab website detection - require exact match or very specific context
            if any(keyword in text for keyword in lab_exact_keywords):
                # Additional validation: lab URLs should not be generic site pages
                if not any(bad in url.lower() for bad in ['/about', '/contact', '/news', '/people']):
                    if result['lab_website'] is None:
                        result['lab_website'] = url.split('#')[0].rstrip('/')
                        continue
        
        # Phase 2: Look in structured sections if we haven't found them yet
        if result['faculty_website'] is None or result['lab_website'] is None:
            # Look for links in info/contact sections
            for section in soup.find_all(['div', 'section', 'ul', 'dl'], 
                                        class_=re.compile(r'info|contact|link|detail|profile', re.I)):
                for a in section.find_all('a', href=True):
                    href = a['href']
                    text = a.get_text(strip=True).lower()
                    absolute_url = urljoin(base_url, href)
                    
                    # Skip unwanted patterns
                    skip_patterns = ['/category/', '/tag/', '/archive/', '/feed/', '/wp-', '/openings']
                    if any(pattern in absolute_url.lower() for pattern in skip_patterns):
                        continue
                    
                    if result['faculty_website'] is None:
                        if 'faculty' in text or 'personal' in text or 'homepage' in text:
                            result['faculty_website'] = absolute_url.split('#')[0].rstrip('/')
                    
                    if result['lab_website'] is None:
                        if 'lab' in text or 'research' in text:
                            # Ensure it's not a generic page
                            if not any(bad in absolute_url.lower() for bad in ['/about', '/contact', '/news', '/people']):
                                result['lab_website'] = absolute_url.split('#')[0].rstrip('/')
        
        # Phase 3: Look for definition lists (dt/dd pairs)
        if result['faculty_website'] is None or result['lab_website'] is None:
            for dt in soup.find_all('dt'):
                dt_text = dt.get_text(strip=True).lower()
                dd = dt.find_next_sibling('dd')
                
                if dd:
                    link = dd.find('a', href=True)
                    if link:
                        absolute_url = urljoin(base_url, link['href'])
                        
                        # Skip unwanted patterns
                        skip_patterns = ['/category/', '/tag/', '/archive/', '/feed/', '/wp-', '/openings']
                        if any(pattern in absolute_url.lower() for pattern in skip_patterns):
                            continue
                        
                        if result['faculty_website'] is None:
                            if 'faculty' in dt_text or 'personal' in dt_text or 'homepage' in dt_text:
                                result['faculty_website'] = absolute_url.split('#')[0].rstrip('/')
                        
                        if result['lab_website'] is None:
                            if 'lab' in dt_text or 'research' in dt_text:
                                if not any(bad in absolute_url.lower() for bad in ['/about', '/contact', '/news', '/people']):
                                    result['lab_website'] = absolute_url.split('#')[0].rstrip('/')
        
        return result
    
    async def crawl_page(self, session: aiohttp.ClientSession, url: str, parent_id: Optional[str], level: int) -> Optional[tuple]:
        """Crawl a single page and create a node"""
        if url in self.visited_urls:
            return None
        
        self.visited_urls.add(url)
        indent = '  ' * level
        print(f"{indent}📄 Crawling [{level}]: {url}")
        
        html = await self.fetch_page(session, url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract information
        title = self.extract_title(soup, url)
        content = self.clean_content(soup, url)
        node_type = self.classify_node_type(url, title)
        
        print(f"{indent}   ✓ Type: {node_type}, Title: {title[:50]}, Content: {len(content)} chars")
        
        # Create node
        node = PageNode(
            url=url,
            title=title,
            content=content,
            level=level,
            parent_id=parent_id,
            node_type=node_type,
            metadata={'crawl_timestamp': time.time()}
        )
        
        # For faculty nodes, detect their personal and lab websites
        if node_type == 'faculty':
            websites = self.detect_faculty_and_lab_websites(soup, url)
            if websites['faculty_website'] or websites['lab_website']:
                self.faculty_websites[url] = websites
                print(f"{indent}   🔗 Found websites:")
                if websites['faculty_website']:
                    print(f"{indent}      - Faculty: {websites['faculty_website']}")
                if websites['lab_website']:
                    print(f"{indent}      - Lab: {websites['lab_website']}")
        
        # Store node
        self.nodes[node.node_id] = node
        
        # Update parent's children
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node.node_id)
        
        # Extract child links
        child_links = self.extract_links(soup, url)
        
        return node, child_links
    
    async def fetch_and_consolidate_websites(self, session: aiohttp.ClientSession):
        """
        Fetch faculty personal websites and lab websites, then consolidate them
        into the faculty node content.
        """
        print("\n" + "="*80)
        print("🔗 PHASE 2: Consolidating Faculty with Personal & Lab Websites")
        print("="*80)
        
        faculty_nodes = [node for node in self.nodes.values() if node.node_type == 'faculty']
        print(f"Found {len(faculty_nodes)} faculty nodes")
        
        for i, faculty_node in enumerate(faculty_nodes, 1):
            websites = self.faculty_websites.get(faculty_node.url, {})
            faculty_site = websites.get('faculty_website')
            lab_site = websites.get('lab_website')
            
            if not faculty_site and not lab_site:
                continue
            
            print(f"\n[{i}/{len(faculty_nodes)}] 👤 {faculty_node.title}")
            
            consolidated_parts = [faculty_node.content]
            
            # Fetch Faculty Website
            if faculty_site:
                print(f"  📥 Fetching Faculty Website: {faculty_site}")
                html = await self.fetch_page(session, faculty_site)
                if html:
                    soup = BeautifulSoup(html, 'html.parser')
                    content = self.clean_content(soup, faculty_site)
                    title = self.extract_title(soup, faculty_site)
                    
                    if len(content) > 100:  # Only add if substantial content
                        consolidated_parts.append(f"\n\n{'='*60}\nFACULTY PERSONAL WEBSITE: {title}\n{'='*60}\n{content}")
                        faculty_node.metadata['faculty_website'] = faculty_site
                        faculty_node.metadata['faculty_website_chars'] = len(content)
                        print(f"     ✓ Added {len(content)} chars")
                    else:
                        print(f"     ⚠ Insufficient content ({len(content)} chars)")
            
            # Fetch Lab Website
            if lab_site:
                print(f"  📥 Fetching Lab Website: {lab_site}")
                html = await self.fetch_page(session, lab_site)
                if html:
                    soup = BeautifulSoup(html, 'html.parser')
                    content = self.clean_content(soup, lab_site)
                    title = self.extract_title(soup, lab_site)
                    
                    if len(content) > 100:
                        consolidated_parts.append(f"\n\n{'='*60}\nLAB/RESEARCH GROUP WEBSITE: {title}\n{'='*60}\n{content}")
                        faculty_node.metadata['lab_website'] = lab_site
                        faculty_node.metadata['lab_website_chars'] = len(content)
                        print(f"     ✓ Added {len(content)} chars")
                    else:
                        print(f"     ⚠ Insufficient content ({len(content)} chars)")
            
            # Update faculty node with consolidated content
            faculty_node.content = "\n".join(consolidated_parts)
            faculty_node.metadata['has_consolidated_websites'] = True
            faculty_node.metadata['total_content_chars'] = len(faculty_node.content)
            
            self.nodes[faculty_node.node_id] = faculty_node
    
    async def crawl_bfs(self):
        """Breadth-first crawl to build hierarchy"""
        async with aiohttp.ClientSession() as session:
            print("="*80)
            print("🕷️  PHASE 1: Crawling Site Hierarchy")
            print("="*80 + "\n")
            
            # Start with root
            queue = deque([(self.base_url, None, 0)])
            
            while queue:
                # Process batch of URLs concurrently
                batch = []
                batch_size = min(self.max_concurrent, len(queue))
                
                for _ in range(batch_size):
                    if queue:
                        batch.append(queue.popleft())
                
                # Crawl batch
                tasks = [self.crawl_page(session, url, parent_id, level) for url, parent_id, level in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and add children to queue
                for result in results:
                    if isinstance(result, Exception):
                        print(f"  ⚠ Error in batch: {result}")
                        continue
                    
                    if result is None:
                        continue
                    
                    node, child_links = result
                    
                    # Add unvisited children to queue (limit depth)
                    if node.level < 4:  # Max depth
                        for child_url in child_links[:50]:  # Limit children per page
                            if child_url not in self.visited_urls:
                                queue.append((child_url, node.node_id, node.level + 1))
            
            # Phase 2: Fetch and consolidate faculty websites
            await self.fetch_and_consolidate_websites(session)
    
    def export_to_json(self, filename: str = "cds_hierarchical_corpus.json"):
        """Export the hierarchical structure to JSON"""
        export_data = {
            'metadata': {
                'base_url': self.base_url,
                'total_nodes': len(self.nodes),
                'crawl_timestamp': time.time(),
                'total_faculty': len([n for n in self.nodes.values() if n.node_type == 'faculty']),
                'faculty_with_consolidated': len([n for n in self.nodes.values() if n.metadata.get('has_consolidated_websites')]),
                'faculty_with_personal_site': len([n for n in self.nodes.values() if n.metadata.get('faculty_website')]),
                'faculty_with_lab_site': len([n for n in self.nodes.values() if n.metadata.get('lab_website')])
            },
            'nodes': [asdict(node) for node in self.nodes.values()]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Exported {len(self.nodes)} nodes to {filename}")
    
    def print_hierarchy(self, node_id: Optional[str] = None, indent: int = 0, max_depth: int = 3):
        """Print the hierarchy tree"""
        if indent > max_depth:
            return
            
        if node_id is None:
            root_nodes = [n for n in self.nodes.values() if n.parent_id is None]
            for root in root_nodes:
                self.print_hierarchy(root.node_id, 0, max_depth)
            return
        
        node = self.nodes.get(node_id)
        if not node:
            return
        
        prefix = "  " * indent
        
        # Show website info for faculty
        website_info = ""
        if node.node_type == 'faculty':
            indicators = []
            if node.metadata.get('faculty_website'):
                chars = node.metadata.get('faculty_website_chars', 0)
                indicators.append(f"Faculty:{chars}c")
            if node.metadata.get('lab_website'):
                chars = node.metadata.get('lab_website_chars', 0)
                indicators.append(f"Lab:{chars}c")
            if indicators:
                website_info = f" [{', '.join(indicators)}]"
        
        content_preview = f" ({len(node.content)} chars)" if node.content else " (empty)"
        print(f"{prefix}├─ [{node.node_type}] {node.title[:50]}{website_info}{content_preview}")
        
        for child_id in node.children_ids[:10]:  # Limit children shown
            self.print_hierarchy(child_id, indent + 1, max_depth)
    
    def get_statistics(self):
        """Print crawl statistics"""
        node_types = {}
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        
        print("\n" + "="*80)
        print("📊 CRAWL STATISTICS")
        print("="*80)
        print(f"Total pages crawled: {len(self.nodes)}")
        print(f"Total URLs visited: {len(self.visited_urls)}")
        
        print("\n📑 Node Types:")
        for node_type, count in sorted(node_types.items()):
            print(f"  {node_type:.<20} {count:>4}")
        
        total_faculty = len([n for n in self.nodes.values() if n.node_type == 'faculty'])
        faculty_with_consolidated = len([n for n in self.nodes.values() if n.metadata.get('has_consolidated_websites')])
        faculty_with_personal = len([n for n in self.nodes.values() if n.metadata.get('faculty_website')])
        faculty_with_lab = len([n for n in self.nodes.values() if n.metadata.get('lab_website')])
        
        print(f"\n👥 Faculty Information:")
        print(f"  Total faculty nodes: {total_faculty}")
        print(f"  With consolidated websites: {faculty_with_consolidated}")
        print(f"  With faculty personal site: {faculty_with_personal}")
        print(f"  With lab website: {faculty_with_lab}")
        
        # Content statistics
        total_chars = sum(len(n.content) for n in self.nodes.values())
        avg_chars = total_chars / len(self.nodes) if self.nodes else 0
        print(f"\n📝 Content Statistics:")
        print(f"  Total content: {total_chars:,} characters")
        print(f"  Average per node: {avg_chars:,.0f} characters")

async def main():
    """Main execution function"""
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "HIERARCHICAL CDS WEBSITE CRAWLER" + " "*26 + "║")
    print("╚" + "="*78 + "╝\n")
    
    crawler = HierarchicalCDSCrawler(
        base_url="https://cds.iisc.ac.in",
        max_concurrent=8  # Reduced for stability
    )
    
    start_time = time.time()
    await crawler.crawl_bfs()
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Crawling completed in {elapsed_time:.1f} seconds")
    
    crawler.get_statistics()
    
    print("\n" + "="*80)
    print("🌳 SITE HIERARCHY (sample)")
    print("="*80)
    crawler.print_hierarchy(max_depth=3)
    
    crawler.export_to_json("cds_hierarchical_corpus.json")
    
    print("\n✨ All done! Ready for embedding and PostgreSQL storage.\n")

if __name__ == "__main__":
    asyncio.run(main())