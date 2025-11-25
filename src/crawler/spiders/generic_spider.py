"""
Generic Spider - For crawling external faculty/lab websites with NLP filtering
"""

import scrapy
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from typing import Generator, Set
from loguru import logger

from src.crawler.items import PageItem
from src.config import config


class GenericSpider(scrapy.Spider):
    """Generic spider for crawling external faculty and lab websites"""
    
    name = 'generic_spider'
    
    def __init__(self, start_urls=None, allowed_domains=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get configuration
        self.config = config
        
        # Set start URLs
        if start_urls:
            if isinstance(start_urls, str):
                self.start_urls = [start_urls]
            else:
                self.start_urls = start_urls
        else:
            self.start_urls = []
        
        # Set allowed domains
        if allowed_domains:
            if isinstance(allowed_domains, str):
                self.allowed_domains = [allowed_domains]
            else:
                self.allowed_domains = allowed_domains
        else:
            # Extract domains from start URLs
            self.allowed_domains = [urlparse(url).netloc for url in self.start_urls]
        
        # Depth configuration for external sites
        self.max_depth_external = self.config.get('crawler', 'depth_rules', 'external_sites', 'max_depth', default=3)
        
        # Track visited URLs to avoid cycles
        self.visited_urls: Set[str] = set()
        
        logger.info(f"Initialized Generic Spider with start URLs: {self.start_urls}")
        logger.info(f"Allowed domains: {self.allowed_domains}")
    
    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        """Create spider from crawler"""
        spider = super().from_crawler(crawler, *args, **kwargs)
        return spider
    
    def start_requests(self):
        """Generate initial requests"""
        for url in self.start_urls:
            yield scrapy.Request(
                url,
                callback=self.parse,
                meta={
                    'depth': 0,
                    'max_depth': self.max_depth_external,
                    'page_type': 'homepage'
                },
                errback=self.errback_httpbin,
            )
    
    def parse(self, response):
        """Parse response and extract data"""
        try:
            # Check if already visited
            url = response.url
            if url in self.visited_urls:
                logger.debug(f"Skipping already visited URL: {url}")
                return
            
            self.visited_urls.add(url)
            
            # Extract basic information
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Create item
            item = PageItem()
            item['url'] = url
            item['domain'] = domain
            item['title'] = self._extract_title(response)
            item['raw_html'] = response.text
            item['depth'] = response.meta.get('depth', 0)
            item['response_status'] = response.status
            item['content_length'] = len(response.text)
            
            # Calculate relevance score
            item['relevance_score'] = self._calculate_relevance(response)
            
            logger.info(f"Crawled: {url} (depth: {item['depth']}, relevance: {item.get('relevance_score', 0):.2f})")
            
            yield item
            
            # Extract and follow links if relevant enough
            current_depth = response.meta.get('depth', 0)
            max_depth = response.meta.get('max_depth', self.max_depth_external)
            
            # Only follow links if page is relevant
            if current_depth < max_depth and item.get('relevance_score', 0) >= 0.3:
                for link in self._extract_links(response):
                    if link not in self.visited_urls:
                        yield scrapy.Request(
                            link,
                            callback=self.parse,
                            meta={
                                'depth': current_depth + 1,
                                'max_depth': max_depth,
                                'item': item
                            },
                            errback=self.errback_httpbin,
                        )
        
        except Exception as e:
            logger.error(f"Error parsing {response.url}: {e}")
    
    def _calculate_relevance(self, response) -> float:
        """Calculate relevance score based on research-related keywords"""
        try:
            text = response.text.lower()
            
            # Research-related keywords (weighted)
            keyword_weights = {
                'research': 2.0,
                'publication': 2.0,
                'laboratory': 1.5,
                'project': 1.5,
                'professor': 1.5,
                'phd': 1.5,
                'paper': 1.0,
                'conference': 1.0,
                'journal': 1.0,
                'study': 1.0,
                'experiment': 1.0,
                'thesis': 1.5,
                'dissertation': 1.5,
                'faculty': 1.5,
                'lab': 1.5,
                'team': 1.0,
                'collaboration': 1.0,
            }
            
            # Calculate score
            score = 0.0
            total_weight = 0.0
            
            for keyword, weight in keyword_weights.items():
                count = text.count(keyword)
                if count > 0:
                    score += min(count, 5) * weight  # Cap at 5 occurrences per keyword
                    total_weight += weight
            
            # Normalize score (0-1 range)
            if total_weight > 0:
                normalized_score = min(score / (total_weight * 2), 1.0)
            else:
                normalized_score = 0.0
            
            return normalized_score
        
        except Exception as e:
            logger.warning(f"Error calculating relevance for {response.url}: {e}")
            return 0.0
    
    def _extract_title(self, response) -> str:
        """Extract page title"""
        try:
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Try different title sources
            title = None
            
            if soup.title:
                title = soup.title.string
            
            if not title:
                h1 = soup.find('h1')
                if h1:
                    title = h1.get_text(strip=True)
            
            if not title:
                og_title = soup.find('meta', property='og:title')
                if og_title:
                    title = og_title.get('content')
            
            if not title:
                title = response.url.split('/')[-1] or 'Untitled'
            
            return str(title).strip()
        
        except Exception as e:
            logger.warning(f"Error extracting title from {response.url}: {e}")
            return 'Untitled'
    
    def _extract_links(self, response) -> Generator[str, None, None]:
        """Extract and filter links from response"""
        try:
            soup = BeautifulSoup(response.text, 'lxml')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(response.url, href)
                parsed = urlparse(absolute_url)
                
                if self._should_follow_link(parsed, response.url):
                    yield absolute_url
        
        except Exception as e:
            logger.warning(f"Error extracting links from {response.url}: {e}")
    
    def _should_follow_link(self, parsed_url, source_url: str) -> bool:
        """Determine if a link should be followed"""
        # Check scheme
        if parsed_url.scheme not in ['http', 'https']:
            return False
        
        # Check domain
        if not any(domain in parsed_url.netloc for domain in self.allowed_domains):
            return False
        
        # Skip file downloads
        skip_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
                          '.zip', '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif', 
                          '.mp4', '.mp3', '.avi', '.mov']
        
        if any(parsed_url.path.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip fragments
        if parsed_url.fragment and not parsed_url.query:
            full_url_without_fragment = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            source_parsed = urlparse(source_url)
            source_url_without_fragment = f"{source_parsed.scheme}://{source_parsed.netloc}{source_parsed.path}"
            if full_url_without_fragment == source_url_without_fragment:
                return False
        
        return True
    
    def errback_httpbin(self, failure):
        """Handle request failures"""
        logger.error(f"Request failed: {failure.request.url} - {failure.value}")
