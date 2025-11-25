"""
IISc Spider - Structure-aware spider for IISc department websites
"""

import scrapy
from scrapy.exceptions import IgnoreRequest
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError, TimeoutError, ConnectionRefusedError
from urllib.parse import urlparse, urljoin, parse_qs, urlencode, urlunparse
from bs4 import BeautifulSoup
from typing import Generator
from loguru import logger

from src.crawler.items import PageItem
from src.config import config


class IIScSpider(scrapy.Spider):
    """Spider for crawling IISc department websites with structure awareness"""
    
    name = 'iisc_spider'
    
    def __init__(self, start_url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get configuration
        self.config = config
        
        # Set start URLs - All IISc department websites
        if start_url:
            self.start_urls = [start_url]
        else:
            # Comprehensive list of IISc department URLs (including CDS for full crawling)
            self.start_urls = [
                'https://biochem.iisc.ac.in/',
                'https://caf.iisc.ac.in/',
                'https://ces.iisc.ac.in/',
                'https://cidr.iisc.ac.in/',
                'https://cns.iisc.ac.in/',
                'https://mcb.iisc.ac.in/',
                'http://mbu.iisc.ac.in/',
                'https://dbg.iisc.ac.in/',
                'https://ipc.iisc.ac.in/',
                'https://mrc.iisc.ac.in/',
                'https://orgchem.iisc.ac.in/',
                'https://sscu.iisc.ac.in/',
                'https://www.csa.iisc.ac.in/',
                'https://ece.iisc.ac.in/',
                'https://dese.iisc.ac.in/',
                'https://ee.iisc.ac.in/',
                'https://cistup.iisc.ac.in/',
                'https://be.iisc.ac.in/',
                'https://csp.iisc.ac.in/',
                'https://www.cense.iisc.ac.in/',
                'https://cds.iisc.ac.in/',  
                'https://mgmt.iisc.ac.in/',
                'https://icer.iisc.ac.in/',
                'https://icwar.iisc.ac.in/',
                'https://cps.iisc.ac.in/',
                'https://msci.iisc.ac.in/',
                'https://serc.iisc.ac.in/',
                'https://iqti.iisc.ac.in/',
                'https://abcmc.iisc.ac.in/',
                'https://www.longevity.iisc.ac.in/',
                'https://aero.iisc.ac.in/',
                'https://caos.iisc.ac.in/',
                'https://ceas.iisc.ac.in/',
                'https://camm.iisc.ac.in/',
                'https://dm.iisc.ac.in/dm/',
                'https://cst.iisc.ac.in/cst/',
                'https://chemeng.iisc.ac.in/',
                'https://www.civil.iisc.ac.in/',
                'http://dccc.iisc.ac.in/',
                'https://materials.iisc.ac.in/',
                'https://mecheng.iisc.ac.in/',
                'https://physics.iisc.ac.in/~jap/',
                'https://cct.iisc.ac.in/',
                'https://chep.iisc.ac.in/',
                'https://math.iisc.ac.in/',
                'https://iap.iisc.ac.in/',
                'https://physics.iisc.ac.in/',
                'https://www.iisc.ac.in/centre-for-brain-research/',
                'https://www.fsid-iisc.in/',
                'https://diarcoe.iisc.ac.in/',
            ]
        
        # Allow all domains (no restriction)
        self.allowed_domains = []
        
        # Depth configuration - INDEPENDENT tracking per domain type
        self.max_depth_department = 5  # Depth 5 for IISc domains
        self.external_depth = 2  # Depth 2 for non-iisc domains (INDEPENDENT)
        
        logger.info(f"Initialized IISc Spider with {len(self.start_urls)} department URLs")
        logger.info(f"Depth policy: IISc={self.max_depth_department} (independent), External={self.external_depth} (independent)")
        logger.info("CDS crawling: ENABLED - Full CDS department will be crawled")
        logger.info("External domains: ALL ALLOWED - Including unsecure and any domain type")
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL for better deduplication.
        Handles: protocol, domain case, trailing slashes, parameter order
        """
        try:
            parsed = urlparse(url)
            
            # 1. Normalize scheme (prefer HTTPS)
            scheme = 'https' if parsed.scheme in ('http', 'https') else parsed.scheme
            
            # 2. Normalize domain (lowercase, remove trailing slash)
            netloc = parsed.netloc.lower().rstrip('/')
            
            # 3. Normalize path (remove trailing slash except for root)
            path = parsed.path.rstrip('/') if parsed.path != '/' else '/'
            
            # 4. Sort and filter query parameters for consistency
            query = ''
            if parsed.query:
                params = parse_qs(parsed.query, keep_blank_values=True)
                # Sort by key for consistency
                sorted_params = sorted(params.items())
                query = urlencode(sorted_params, doseq=True)
            
            # 5. Remove fragment (never used for content)
            fragment = ''
            
            normalized = urlunparse((scheme, netloc, path, '', query, fragment))
            
            if normalized != url:
                logger.debug(f"URL normalized: {url[:60]}... → {normalized[:60]}...")
            
            return normalized
        
        except Exception as e:
            logger.warning(f"Error normalizing URL {url}: {e}")
            return url
    
    def start_requests(self):
        """Generate initial requests"""
        for url in self.start_urls:
            is_iisc = 'iisc.ac.in' in url
            yield scrapy.Request(
                url,
                callback=self.parse,
                meta={
                    'depth': 0,
                    'max_depth': self.max_depth_department if is_iisc else self.external_depth,
                    'domain_type': 'iisc' if is_iisc else 'external',
                    'page_type': 'homepage'
                },
                errback=self.errback_httpbin,
            )
    
    def parse(self, response):
        """Parse response and extract data"""
        try:
            # Extract basic information
            url = response.url
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
            
            # Determine domain type
            is_iisc = 'iisc.ac.in' in domain
            domain_type = "IISc" if is_iisc else "External"
            
            logger.info(f"✓ Crawled [{domain_type}]: {url} (depth: {item['depth']})")
            print(f"[Spider] Crawled ({domain_type}) depth={item['depth']}: {url[:80]}...", flush=True)
            
            yield item
            
            # Extract and follow links with INDEPENDENT depth tracking
            current_depth = response.meta.get('depth', 0)
            current_domain_type = response.meta.get('domain_type', 'iisc' if is_iisc else 'external')
            
            # Determine if we should follow more links from this page
            # Each domain type tracks depth independently
            # Logic: current_depth < max means we can follow links to create next depth
            # Example: max=5 means we crawl depths 0,1,2,3,4,5 but only follow FROM 0,1,2,3,4
            if is_iisc:
                can_follow = current_depth < self.max_depth_department
            else:
                can_follow = current_depth < self.external_depth
            
            if can_follow:
                for link in self._extract_links(response):
                    # Determine link domain type
                    link_domain = urlparse(link).netloc
                    link_is_iisc = 'iisc.ac.in' in link_domain
                    
                    # INDEPENDENT depth tracking:
                    # - If staying within same domain type, increment depth
                    # - If switching domain type, RESET depth to 0
                    if link_is_iisc == is_iisc:
                        # Same domain type - increment depth
                        next_depth = current_depth + 1
                    else:
                        # Crossing boundary - reset depth for new domain type
                        next_depth = 0
                    
                    # Determine max depth for the link
                    link_max_depth = self.max_depth_department if link_is_iisc else self.external_depth
                    link_domain_type = 'iisc' if link_is_iisc else 'external'
                    
                    # Calculate priority for this request
                    priority = self._calculate_request_priority(link, item.get('page_type'))
                    
                    yield scrapy.Request(
                        link,
                        callback=self.parse,
                        priority=priority,
                        meta={
                            'depth': next_depth,
                            'max_depth': link_max_depth,
                            'domain_type': link_domain_type,
                            'source_url': url  # Track source for debugging
                        },
                        errback=self.errback_httpbin,
                    )
        
        except Exception as e:
            logger.error(f"Error parsing {response.url}: {e}")
    
    def _extract_title(self, response) -> str:
        """Extract page title"""
        try:
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Try different title sources
            title = None
            
            # 1. <title> tag
            if soup.title:
                title = soup.title.string
            
            # 2. <h1> tag
            if not title:
                h1 = soup.find('h1')
                if h1:
                    title = h1.get_text(strip=True)
            
            # 3. og:title meta tag
            if not title:
                og_title = soup.find('meta', property='og:title')
                if og_title:
                    title = og_title.get('content')
            
            # 4. Fallback to URL
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
            
            # Find all anchor tags
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Convert relative URLs to absolute
                absolute_url = urljoin(response.url, href)
                
                # Normalize URL to eliminate duplicates
                absolute_url = self._normalize_url(absolute_url)
                
                # Parse URL
                parsed = urlparse(absolute_url)
                
                # Filter links
                if self._should_follow_link(parsed, response.url):
                    yield absolute_url
        
        except Exception as e:
            logger.warning(f"Error extracting links from {response.url}: {e}")
    
    def _calculate_request_priority(self, url: str, source_page_type: str = None) -> int:
        """
        Calculate priority for a request.
        Higher priority = processed first in queue
        Range: -100 (lowest) to 1000 (highest)
        """
        priority = 500  # Default medium priority
        url_lower = url.lower()
        
        # Faculty pages - HIGHEST priority
        if any(term in url_lower for term in ['/faculty', '/people/', '/member', '/staff']):
            priority += 300
        
        # Research pages - HIGH priority
        elif any(term in url_lower for term in ['/research', '/lab', 'group', '/projects']):
            priority += 250
        
        # Publication pages - MEDIUM-HIGH priority
        elif any(term in url_lower for term in ['/publication', '/paper', 'thesis', '/book']):
            priority += 150
        
        # Course pages - MEDIUM priority
        elif any(term in url_lower for term in ['/course', '/teach', '/class']):
            priority += 50
        
        # Navigation/static pages - LOW priority
        elif any(term in url_lower for term in ['/about', '/contact', '/gallery']):
            priority -= 50
        
        # Archives/old content - LOWEST priority
        elif any(term in url_lower for term in ['/news', '/archive', '/old', '/past', '/2020', '/2019', '/2018']):
            priority -= 100
        
        # Referrer boost: if referred from faculty page, boost priority
        if source_page_type == 'faculty':
            priority += 50
        
        return max(-100, min(1000, priority))  # Clamp between -100 and 1000
    
    def _should_follow_link(self, parsed_url, source_url: str) -> bool:
        """Determine if a link should be followed"""
        # Allow HTTP, HTTPS, and even unsecure protocols
        if parsed_url.scheme not in ['http', 'https']:
            return False
        
        # REMOVED: CDS blocking - Now CDS will be crawled fully
        # Allow ALL domains including CDS and any external site
        
        # Allow all domains (no domain restriction)
        
        # Skip common non-content files
        skip_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', 
                          '.zip', '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3']
        
        if any(parsed_url.path.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip fragments (same page anchors)
        if parsed_url.fragment and not parsed_url.query:
            full_url_without_fragment = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            source_parsed = urlparse(source_url)
            source_url_without_fragment = f"{source_parsed.scheme}://{source_parsed.netloc}{source_parsed.path}"
            if full_url_without_fragment == source_url_without_fragment:
                return False
        
        return True
    
    def errback_httpbin(self, failure):
        """Enhanced error handling with smart retry strategy"""
        request = failure.request
        url = request.url
        retry_count = request.meta.get('retry_count', 0)
        max_retries = 3
        
        try:
            if failure.check(HttpError):
                # HTTP error responses
                response = failure.value.response
                status_code = response.status
                
                if status_code == 429:  # Too many requests (rate limited)
                    logger.warning(f"[429] Rate limited on {urlparse(url).netloc}")
                    if retry_count < 1:
                        # Retry with backoff - very low priority
                        return scrapy.Request(
                            url,
                            callback=self.parse,
                            priority=-100,
                            meta={**request.meta, 'retry_count': retry_count + 1},
                            errback=self.errback_httpbin,
                            dont_filter=True
                        )
                
                elif status_code in [500, 502, 503, 504]:  # Server errors
                    logger.warning(f"[{status_code}] Server error on {urlparse(url).netloc}")
                    if retry_count < 2:  # Allow 2 retries for server errors
                        return scrapy.Request(
                            url,
                            callback=self.parse,
                            priority=100 - (retry_count * 50),
                            meta={**request.meta, 'retry_count': retry_count + 1},
                            errback=self.errback_httpbin,
                            dont_filter=True
                        )
                    else:
                        logger.error(f"Giving up after {retry_count} retries: {url}")
                
                elif status_code == 403:  # Forbidden
                    logger.debug(f"[403] Access forbidden: {url}")
                
                elif status_code == 404:  # Not found
                    logger.debug(f"[404] Page not found: {url}")
                
                else:
                    logger.warning(f"[{status_code}] HTTP error: {url}")
            
            elif failure.check(TimeoutError):
                # Connection/read timeout
                domain = urlparse(url).netloc
                if 'iisc.ac.in' in domain and retry_count < 2:
                    logger.info(f"[Timeout] Retrying IISc domain {domain}")
                    return scrapy.Request(
                        url,
                        callback=self.parse,
                        priority=150 - (retry_count * 50),
                        meta={**request.meta, 'retry_count': retry_count + 1},
                        errback=self.errback_httpbin,
                        dont_filter=True
                    )
                else:
                    logger.warning(f"[Timeout] Giving up: {url}")
            
            elif failure.check(DNSLookupError):
                domain = urlparse(url).netloc
                logger.warning(f"[DNS] Resolution failed for {domain}")
            
            elif failure.check(ConnectionRefusedError):
                domain = urlparse(url).netloc
                logger.warning(f"[Connection] Refused by {domain}")
            
            else:
                logger.error(f"[{failure.type.__name__}] Unexpected error for {url}: {failure.value}")
        
        except Exception as e:
            logger.error(f"Error in errback handler for {url}: {e}")
