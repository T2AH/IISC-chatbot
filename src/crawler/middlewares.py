"""
Scrapy middlewares for IISc crawler
"""

from scrapy import signals
from scrapy.http import Request
from scrapy.exceptions import IgnoreRequest
from urllib.parse import urlparse
import re
from typing import Optional
from loguru import logger


class DepthAdjustmentMiddleware:
    """Middleware for dynamic depth adjustment based on content quality"""
    
    def __init__(self, config):
        self.config = config
        self.high_content_threshold = config.get('crawler', 'dynamic_depth', 'high_content_threshold', default=500)
        self.low_content_threshold = config.get('crawler', 'dynamic_depth', 'low_content_threshold', default=100)
        # Default disabled to ensure configured depth limits are honored unless explicitly enabled
        self.enabled = config.get('crawler', 'dynamic_depth', 'enabled', default=False)
    
    @classmethod
    def from_crawler(cls, crawler):
        from src.config import config
        middleware = cls(config)
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        return middleware
    
    def spider_opened(self, spider):
        logger.info(f'Dynamic depth adjustment middleware enabled: {self.enabled}')
    
    def process_spider_output(self, response, result, spider):
        """Adjust depth for outgoing requests based on current page content.

        Note:
        - Computes word_count directly from the current response body to avoid relying on pipeline-populated fields.
        - Never reduces allowed depth below current_depth + 1 to prevent premature crawl termination.
        """
        if not self.enabled:
            yield from result
            return
        
        # Analyze current page by computing a lightweight word count from HTML
        try:
            from bs4 import BeautifulSoup
            text = BeautifulSoup(response.text, 'lxml').get_text(" ", strip=True)
            word_count = len(text.split())
        except Exception:
            word_count = 0
        
        for item_or_request in result:
            if isinstance(item_or_request, Request):
                current_depth = response.meta.get('depth', 0)
                max_depth = response.meta.get('max_depth', 5)  # Default to 5 (IISc max depth)
                
                # Adjust depth based on content quality
                if word_count > self.high_content_threshold:
                    # High-quality content - allow deeper crawling
                    adjusted_max_depth = max_depth + 1
                    item_or_request.meta['max_depth'] = adjusted_max_depth
                    logger.debug(f"Increased depth limit to {adjusted_max_depth} for high-quality page: {response.url}")
                
                elif word_count < self.low_content_threshold:
                    # Low-quality content - reduce depth
                    adjusted_max_depth = max(current_depth + 1, max_depth - 1)
                    item_or_request.meta['max_depth'] = adjusted_max_depth
                    logger.debug(f"Reduced depth limit to {adjusted_max_depth} for low-quality page: {response.url}")
            
            yield item_or_request


class DomainFilterMiddleware:
    """Middleware for filtering domains based on whitelist/blacklist with enhanced noise reduction"""
    
    def __init__(self, config):
        self.config = config
        self.whitelist_patterns = config.get('crawler', 'domains', 'whitelist_patterns', default=[])
        self.blacklist_patterns = config.get('crawler', 'domains', 'blacklist_patterns', default=[])
        
        # Compile regex patterns
        self.blacklist_regex = [re.compile(pattern.replace('*', '.*')) 
                               for pattern in self.blacklist_patterns]
        
        # Enhanced noise reduction patterns
        self.noise_keywords = [
            'schedule', 'timetable', 'time-table', 'exam-schedule', 'calendar',
            'event', 'news', 'announcement', 'gallery', 'photo', 'image',
            'login', 'signin','download-', 'print', 'share', 'comment', 'reply'
        ]
        
        self.file_extensions = [
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
            '.zip', '.tar', '.gz', '.rar', '.7z',
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.webp',
            '.mp4', '.mp3', '.avi', '.mov', '.wav',
            '.exe', '.dmg', '.apk', '.deb', '.rpm'
        ]
    
    @classmethod
    def from_crawler(cls, crawler):
        from src.config import config
        middleware = cls(config)
        return middleware
    
    def process_spider_output(self, response, result, spider):
        """Filter requests based on URL patterns with comprehensive noise reduction"""
        for item_or_request in result:
            if isinstance(item_or_request, Request):
                url = item_or_request.url
                lower_url = url.lower()
                parsed = urlparse(url)
                path = parsed.path.lower()
                query_params = parsed.query.lower()
                
                # 1. File extension filter
                if any(path.endswith(ext) for ext in self.file_extensions):
                    logger.debug(f"Filtered file extension: {url}")
                    continue
                
                # 2. Schedule/calendar/event noise reduction
                # Allow shallow pages (depth ≤1) but block deeper calendar/schedule trees
                if any(kw in lower_url for kw in self.noise_keywords):
                    try:
                        depth_segments = [seg for seg in path.split('/') if seg]
                        # Block if > 1 path segment deep AND contains noise keywords
                        if len(depth_segments) > 1:
                            logger.debug(f"Filtered noise URL ({len(depth_segments)} segments): {url}")
                            continue
                    except Exception:
                        logger.debug(f"Filtered noise URL (parse error): {url}")
                        continue
                
                # 3. Date/time parameter filters (calendar pagination)
                if query_params and any(param in query_params for param in ['date=', 'month=', 'year=', 'day=', 'time=']):
                    if any(kw in lower_url for kw in ['calendar', 'schedule', 'event', 'news']):
                        logger.debug(f"Filtered date-parameter URL: {url}")
                        continue
                
                # 4. Pagination/sorting noise (non-content variants)
                if any(param in query_params for param in ['page=', 'offset=', 'sort=', 'order=', 'filter=']):
                    # Allow page=1 or page=2, block deeper pagination
                    if 'page=' in query_params:
                        try:
                            page_num = int(query_params.split('page=')[1].split('&')[0])
                            if page_num > 3:  # Block beyond page 3
                                logger.debug(f"Filtered deep pagination: {url}")
                                continue
                        except (ValueError, IndexError):
                            pass
                
                # 5. Session/tracking parameters (low value)
                if any(param in query_params for param in ['session=', 'sid=', 'token=', 'utm_', 'ref=']):
                    logger.debug(f"Filtered tracking parameter URL: {url}")
                    continue
                
                # 6. Check blacklist (wildcard -> regex)
                if any(regex.search(url) for regex in self.blacklist_regex):
                    logger.debug(f"Filtered blacklisted URL: {url}")
                    continue
            
            yield item_or_request


class RetryMiddleware:

    def __init__(self, settings):
        self.max_retry_times = settings.getint('RETRY_TIMES')
        self.retry_http_codes = set(int(x) for x in settings.getlist('RETRY_HTTP_CODES'))
        self.priority_adjust = settings.getint('RETRY_PRIORITY_ADJUST', -1)
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)
    
    def process_response(self, request, response, spider):
        if request.meta.get('dont_retry', False):
            return response
        
        if response.status in self.retry_http_codes:
            return self._retry(request, response.status, spider) or response
        
        return response
    
    def process_exception(self, request, exception, spider):
        if request.meta.get('dont_retry', False):
            return None
        
        return self._retry(request, str(exception), spider)
    
    def _retry(self, request, reason, spider):
        retry_times = request.meta.get('retry_times', 0) + 1
        
        if retry_times <= self.max_retry_times:
            logger.debug(f"Retrying {request.url} (attempt {retry_times}/{self.max_retry_times}), reason: {reason}")
            
            retry_req = request.copy()
            retry_req.meta['retry_times'] = retry_times
            retry_req.priority = request.priority + self.priority_adjust
            
            # Exponential backoff
            retry_req.meta['download_delay'] = 2 ** retry_times
            
            return retry_req
        else:
            logger.warning(f"Gave up retrying {request.url} (failed {retry_times} times)")
            return None


class RobotsOverrideMiddleware:
    """Middleware to override robots.txt for internal IISc domains"""
    
    def __init__(self, config):
        self.config = config
        self.internal_override = config.get('crawler', 'settings', 'internal_override', default=True)
        self.primary_domains = config.get('crawler', 'domains', 'primary', default=[])
    
    @classmethod
    def from_crawler(cls, crawler):
        from src.config import config
        return cls(config)
    
    def process_request(self, request, spider):
        """Override robots.txt for internal domains"""
        if not self.internal_override:
            return None
        
        parsed_url = urlparse(request.url)
        domain = parsed_url.netloc
        
        # Check if domain is internal
        if any(internal_domain in domain for internal_domain in self.primary_domains):
            request.meta['dont_obey_robotstxt'] = True
            logger.debug(f"Overriding robots.txt for internal domain: {domain}")
        
        return None
