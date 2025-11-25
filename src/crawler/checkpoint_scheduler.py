"""
Simple checkpoint system using JSONL output for resume
No complex scheduler needed - just reads visited URLs from previous crawl output
"""

import json
from pathlib import Path
from scrapy.dupefilters import RFPDupeFilter
from loguru import logger

# Scrapy 2.x compatibility: request_fingerprint moved
try:
    from scrapy.utils.request import request_fingerprint
except ImportError:
    from scrapy.utils.request import fingerprint as request_fingerprint


class OutputBasedDupeFilter(RFPDupeFilter):
    """
    Duplicate filter that loads visited URLs from previous JSONL output
    Much simpler than complex checkpoint system
    """
    
    def __init__(self, path=None, debug=False, fingerprinter=None):
        super().__init__(path, debug, fingerprinter=fingerprinter)
        self.visited_urls = set()
    
    @classmethod
    def from_spider(cls, spider):
        # Get fingerprinter from settings for Scrapy 2.x compatibility
        from scrapy.utils.request import request_fingerprinter_from_settings
        fingerprinter = request_fingerprinter_from_settings(spider.settings)
        instance = cls(debug=spider.settings.getbool('DUPEFILTER_DEBUG'), fingerprinter=fingerprinter)
        
        # Load visited URLs from previous crawl output (if exists)
        output_dir = Path('data/crawled_pages')
        if output_dir.exists():
            # Find latest JSONL file for this spider
            pattern = f"pages_{spider.name}_*.jsonl"
            jsonl_files = sorted(output_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
            
            if jsonl_files:
                latest_file = jsonl_files[0]
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                page = json.loads(line.strip())
                                if 'url' in page:
                                    instance.visited_urls.add(page['url'])
                            except json.JSONDecodeError:
                                continue
                    
                    logger.info(f"✓ Loaded {len(instance.visited_urls)} visited URLs from {latest_file.name}")
                except Exception as e:
                    logger.warning(f"Could not load previous crawl data: {e}")
        
        if not instance.visited_urls:
            logger.info("Starting fresh crawl (no previous data found)")
        
        return instance
    
    def request_seen(self, request):
        """Check if URL was already crawled"""
        # Check if URL already in output file
        if request.url in self.visited_urls:
            return True
        
        # Use standard fingerprint check for current session
        fp = self.request_fingerprint(request)
        if fp in self.fingerprints:
            return True
        
        self.fingerprints.add(fp)
        return False
    
    def close(self, reason):
        """No need to save - JSONL output is our checkpoint"""
        logger.info(f"DupeFilter closed: {len(self.fingerprints)} URLs in current session (reason: {reason})")
