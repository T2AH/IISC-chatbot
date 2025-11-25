"""
Scrapy Item definitions for crawled pages
"""

import scrapy
from typing import List, Dict, Optional
from datetime import datetime


class PageItem(scrapy.Item):
    """Item representing a crawled web page"""
    
    # Basic metadata
    page_id = scrapy.Field()
    url = scrapy.Field()
    title = scrapy.Field()
    domain = scrapy.Field()
    page_type = scrapy.Field()
    crawl_date = scrapy.Field()
    
    # Content
    raw_html = scrapy.Field()
    cleaned_text = scrapy.Field()
    
    # Metadata
    depth = scrapy.Field()
    response_status = scrapy.Field()
    content_length = scrapy.Field()
    
    # For pipeline processing
    word_count = scrapy.Field()
    relevance_score = scrapy.Field()
    metadata = scrapy.Field()
    text_quality_score = scrapy.Field()
    chunking_hints = scrapy.Field()
