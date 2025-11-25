# Scrapy settings for IISc crawler project

BOT_NAME = 'iisc_crawler'

SPIDER_MODULES = ['src.crawler.spiders']
NEWSPIDER_MODULE = 'src.crawler.spiders'

# Crawl responsibly by identifying yourself
USER_AGENT = 'IISc-Research-Bot/1.0 (+https://iisc.ac.in)'

# Obey robots.txt rules (can be overridden for internal crawling)
ROBOTSTXT_OBEY = True

# ============================================================================
# PARALLELISM OPTIMIZATION for 13th Gen Intel i5-1335U (10 cores: 2P + 8E, 16GB RAM)
# ============================================================================
# Total parallel requests - tuned for 10-core CPU with multi-domain crawling
CONCURRENT_REQUESTS = 80  # 8x per core for I/O-bound crawling
CONCURRENT_REQUESTS_PER_DOMAIN = 8  # Higher per-domain (was 4) for better throughput
CONCURRENT_REQUESTS_PER_IP = 8  # Match per-domain

# Download settings - faster timeouts for efficiency
DOWNLOAD_DELAY = 0.3  # Reduced from 0.5s - more aggressive with 10 cores
DOWNLOAD_TIMEOUT = 15  # Faster timeout (was 20s)

# Reactor and DNS optimization
REACTOR_THREADPOOL_MAXSIZE = 40  # 4x cores for network I/O
DNSCACHE_ENABLED = True
DNSCACHE_SIZE = 10000  # Larger cache for multi-domain crawling

# ============================================================================
# CHECKPOINT & RESUME CONFIGURATION
# ============================================================================
CHECKPOINT_DIR = 'data/checkpoints'
# Simple resume system: uses JSONL output to avoid re-crawling visited URLs
DUPEFILTER_CLASS = 'src.crawler.checkpoint_scheduler.OutputBasedDupeFilter'

# ============================================================================
# STANDARD SCRAPY SETTINGS
# ============================================================================
# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Override the default request headers
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en',
}

# Enable or disable spider middlewares
SPIDER_MIDDLEWARES = {
    'src.crawler.middlewares.DepthAdjustmentMiddleware': 543,
    'src.crawler.middlewares.DomainFilterMiddleware': 544,
}

# Enable or disable downloader middlewares
DOWNLOADER_MIDDLEWARES = {
    'src.crawler.middlewares.RetryMiddleware': 550,
    'src.crawler.middlewares.RobotsOverrideMiddleware': 551,
}

# Enable or disable extensions
EXTENSIONS = {
    'scrapy.extensions.telnet.TelnetConsole': None,
}

# Configure item pipelines
ITEM_PIPELINES = {
    'src.crawler.pipelines.ValidationPipeline': 100,
    'src.crawler.pipelines.TextCleaningPipeline': 200,
    'src.crawler.pipelines.ContentQualityFilterPipeline': 250,  # Filter low-quality content
    'src.crawler.pipelines.PageTypeClassificationPipeline': 300,
    'src.crawler.pipelines.CrawlProgressPipeline': 350,  # Track progress per domain
    'src.crawler.pipelines.JsonExportPipeline': 400,
}

# Retry configuration - faster failure recovery
RETRY_ENABLED = True
RETRY_TIMES = 2  # Quick retries
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]

# Depth limit (base setting - overridden by spider per domain type)
DEPTH_LIMIT = 5  # IISc depth (external is 2, tracked independently)
DEPTH_PRIORITY = 1

# AutoThrottle extension - Tuned for high-concurrency multi-domain crawling
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.3  # Match DOWNLOAD_DELAY
AUTOTHROTTLE_MAX_DELAY = 3.0  # Cap backoff at 2s (was 3s)
AUTOTHROTTLE_TARGET_CONCURRENCY = 4.0  # 4 concurrent per domain (was 2.0)
AUTOTHROTTLE_DEBUG = False

# HTTP caching - disabled for fresh crawl with checkpoint resume
HTTPCACHE_ENABLED = False  # Checkpoint system handles resume
HTTPCACHE_EXPIRATION_SECS = 0
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

# Logging (LOG_FILE set dynamically in main.py per crawl)
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
LOG_DATEFORMAT = '%Y-%m-%d %H:%M:%S'
LOG_STDOUT = True  # Enable console output
LOG_ENABLED = True

# Reduce noise from robots.txt failures (404/403 are normal)
import logging
logging.getLogger('scrapy.downloadermiddlewares.robotstxt').setLevel(logging.ERROR)

# FEEDS disabled - using JsonExportPipeline for real-time writes instead
# FEEDS = {}
