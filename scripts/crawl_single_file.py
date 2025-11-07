"""Run a BFS crawl and write all pages to a single JSON file.
Prints each URL as it's crawled.
"""
import os
import sys
import json
import argparse
from urllib.parse import urlparse, urljoin
from collections import deque

import requests
from bs4 import BeautifulSoup


def is_file_url(path: str) -> bool:
    ignored = (
        # Images
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp', '.ico',
        # Documents
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        # Archives
        '.zip', '.tar', '.gz', '.rar', '.7z',
        # Audio/Video
        '.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma',
        '.mp4', '.avi', '.mov', '.mkv', '.webm', '.mpeg', '.mpg', '.m4v',
        # Feeds and calendar
        '.xml', '.rss', '.ics',
        # Static assets
        '.css', '.js', '.woff', '.woff2', '.ttf', '.eot',
        # Other non-html text/binary we want to skip
        '.json', '.txt'
    )
    pl = (path or '').lower()
    return any(pl.endswith(ext) for ext in ignored)


def is_allowed(url: str, allowed_domains) -> bool:
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            return False
        if is_file_url(p.path):
            return False
        host = p.netloc.lower()
        for d in allowed_domains:
            d = d.lower()
            if host == d or host.endswith('.' + d):
                return True
        return False
    except Exception:
        return False


def _build_session():
    """Create a requests session with retries and a friendly User-Agent."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "CDS-Chatbot-Crawler/1.0 (+https://cds.iisc.ac.in)"
    })
    return session


def crawl_bfs(start_urls, allowed_domains, max_depth, progress_file: str | None = None, checkpoint_interval: int = 50):
    session = _build_session()
    visited = set()
    queue = deque([(u, 0) for u in start_urls])
    pages = []
    failed_hosts = {}

    try:
        while queue:
            url, depth = queue.popleft()
            if url in visited or depth > max_depth:
                continue

            print(f"Crawling: {url} at depth {depth}")
            visited.add(url)
            # Determine host early for failure tracking
            host = urlparse(url).netloc
            # Skip hosts that have failed too many times (only network/5xx)
            if failed_hosts.get(host, 0) >= 3:
                print(f"Skipping {url} due to repeated failures for host {host}")
                continue

            try:
                r = session.get(url, timeout=10)
                status = r.status_code
                # Handle HTTP errors explicitly
                if 400 <= status < 500:
                    # Do not count towards host failures for client errors
                    print(f"Client error {status} for {url}")
                    continue
                if 500 <= status < 600:
                    failed_hosts[host] = failed_hosts.get(host, 0) + 1
                    print(f"Server error {status} for {url}")
                    continue

                # Content-Type guard: only parse HTML
                ctype = (r.headers.get('Content-Type') or '').lower()
                if ('text/html' not in ctype) and ('application/xhtml+xml' not in ctype):
                    # Skip non-HTML content
                    print(f"Skipping non-HTML content at {url} ({ctype or 'unknown content-type'})")
                    continue

                soup = BeautifulSoup(r.content, 'html.parser')
                title = soup.title.string.strip() if soup.title and soup.title.string else 'No Title'
                content = soup.get_text(separator='\n', strip=True)

                # Collect outgoing links with anchor text for link graph
                out_links = []
                for a in soup.find_all('a', href=True):
                    href = a['href'].strip()
                    if href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
                        continue
                    nxt = urljoin(url, href)
                    text = (a.get_text(strip=True) or '').strip()
                    out_links.append({
                        'url': nxt,
                        'text': text,
                    })

                pages.append({
                    'url': url,
                    'title': title,
                    'content': content,
                    'depth': depth,
                    'links': out_links,
                })

                # Optional checkpoint save
                if progress_file and (len(pages) % checkpoint_interval == 0):
                    out_dir = os.path.dirname(progress_file)
                    if out_dir and not os.path.exists(out_dir):
                        os.makedirs(out_dir, exist_ok=True)
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(pages, f, indent=2, ensure_ascii=False)
                        f.flush()

                if depth < max_depth:
                    for a in soup.find_all('a', href=True):
                        href = a['href'].strip()
                        if href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
                            continue
                        nxt = urljoin(url, href)
                        if nxt not in visited and is_allowed(nxt, allowed_domains):
                            queue.append((nxt, depth + 1))
            except requests.ConnectionError as e:
                failed_hosts[host] = failed_hosts.get(host, 0) + 1
                print(f"Connection error for {url}: {e}")
            except requests.Timeout as e:
                failed_hosts[host] = failed_hosts.get(host, 0) + 1
                print(f"Timeout for {url}: {e}")
            except requests.RequestException as e:
                # Other request exceptions (DNS, SSL, etc.)
                failed_hosts[host] = failed_hosts.get(host, 0) + 1
                print(f"Request error for {url}: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted! Returning pages collected so far...")
    return pages


def main():
    # Ensure project root on path for config import
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from config import CRAWLER_CONFIG

    parser = argparse.ArgumentParser(description="BFS crawler to a single JSON file")
    parser.add_argument("--start", nargs="*", help="Start URL(s) to crawl; defaults to config.start_urls")
    parser.add_argument("--max-depth", type=int, default=None, help="Max crawl depth; defaults to config.max_depth")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path; defaults to config.output_file")
    parser.add_argument("--allow", nargs="*", default=None, help="Allowed domains; defaults to config.allowed_domains")
    args = parser.parse_args()

    start_urls = args.start or CRAWLER_CONFIG['start_urls']
    allowed_domains = args.allow or CRAWLER_CONFIG.get('allowed_domains', [])
    max_depth = args.max_depth if args.max_depth is not None else CRAWLER_CONFIG['max_depth']
    output_file = args.out or CRAWLER_CONFIG['output_file']

    pages = crawl_bfs(start_urls, allowed_domains, max_depth, progress_file=output_file)

    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(pages)} pages to {output_file}")


if __name__ == '__main__':
    main()
