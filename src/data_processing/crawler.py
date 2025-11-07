# c:\Users\harsh\Documents\chat application\src\data_processing\crawler.py
import requests
from bs4 import BeautifulSoup
import json
import os
from urllib.parse import urljoin, urlparse
import hashlib
from collections import deque

class WebCrawler:
    def __init__(self, start_urls, output_dir, max_depth=3, allowed_domains=None):
        self.start_urls = start_urls
        self.output_dir = output_dir
        self.max_depth = max_depth
        self.visited = set()
        self.session = requests.Session()
        if allowed_domains is None:
            # Default behavior: only allow domains from start_urls
            self.allowed_domains = [urlparse(url).netloc for url in start_urls]
        else:
            self.allowed_domains = allowed_domains

    def crawl(self):
        """Starts the crawling process using Breadth-First Search (BFS)."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        queue = deque([(url, 0) for url in self.start_urls])
        
        while queue:
            url, depth = queue.popleft()
            
            if url in self.visited:
                continue

            if depth > self.max_depth:
                continue

            print(f"Crawling: {url} at depth {depth}")
            self.visited.add(url)

            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                title = soup.title.string if soup.title else 'No Title'

                self._save_content(url, title, soup.get_text(), depth)

                if depth < self.max_depth:
                    for link in self._get_links(url, soup):
                        if link not in self.visited:
                            queue.append((link, depth + 1))

            except requests.RequestException as e:
                print(f"Error crawling {url}: {e}")

    def _get_links(self, url, soup):
        """Extracts all valid links from a page."""
        links = set()
        for a_tag in soup.find_all('a', href=True):
            link = urljoin(url, a_tag['href'])
            if self._is_valid_url(link):
                links.add(link)
        return links

    def _is_valid_url(self, url):
        """
        Checks if a URL is valid, within an allowed domain, and not a file.
        """
        try:
            parsed_url = urlparse(url)
            
            # 1. Check for valid scheme
            if parsed_url.scheme not in ['http', 'https']:
                return False
            
            # 2. Check if it's a file URL we want to ignore
            if self._is_file_url(parsed_url.path):
                return False

            # 3. Check if the domain is one of the allowed domains or a subdomain
            return any(parsed_url.netloc == domain or parsed_url.netloc.endswith('.' + domain) for domain in self.allowed_domains)
        except ValueError:
            # Handle potential malformed URLs
            return False

    def _is_file_url(self, path):
        """Checks if the URL path points to a file to be ignored."""
        ignored_extensions = [
            # Images
            '.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp', '.ico',
            # Documents
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            # Archives
            '.zip', '.tar', '.gz', '.rar',
            # Other
            '.css', '.js', '.xml', '.rss'
        ]
        return any(path.lower().endswith(ext) for ext in ignored_extensions)

    def _save_content(self, url, title, content, depth):
        """Saves the content of a page to a file using a hash of the URL as the filename."""
        # Use a SHA-256 hash of the URL for a unique, fixed-length filename
        filename = f"{hashlib.sha256(url.encode()).hexdigest()}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'url': url, 'title': title, 'content': content, 'depth': depth}, f, indent=4)


def main():
    """Main function to run the crawler with configuration."""
    import sys
    import os
    # Add the project root to the Python path to allow for absolute imports
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
