#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from collections import deque
from typing import Deque, Dict, Optional, Set, Tuple
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; SimpleTextCrawler/1.1; +https://example.com/bot)"
DEFAULT_TIMEOUT = 15  # seconds

# Patterns to identify faculty/research/student pages
FACULTY_PATTERNS = [
    r"/faculty", r"/people", r"/staff", r"/members", r"/research", r"/students"
]

def normalize_url(url: str) -> str:
    """Normalize URL for deduping: remove fragments, lowercase scheme/host, normalize path."""
    u, _frag = urldefrag(url)
    p = urlparse(u)

    scheme = (p.scheme or "http").lower()
    hostname = (p.hostname or "").lower()

    # Keep port only if non-default
    netloc = hostname
    if p.port:
        if not ((scheme == "http" and p.port == 80) or (scheme == "https" and p.port == 443)):
            netloc = f"{hostname}:{p.port}"

    # Normalize path: collapse slashes, remove trailing slash (except root)
    path = re.sub(r"/+", "/", p.path or "/")
    if path != "/":
        path = path.rstrip("/")

    return urlunparse((scheme, netloc, path, "", p.query or "", ""))


def same_reg_domain(a: str, b: str, include_subdomains: bool) -> bool:
    """Check if URL 'a' is within the same domain as URL 'b' (optionally including subdomains)."""
    ha = (urlparse(a).hostname or "").lower()
    hb = (urlparse(b).hostname or "").lower()
    if not include_subdomains:
        return ha == hb
    return ha == hb or ha.endswith("." + hb)


def is_http_url(url: str) -> bool:
    scheme = urlparse(url).scheme.lower()
    return scheme in ("http", "https")


def build_robot_parser(base_url: str) -> robotparser.RobotFileParser:
    p = urlparse(base_url)
    robots_url = urlunparse((p.scheme, p.netloc, "/robots.txt", "", "", ""))
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        # If robots can't be fetched, default to allowing (common convention)
        rp = robotparser.RobotFileParser()
        rp.parse([])
    return rp


def extract_visible_text(html: str) -> Tuple[str, str]:
    # Prefer lxml parser if installed, fallback to built-in
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Remove non-visible or boilerplate-ish nodes
    for tag in soup(["script", "style", "noscript", "svg", "img", "meta", "link", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Get text with line separation
    raw = soup.get_text(separator="\n")

    # Normalize whitespace; drop empty lines
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)

    return "\n".join(lines), (soup.title.get_text(strip=True) if soup.title else "")

def is_faculty_link(url: str) -> bool:
    return any(re.search(pat, url, re.IGNORECASE) for pat in FACULTY_PATTERNS)

def extract_faculty_info(text: str) -> Dict[str, str]:
    # Simple heuristics: you can improve these with more samples
    info = {}
    # Extract faculty name (look for lines starting with Dr./Prof./Mr./Ms.)
    match = re.search(r"(Dr\.|Prof\.|Mr\.|Ms\.)\s+[A-Z][a-zA-Z\-\s]+", text)
    if match:
        info["faculty_name"] = match.group(0)
    # Extract research area (look for 'Research Area' or 'Research Interests')
    match = re.search(r"Research (Area|Interests|Fields):?\s*(.+)", text)
    if match:
        info["research_area"] = match.group(2).strip()
    # Extract students (look for 'Students:' or 'PhD Students:')
    match = re.search(r"(PhD Students|Students):?\s*(.+)", text)
    if match:
        info["students"] = match.group(2).strip()
    return info

def crawl(
    start_url: str,
    max_depth: int,
    same_domain: bool,
    include_subdomains: bool,
    max_pages: int,
    delay: float,
    output_path: Optional[str],
    timeout: int,
    verbose: bool,
    ignore_robots: bool,
) -> None:
    def log(msg: str):
        if verbose:
            print(msg, file=sys.stderr, flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"})

    start_url_norm = normalize_url(start_url)
    queue: Deque[Tuple[str, int]] = deque([(start_url_norm, 0)])
    visited: Set[str] = set()  # final URLs processed (after redirects)
    seen: Set[str] = {start_url_norm}  # URLs enqueued
    pages_crawled = 0

    # robots.txt cache by netloc
    robots_cache: Dict[str, robotparser.RobotFileParser] = {}

    out_fh = open(output_path, "a", encoding="utf-8") if output_path else None
    try:
        while queue and pages_crawled < max_pages:
            url, depth = queue.popleft()
            log(f"[QUEUE] pop depth={depth} url={url}")

            if not is_http_url(url):
                log(f"[SKIP] non-http url={url}")
                continue

            if same_domain and not same_reg_domain(url, start_url_norm, include_subdomains):
                log(f"[SKIP] off-domain url={url}")
                continue

            # robots.txt
            if not ignore_robots:
                netloc = urlparse(url).netloc
                rp = robots_cache.get(netloc)
                if rp is None:
                    base_for_robots = urlunparse((urlparse(url).scheme, netloc, "/", "", "", ""))
                    rp = build_robot_parser(base_for_robots)
                    robots_cache[netloc] = rp
                    log(f"[ROBOTS] loaded for netloc={netloc}")
                try:
                    if not rp.can_fetch(USER_AGENT, url):
                        log(f"[SKIP] robots disallow url={url}")
                        continue
                except Exception as e:
                    log(f"[WARN] robots check failed for url={url}: {e}")

            try:
                resp = session.get(url, timeout=timeout, allow_redirects=True)
            except requests.RequestException as e:
                log(f"[ERROR] fetch failed url={url}: {e}")
                continue

            final_url = normalize_url(resp.url)
            ctype = resp.headers.get("Content-Type", "")
            log(f"[FETCH] {resp.status_code} url={url} -> final={final_url} ctype='{ctype}'")

            # Dedup AFTER resolving redirects
            if final_url in visited:
                log(f"[SKIP] already visited final_url={final_url}")
                if delay > 0:
                    time.sleep(delay)
                continue
            visited.add(final_url)

            if "text/html" not in ctype:
                log(f"[SKIP] non-HTML ctype for final_url={final_url}")
                if delay > 0:
                    time.sleep(delay)
                continue

            try:
                text, title = extract_visible_text(resp.text)
            except Exception as e:
                log(f"[ERROR] parse failed final_url={final_url}: {e}")
                if delay > 0:
                    time.sleep(delay)
                continue

            # Extract faculty info if on a faculty page
            faculty_info = {}
            if is_faculty_link(final_url):
                faculty_info = extract_faculty_info(text)

            record = {
                "url": final_url,
                "depth": depth,
                "title": title,
                "text": text,
                **faculty_info
            }

            if out_fh:
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                print(json.dumps(record, ensure_ascii=False))
            log(f"[WRITE] final_url={final_url} depth={depth} title='{title[:80]}' chars={len(text)}")
            pages_crawled += 1

            # Stop expanding links beyond max_depth
            if depth >= max_depth:
                log(f"[STOP] max depth reached for final_url={final_url}")
                if delay > 0:
                    time.sleep(delay)
                continue

            # Extract and enqueue links
            new_count = 0
            try:
                try:
                    soup = BeautifulSoup(resp.text, "lxml")
                except Exception:
                    soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a.get("href", "").strip()
                    # Skip empty, anchor, mailto, javascript links
                    if not href or href.startswith("#") or href.startswith(("mailto:", "javascript:", "tel:")):
                        continue
                    # Skip language switcher and navigation links
                    if "?lang=" in href or "about" in href or "contact" in href or "corner" in href:
                        continue
                    joined = urljoin(final_url, href)
                    norm = normalize_url(joined)
                    if same_domain and not same_reg_domain(norm, start_url_norm, include_subdomains):
                        continue
                    # Only enqueue faculty-related links
                    if not is_faculty_link(norm):
                        continue
                    if norm not in seen:
                        queue.append((norm, depth + 1))
                        seen.add(norm)
                        new_count += 1
            except Exception as e:
                log(f"[WARN] link extraction failed final_url={final_url}: {e}")

            log(f"[ENQUEUE] added {new_count} links from final_url={final_url} (queue={len(queue)})")

            if delay > 0:
                time.sleep(delay)

    finally:
        if out_fh:
            out_fh.close()
        log(f"[DONE] pages_crawled={pages_crawled}, visited={len(visited)}, queue_left={len(queue)}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Crawl a website up to a given depth and extract only visible text and faculty info.")
    p.add_argument("--url", required=True, help="Starting URL (e.g., https://example.com)")
    p.add_argument("--depth", type=int, default=2, help="Maximum crawl depth (BFS). Default: 2")
    p.add_argument("--max-pages", type=int, default=500, help="Maximum number of pages to crawl. Default: 500")
    p.add_argument("--same-domain", action="store_true", help="Restrict crawling to the same domain as the start URL.")
    p.add_argument("--include-subdomains", action="store_true", help="When restricting domain, include subdomains.")
    p.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds. Default: 0.5")
    p.add_argument("--output", help="Path to JSONL (NDJSON) output file. Defaults to stdout if not set.")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Request timeout in seconds. Default: {DEFAULT_TIMEOUT}")
    p.add_argument("--verbose", action="store_true", help="Print verbose logs to stderr.")
    p.add_argument("--ignore-robots", action="store_true", help="Ignore robots.txt (for debugging; use only with permission).")
    return p.parse_args(argv)

def get_department_urls(departments_page_url: str) -> list:
    """Extract department URLs from the main content container of the IISc departments page."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    resp = session.get(departments_page_url, timeout=DEFAULT_TIMEOUT)
    soup = BeautifulSoup(resp.text, "lxml")
    dept_urls = set()
    base_url = departments_page_url

    # Find the main content container
    main_container = soup.find("div", class_="container commonoutercontainer")
    if not main_container:
        main_container = soup  # fallback to whole page

    # Find all department links inside this container
    for a in main_container.find_all("a", href=True):
        href = a["href"].strip()
        # Skip empty, anchor, mailto, javascript links
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        # Resolve relative URLs
        full_url = urljoin(base_url, href)
        # Only pick links under iisc.ac.in
        if "iisc.ac.in" not in full_url:
            continue
        # Filter out main page and obvious non-department links
        if full_url == departments_page_url:
            continue
        dept_urls.add(full_url)
    return sorted(dept_urls)

def main(argv=None):
    args = parse_args(argv)
    # If starting from departments page, extract department URLs and crawl each
    if args.url == "https://iisc.ac.in/academics/departments/":
        dept_urls = get_department_urls(args.url)
        print(f"Found {len(dept_urls)} department URLs.")
        for dept_url in dept_urls:
            print(f"Crawling department: {dept_url}")
            crawl(
                start_url=dept_url,
                max_depth=args.depth,
                same_domain=args.same_domain,
                include_subdomains=args.include_subdomains,
                max_pages=args.max_pages,
                delay=args.delay,
                output_path=args.output,
                timeout=args.timeout,
                verbose=args.verbose,
                ignore_robots=args.ignore_robots,
            )
    else:
        crawl(
            start_url=args.url,
            max_depth=args.depth,
            same_domain=args.same_domain,
            include_subdomains=args.include_subdomains,
            max_pages=args.max_pages,
            delay=args.delay,
            output_path=args.output,
            timeout=args.timeout,
            verbose=args.verbose,
            ignore_robots=args.ignore_robots,
        )

if __name__ == "__main__":
    main()