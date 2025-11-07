#!/usr/bin/env python3
"""
Portable whole-site crawler for iisc.ac.in

- Respects robots.txt by default
- BFS with depth limit
- Extracts visible text + title
- Skips non-HTML and common binary assets
- Restrict to iisc.ac.in (optionally include subdomains); github.io is always treated as external
- Optionally crawl external links (github.io only) up to a small depth (default: 2)
- Writes JSONL with fields: url, depth, title, text, links
- Graceful Ctrl+C/TERM handling: finishes current step, closes output cleanly
 - Durability: use --format jsonl (recommended) and --fsync to persist each record immediately

Usage examples:
      python scripts/crawl_iisc_full.py --url https://iisc.ac.in --depth 8 \
          --same-domain --include-subdomains --external-depth 2 \
      --delay 0.5 --output /data/iisc_crawl.jsonl --verbose

  # Example focused crawl from departments hub
  python scripts/crawl_iisc_full.py --url https://iisc.ac.in/academics/departments/ \
      --depth 8 --same-domain --include-subdomains --external-depth 2 \
      --delay 0.5 --output /data/iisc_departments.jsonl --verbose
"""
import argparse
import json
import re
import sys
import time
from collections import deque
from typing import Deque, Dict, Optional, Set, Tuple, List
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse
from urllib import robotparser
import signal
import os

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone

USER_AGENT = "Mozilla/5.0 (compatible; IIScCrawler/1.0; +https://iisc.ac.in)"
DEFAULT_TIMEOUT = 15
ASSET_EXTS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".gz", ".tar", ".rar", ".7z",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp3", ".wav", ".mp4", ".mov", ".avi", ".mkv",
    ".json", ".xml", ".csv",
}


def normalize_url(url: str) -> str:
    u, _frag = urldefrag(url)
    p = urlparse(u)
    scheme = (p.scheme or "http").lower()
    hostname = (p.hostname or "").lower()
    netloc = hostname
    if p.port and not ((scheme == "http" and p.port == 80) or (scheme == "https" and p.port == 443)):
        netloc = f"{hostname}:{p.port}"
    path = re.sub(r"/+", "/", p.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", p.query or "", ""))


def same_reg_domain(a: str, b: str, include_subdomains: bool) -> bool:
    ha = (urlparse(a).hostname or "").lower()
    hb = (urlparse(b).hostname or "").lower()
    return ha == hb or (include_subdomains and ha.endswith("." + hb))


def is_http_url(url: str) -> bool:
    return urlparse(url).scheme.lower() in ("http", "https")


def is_asset_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in ASSET_EXTS)


def is_github_io_host(hostname: str) -> bool:
    host = (hostname or "").lower()
    return host == "github.io" or host.endswith(".github.io")


def build_robot_parser(base_url: str) -> robotparser.RobotFileParser:
    p = urlparse(base_url)
    robots_url = urlunparse((p.scheme, p.netloc, "/robots.txt", "", "", ""))
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        rp = robotparser.RobotFileParser()
        rp.parse([])
    return rp


def extract_visible_text(html: str) -> Tuple[str, str]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "img", "meta", "link", "nav", "header", "footer", "aside"]):
        tag.decompose()
    raw = soup.get_text(separator="\n")
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    title = soup.title.get_text(strip=True) if soup.title else ""
    return "\n".join(lines), title


# Removed departments-first helper; a focused crawl can start at the hub URL directly.


def crawl(
    start_url: str,
    max_depth: int,
    same_domain: bool,
    include_subdomains: bool,
    delay: float,
    output_path: Optional[str],
    output_format: str,
    timeout: int,
    verbose: bool,
    ignore_robots: bool,
    external_depth: int,
    fsync_writes: bool,
    tee: bool,
) -> None:
    def log(msg: str):
        if verbose:
            print(msg, file=sys.stderr, flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"})

    start_url_norm = normalize_url(start_url)

    # Handle graceful shutdown on SIGINT/SIGTERM
    stop_requested = False

    def _handle_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        log(f"[SIGNAL] Received {signum}, stopping gracefully after current cycle…")

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        # Some platforms/threads may not allow signal setup; ignore
        pass
    # queue items: (url, depth_from_start, parent_url, external_hops)
    # external_hops counts consecutive steps taken on domains outside the start domain
    queue: Deque[Tuple[str, int, Optional[str], int]] = deque([(start_url_norm, 0, None, 0)])
    visited: Set[str] = set()
    seen: Set[str] = {start_url_norm}
    pages_crawled = 0
    robots_cache: Dict[str, robotparser.RobotFileParser] = {}

    out_fh = open(output_path, "w", encoding="utf-8") if output_path else None
    json_array_mode = bool(out_fh and (output_format == "json" or (output_path and output_path.lower().endswith(".json"))))
    first_record_written = False
    if out_fh and json_array_mode:
        out_fh.write("[")
    try:
        while queue and not stop_requested:
            url, depth, parent, ext_hops = queue.popleft()
            log(f"[QUEUE] depth={depth} url={url}")

            if not is_http_url(url):
                continue
            # Determine if current URL is in the same domain as start
            is_same = same_reg_domain(url, start_url_norm, include_subdomains)
            if not is_same:
                # External crawling only allowed for github.io and only when external_depth > 0
                if external_depth < 1:
                    continue
                host = (urlparse(url).hostname or "").lower()
                if not is_github_io_host(host):
                    continue
                # Enforce external hop depth
                if ext_hops > external_depth:
                    continue

            # robots.txt
            if not ignore_robots:
                netloc = urlparse(url).netloc
                rp = robots_cache.get(netloc)
                if rp is None:
                    base_for_robots = urlunparse((urlparse(url).scheme, netloc, "/", "", "", ""))
                    rp = build_robot_parser(base_for_robots)
                    robots_cache[netloc] = rp
                    log(f"[ROBOTS] loaded {netloc}")
                try:
                    if not rp.can_fetch(USER_AGENT, url):
                        log(f"[SKIP] robots disallow {url}")
                        continue
                except Exception:
                    pass

            try:
                resp = session.get(url, timeout=timeout, allow_redirects=True)
            except requests.RequestException as e:
                log(f"[ERROR] fetch failed {url}: {e}")
                continue

            final_url = normalize_url(resp.url)
            ctype = resp.headers.get("Content-Type", "")
            status = resp.status_code
            fetched_at = datetime.now(timezone.utc).isoformat()
            if final_url in visited:
                if delay > 0:
                    time.sleep(delay)
                continue
            visited.add(final_url)

            if "text/html" not in ctype:
                if delay > 0:
                    time.sleep(delay)
                continue

            try:
                text, title = extract_visible_text(resp.text)
            except Exception as e:
                log(f"[ERROR] parse failed {final_url}: {e}")
                if delay > 0:
                    time.sleep(delay)
                continue

            # Page-level metadata
            lang = None
            canonical = None
            try:
                soup_head = BeautifulSoup(resp.text, "html.parser")
                html_tag = soup_head.find("html")
                if html_tag and html_tag.has_attr("lang"):
                    lang = (html_tag["lang"] or "").strip() or None
                link_canon = soup_head.find("link", rel=lambda v: v and "canonical" in v)
                if link_canon and link_canon.has_attr("href"):
                    canonical = normalize_url(urljoin(final_url, link_canon["href"]))
            except Exception:
                pass

            # Outgoing links (normalized, HTML-like only) with anchor text
            out_links: List[Dict[str, str]] = []
            try:
                try:
                    soup = BeautifulSoup(resp.text, "lxml")
                except Exception:
                    soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a.get("href", "").strip()
                    if not href or href.startswith("#") or href.startswith(("mailto:", "javascript:", "tel:")):
                        continue
                    joined = urljoin(final_url, href)
                    norm = normalize_url(joined)
                    if is_asset_url(norm):
                        continue
                    anchor_text = a.get_text(" ", strip=True)
                    out_links.append({"url": norm, "text": anchor_text})
            except Exception as e:
                log(f"[WARN] link extraction failed {final_url}: {e}")

            record = {
                "url": final_url,
                "depth": depth,
                "title": title,
                "text": text,
                "content": text,
                "status": status,
                "content_type": ctype,
                "encoding": resp.encoding or None,
                "fetched_at": fetched_at,
                "lang": lang,
                "canonical_url": canonical,
                "parent": parent,
                "links": out_links,
                "char_count": len(text),
                "word_count": len(text.split()) if text else 0,
            }
            if out_fh:
                if json_array_mode:
                    if first_record_written:
                        out_fh.write(",\n")
                    out_fh.write(json.dumps(record, ensure_ascii=False))
                    first_record_written = True
                else:
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                # Ensure data is persisted frequently in case of interruption
                try:
                    out_fh.flush()
                    if fsync_writes:
                        os.fsync(out_fh.fileno())
                except Exception:
                    pass
            else:
                print(json.dumps(record, ensure_ascii=False))

            # Optionally echo records to stdout even when writing to file
            if out_fh and tee:
                try:
                    print(json.dumps(record, ensure_ascii=False))
                    sys.stdout.flush()
                except Exception:
                    pass
            pages_crawled += 1

            if depth >= max_depth:
                if delay > 0:
                    time.sleep(delay)
                continue

            # Enqueue links
            new_count = 0
            for link in out_links:
                norm = link.get("url")
                if not norm:
                    continue
                next_is_same = same_reg_domain(norm, start_url_norm, include_subdomains)
                next_ext_hops = 0 if next_is_same else (ext_hops + 1 if ext_hops > 0 or not is_same else 1)
                if not next_is_same:
                    # Only crawl github.io as external
                    host = (urlparse(norm).hostname or "").lower()
                    if not is_github_io_host(host):
                        continue
                    # Enforce external depth limit
                    if external_depth < 1:
                        continue
                    if next_ext_hops > external_depth:
                        continue
                if norm not in seen and norm not in visited:
                    queue.append((norm, depth + 1, final_url, next_ext_hops))
                    seen.add(norm)
                    new_count += 1

            log(f"[ENQUEUE] +{new_count} from {final_url} (queue={len(queue)})")

            if delay > 0:
                time.sleep(delay)

    finally:
        if out_fh:
            if json_array_mode:
                out_fh.write("]\n")
            out_fh.close()
        log(f"[DONE] pages={pages_crawled} visited={len(visited)} queue={len(queue)}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Crawl a website up to depth and write JSONL of pages.")
    p.add_argument("--url", required=True, help="Starting URL (e.g., https://iisc.ac.in)")
    p.add_argument("--depth", type=int, default=8, help="Max BFS depth (default: 8)")
    p.add_argument("--same-domain", action="store_true", help="Restrict crawl to the same domain as the start URL")
    p.add_argument("--include-subdomains", action="store_true", help="When restricting domain, include subdomains")
    p.add_argument("--external-depth", type=int, default=2, help="Depth limit for external links (0 to disable; default: 2)")
    p.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds (default: 0.5)")
    p.add_argument("--output", help="Path to output file; stdout if omitted")
    p.add_argument("--format", choices=["jsonl", "json"], default="jsonl", help="Output format: jsonl (one JSON per line) or json (array)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout in seconds")
    p.add_argument("--verbose", action="store_true", help="Verbose logs to stderr")
    p.add_argument("--ignore-robots", action="store_true", help="Ignore robots.txt (use only with permission)")
    p.add_argument("--fsync", action="store_true", help="Force fsync after each record write for crash-safe durability (use with --format jsonl)")
    p.add_argument("--tee", action="store_true", help="Echo each JSON record to stdout while also writing to output file")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    crawl(
        start_url=args.url,
        max_depth=args.depth,
        same_domain=args.same_domain,
        include_subdomains=args.include_subdomains,
        delay=args.delay,
        output_path=args.output,
        output_format=args.format,
        timeout=args.timeout,
        verbose=args.verbose,
        ignore_robots=args.ignore_robots,
        external_depth=args.external_depth,
        fsync_writes=args.fsync,
        tee=args.tee,
    )


if __name__ == "__main__":
    main()
