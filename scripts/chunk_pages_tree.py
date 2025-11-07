# c:\Users\harsh\Documents\chat application\scripts\chunk_pages_tree.py
import json
import os
import sys
import argparse

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import CHUNKER_CONFIG
from src.data_processing.tree_chunker import TreeSmartChunker
from src.utils.url_hierarchy import build_url_heading_path


DEFAULT_INPUT_PATH = "data/processed/cleaned_pages_dedup_by_content.json"
DEFAULT_LINK_GRAPH_PATH = "data/processed/link_graph.json"
DEFAULT_OUTPUT_PATH = "data/processed/chunks_tree.jsonl"


def _load_link_graph(path: str):
    if not os.path.exists(path):
        return {"incoming": {}, "outgoing": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _referential_paths(url: str, link_graph: dict, pages: list):
    """Build referential paths for a URL from incoming links.

    For each incoming link (from_url, text), map:
      effective_ref_path = url_path(from_url) + [link text]
    We return a list of heading-path lists.
    """
    refs_scored = []
    incoming = (link_graph.get("incoming") or {}).get(url) or []
    try:
        from urllib.parse import urlparse
        tgt_host = urlparse(url).netloc
    except Exception:
        tgt_host = ""
    for inc in incoming:
        from_url = inc.get("from_url")
        anchor = (inc.get("text") or "").strip()
        if not from_url:
            continue
        # Skip self-links to avoid picking the target page as its own referrer (e.g., nav anchors)
        if from_url == url:
            continue
        base = build_url_heading_path(from_url)
        if anchor:
            base = base + [{"level": (base[-1]["level"] + 1) if base else 1, "text": anchor}]
        # scoring: prefer same host, non-empty anchor, deeper base path
        try:
            src_host = urlparse(from_url).netloc
        except Exception:
            src_host = ""
        same_host = 1 if src_host and tgt_host and (src_host == tgt_host) else 0
        has_anchor = 1 if anchor else 0
        depth_score = len(base)
        # Additional universal preference: if the referrer URL path starts with people/, boost it
        path_prefix_boost = 0
        try:
            from urllib.parse import urlparse as _up
            _p = _up(from_url)
            _segs = [s for s in _p.path.split('/') if s]
            if _segs and _segs[0] in {"people", "people-all"}:
                path_prefix_boost = 6  # strong preference for person directory pages
        except Exception:
            pass

        score = same_host * 10 + has_anchor * 5 + path_prefix_boost + depth_score
        refs_scored.append((score, base))
    # Heuristic fallback: if no explicit incoming links, try to infer a parent person page
    if not refs_scored:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            segs = [s for s in parsed.path.split('/') if s]
            if len(segs) >= 2 and segs[0] in {"faculty", "lab", "group"}:
                last = segs[-1]
                token = last.replace('-', ' ').strip()
                # Find a people page whose slug contains this token (e.g., yogesh-simmhan contains simmhan)
                candidates = []
                for p in pages:
                    u = p.get("url") or ""
                    pu = urlparse(u)
                    psegs = [s for s in pu.path.split('/') if s]
                    if len(psegs) >= 2 and psegs[0] == "people":
                        lastp = psegs[-1].replace('-', ' ').strip()
                        if token and token in lastp:
                            candidates.append(u)
                if candidates:
                    parent = candidates[0]
                    base = build_url_heading_path(parent)
                    label = token or segs[0]
                    base = base + [{"level": (base[-1]["level"] + 1) if base else 1, "text": label}]
                    # modest fallback score
                    refs_scored.append((1 + len(base), base))
        except Exception:
            pass
    # sort by score desc and return just paths
    refs_scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in refs_scored]


def main():
    parser = argparse.ArgumentParser(description="Tree-smart chunking with URL and referential paths")
    parser.add_argument("--in", dest="input", type=str, default=DEFAULT_INPUT_PATH, help="Input pages JSON (cleaned/dedup)")
    parser.add_argument("--links", dest="links", type=str, default=DEFAULT_LINK_GRAPH_PATH, help="Link graph JSON path")
    parser.add_argument("--out", dest="output", type=str, default=DEFAULT_OUTPUT_PATH, help="Output chunks JSONL path")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        pages = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    link_graph = _load_link_graph(args.links)
    chunker = TreeSmartChunker(
        chunk_size=CHUNKER_CONFIG["chunk_size"],
        overlap=CHUNKER_CONFIG["overlap"],
        min_chunk=300,
    )

    total = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for page_idx, p in enumerate(pages):
            text = (p.get("content") or "").strip()
            if not text:
                continue
            chunks, paths = chunker.chunk(text)
            for ci, (c, path) in enumerate(zip(chunks, paths)):
                url_path = build_url_heading_path(p.get("url") or "")
                # Also attach referential paths from incoming links (if any)
                referential_paths = _referential_paths(p.get("url") or "", link_graph, pages)
                # Prefer referential path as primary context when present, else fallback to URL path
                if referential_paths:
                    primary_context = referential_paths[0]
                else:
                    primary_context = url_path
                # Merge chosen context with content-derived headings
                effective_path = primary_context + path
                rec = {
                    "text": c,
                    "url": p.get("url"),
                    "title": p.get("title"),
                    "depth": p.get("depth"),
                    "page_index": page_idx,
                    "chunk_index": ci,
                    "heading_path": path,          # content-derived
                    "url_heading_path": url_path,  # URL-derived
                    "effective_path": effective_path,
                    "referential_paths": referential_paths,  # contexts from referrers
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1

    print(f"Tree chunks written: {total} -> {args.output}")


if __name__ == "__main__":
    main()
