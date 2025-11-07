# c:\Users\harsh\Documents\chat application\scripts\preprocess_and_chunk.py
import json
import os
import sys
import argparse
from typing import Dict, List
from collections import defaultdict
import hashlib

# Ensure project root is on sys.path so `config` and `src` can be imported reliably
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import CRAWLER_CONFIG
from src.data_processing.text_clean import basic_clean


def load_pages(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Clean text and deduplicate pages")
    parser.add_argument("--in", dest="input", type=str, default=None, help="Input crawled JSON; defaults to config.CRAWLER_CONFIG.output_file")
    args = parser.parse_args()

    crawled_path = args.input or CRAWLER_CONFIG["output_file"]
    pages = load_pages(crawled_path)

    # 1) Clean text while preserving metadata
    cleaned_pages = []
    for p in pages:
        cleaned_pages.append({
            "url": p.get("url"),
            "title": p.get("title"),
            "depth": p.get("depth"),
            "content": basic_clean(p.get("content", "")),
            # propagate links (if present) for link-graph building later
            "links": p.get("links", []),
        })

    cleaned_out = "data/processed/cleaned_pages.json"
    save_json(cleaned_pages, cleaned_out)

    # 2) Find duplicates by URL and write a small report
    url_counts: Dict[str, int] = defaultdict(int)
    for p in cleaned_pages:
        if p.get("url"):
            url_counts[p["url"]] += 1
    duplicates = [{"url": u, "count": c} for u, c in url_counts.items() if c > 1]
    duplicates_sorted = sorted(duplicates, key=lambda x: x["count"], reverse=True)
    dup_out = "data/processed/duplicates_by_url.json"
    save_json(duplicates_sorted, dup_out)

    # 3) Save a deduplicated cleaned file (first occurrence kept)
    seen = set()
    deduped: List[Dict] = []
    for p in cleaned_pages:
        url = p.get("url")
        if not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        deduped.append(p)
    dedup_out = "data/processed/cleaned_pages_dedup.json"
    save_json(deduped, dedup_out)

    # 4) Deduplicate by exact content regardless of URL/title/depth
    content_map: Dict[str, List[int]] = defaultdict(list)
    for idx, p in enumerate(cleaned_pages):
        content_map[p.get("content", "")].append(idx)
    content_dups = [
        {
            "content_sample": k[:200],
            "count": len(v),
            "indices": v,
            "urls": [cleaned_pages[i].get("url") for i in v]
        }
        for k, v in content_map.items() if len(v) > 1
    ]
    content_dups_sorted = sorted(content_dups, key=lambda x: x["count"], reverse=True)
    dups_by_content_out = "data/processed/duplicates_by_content.json"
    save_json(content_dups_sorted, dups_by_content_out)

    # Keep only first instance of each unique content, and record a duplicate->canonical URL map
    seen_content = set()
    dedup_by_content: List[Dict] = []
    dup_to_canonical: Dict[str, str] = {}
    def _hash_text(t: str) -> str:
        return hashlib.sha1((t or "").encode("utf-8")).hexdigest()

    content_canonical_url: Dict[str, str] = {}
    for p in cleaned_pages:
        c = p.get("content", "")
        if c in seen_content:
            # map duplicate URL to canonical URL for this content
            if p.get("url"):
                canon = content_canonical_url.get(c)
                if canon:
                    dup_to_canonical[p["url"]] = canon
            continue
        seen_content.add(c)
        # first occurrence is canonical for this content
        content_canonical_url[c] = p.get("url") or ""
        dedup_by_content.append(p)
    dedup_by_content_out = "data/processed/cleaned_pages_dedup_by_content.json"
    save_json(dedup_by_content, dedup_by_content_out)

    # Save duplicate->canonical mapping for downstream link-graph collapsing
    dup_map_out = "data/processed/duplicate_url_to_canonical.json"
    save_json(dup_to_canonical, dup_map_out)

    print(f"Cleaned pages saved to: {cleaned_out} (pages={len(cleaned_pages)})")
    print(f"Duplicate report (by URL): {dup_out} (dups={len(duplicates_sorted)})")
    print(f"Deduplicated cleaned (by URL) saved to: {dedup_out} (pages={len(deduped)})")
    print(f"Duplicate report (by exact content): {dups_by_content_out} (dups={len(content_dups_sorted)})")
    print(f"Deduplicated cleaned (by content) saved to: {dedup_by_content_out} (pages={len(dedup_by_content)})")
    print(f"Duplicate→Canonical URL map saved to: {dup_map_out} (entries={len(dup_to_canonical)})")


if __name__ == "__main__":
    main()
