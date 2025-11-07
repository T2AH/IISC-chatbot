import json
import os
import sys
import argparse
from collections import defaultdict

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import CRAWLER_CONFIG


INPUT_PATH = "data/processed/cleaned_pages.json"  # has links propagated
OUTPUT_PATH = "data/processed/link_graph.json"


def main():
    parser = argparse.ArgumentParser(description="Build link graph (outgoing/incoming)")
    parser.add_argument("--in", dest="input", type=str, default=None, help="Input pages JSON that has 'links'")
    parser.add_argument("--out", dest="output", type=str, default=None, help="Output link graph JSON path")
    args = parser.parse_args()

    # Prefer the latest cleaned set that preserved links; if not present, fallback
    in_path = args.input or INPUT_PATH
    if not os.path.exists(in_path):
        in_path = CRAWLER_CONFIG["output_file"]

    with open(in_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # Load duplicate→canonical URL map if present, to normalize edges
    dup_map_path = os.path.join(os.path.dirname(in_path), "duplicate_url_to_canonical.json")
    dup_to_canon = {}
    if os.path.exists(dup_map_path):
        try:
            with open(dup_map_path, "r", encoding="utf-8") as df:
                dup_to_canon = json.load(df)
        except Exception:
            dup_to_canon = {}

    def canon(url: str) -> str:
        if not url:
            return url
        return dup_to_canon.get(url, url)

    outgoing = defaultdict(list)  # url -> list[{url, text}]
    incoming = defaultdict(list)  # url -> list[{from_url, text}]

    for p in pages:
        src = p.get("url")
        for link in (p.get("links") or []):
            tgt = link.get("url")
            text = link.get("text") or ""
            if not src or not tgt:
                continue
            csrc = canon(src)
            ctgt = canon(tgt)
            outgoing[csrc].append({"url": ctgt, "text": text})
            incoming[ctgt].append({"from_url": csrc, "text": text})

    graph = {
        "outgoing": outgoing,
        "incoming": incoming,
    }

    # Convert defaultdicts to normal dicts for JSON serialization
    graph = {k: {kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v for k, v in graph.items()}

    out_path = args.output or OUTPUT_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"Link graph written to: {out_path} (nodes={len(outgoing) + len(incoming)})")


if __name__ == "__main__":
    main()
