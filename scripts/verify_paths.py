import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Verify effective/referential paths for a URL in chunks JSONL")
    parser.add_argument("--chunks", type=str, default="data/processed/chunks_tree.jsonl", help="Chunks JSONL path")
    parser.add_argument("--url", type=str, required=True, help="Target URL to match")
    parser.add_argument("--limit", type=int, default=5, help="Max matches to print")
    args = parser.parse_args()

    if not os.path.exists(args.chunks):
        print(f"Chunks file not found: {args.chunks}")
        sys.exit(1)

    count = 0
    with open(args.chunks, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("url") == args.url:
                count += 1
                ep = rec.get("effective_path") or []
                rp = rec.get("referential_paths") or []
                up = rec.get("url_heading_path") or []
                print("-")
                print(f"effective_path: {[seg['text'] for seg in ep]}")
                if rp:
                    print(f"referential_paths[0]: {[seg['text'] for seg in rp[0]]}")
                print(f"url_heading_path: {[seg['text'] for seg in up]}")
                if count >= args.limit:
                    break

    if count == 0:
        print("No matches found.")

if __name__ == "__main__":
    main()
