# c:\Users\harsh\Documents\chat application\scripts\chunk_pages.py
import json
import os
import sys
from typing import Dict, List

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import CHUNKER_CONFIG
from src.data_processing.chunker import Chunker


INPUT_PATH = "data/processed/cleaned_pages_dedup_by_content.json"
OUTPUT_PATH = "data/processed/chunks.jsonl"


def load_pages(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    pages = load_pages(INPUT_PATH)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    chunk_size = CHUNKER_CONFIG["chunk_size"]
    overlap = CHUNKER_CONFIG["overlap"]

    total_chunks = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for page_idx, p in enumerate(pages):
            text = (p.get("content") or "").strip()
            if not text:
                continue
            chunker = Chunker(text, chunk_size, overlap)
            parts = chunker.chunk()
            for ci, ch in enumerate(parts):
                ch = ch.strip()
                if not ch:
                    continue
                rec = {
                    "text": ch,
                    "url": p.get("url"),
                    "title": p.get("title"),
                    "depth": p.get("depth"),
                    "page_index": page_idx,
                    "chunk_index": ci,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Chunks written: {total_chunks} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
