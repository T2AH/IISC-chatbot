import re
from typing import List, Dict, Tuple


HeadingBlock = Dict[str, object]
ParagraphBlock = Dict[str, object]
Block = Dict[str, object]


HEADING_PATTERNS = [
    re.compile(r"^\d+(?:[\.)]|\))\s+"),            # 1. or 1)
    re.compile(r"^(?:[IVXLCM]+\.|[a-zA-Z]\))\s+"), # Roman I. II. or a)
    re.compile(r"^#{1,6}\s+"),                        # Markdown-like # ## ###
]


def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if len(s) == 0:
        return False
    # Short lines with Title Case or ALL CAPS tend to be headings
    is_short = len(s) <= 80
    few_punct = sum(ch in ",;:()[]{}" for ch in s) <= 2
    title_case_ratio = sum(w[:1].isupper() for w in s.split()) / max(1, len(s.split()))
    caps_ratio = sum(ch.isupper() for ch in s if ch.isalpha()) / max(1, sum(ch.isalpha() for ch in s))

    if any(p.match(s) for p in HEADING_PATTERNS):
        return True
    if is_short and few_punct and (title_case_ratio > 0.6 or caps_ratio > 0.7):
        return True
    return False


def infer_heading_level(text: str) -> int:
    s = text.strip()
    if s.startswith('#'):
        return min(6, max(1, len(s) - len(s.lstrip('#'))))
    # Simple numeric/roman depth heuristics
    if re.match(r"^\d+\.", s):
        parts = s.split('.')
        return min(6, len([p for p in parts if p.strip()]))
    if re.match(r"^[IVXLCM]+\.", s):
        return 2
    return 2


def split_into_blocks(text: str) -> List[Block]:
    # First split into paragraphs by blank lines
    raw_blocks = re.split(r"\n\s*\n", text)
    blocks: List[Block] = []
    for rb in raw_blocks:
        rb = rb.strip()
        if not rb:
            continue
        lines = [ln.strip() for ln in rb.splitlines() if ln.strip()]
        if not lines:
            continue
        # If first line looks like a heading, treat it as heading block, rest as paragraph
        first = lines[0]
        if looks_like_heading(first):
            level = infer_heading_level(first)
            blocks.append({"type": "heading", "level": level, "text": first})
            if len(lines) > 1:
                para = " ".join(lines[1:]).strip()
                if para:
                    blocks.append({"type": "para", "text": para})
        else:
            blocks.append({"type": "para", "text": " ".join(lines)})
    return blocks


class TreeSmartChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200, min_chunk: int = 300):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk = min_chunk

    def chunk(self, text: str) -> Tuple[List[str], List[List[Dict]]]:
        """Return chunks and their heading paths metadata.

        - Build a heading/paragraph sequence
        - Accumulate paragraphs under current heading path
        - Emit chunks up to chunk_size with overlap; if a paragraph exceeds chunk_size, split it.

        Returns:
            chunks: list of chunk texts
            paths: list of heading-path metadata per chunk (list of {level, text})
        """
        blocks = split_into_blocks(text)

        heading_stack: List[HeadingBlock] = []  # stack of {level, text}
        para_buffer: List[str] = []
        chunks: List[str] = []
        paths: List[List[Dict]] = []

        def flush_buffer():
            if not para_buffer:
                return
            full = "\n\n".join(para_buffer)
            # split full into chunks with overlap
            start = 0
            while start < len(full):
                end = min(len(full), start + self.chunk_size)
                piece = full[start:end].strip()
                if piece:
                    chunks.append(piece)
                    paths.append([{"level": h["level"], "text": h["text"]} for h in heading_stack])
                # advance with overlap
                if end == len(full):
                    break
                start += max(1, self.chunk_size - self.overlap)
            para_buffer.clear()

        for b in blocks:
            if b["type"] == "heading":
                # When heading changes, flush buffer
                flush_buffer()
                level = int(b.get("level", 2))
                # Pop stack to appropriate level-1
                while heading_stack and heading_stack[-1]["level"] >= level:
                    heading_stack.pop()
                heading_stack.append({"level": level, "text": str(b.get("text", "")).strip()})
            else:  # paragraph
                para = str(b.get("text", "")).strip()
                if not para:
                    continue
                # If paragraph is very long, split by sentences to respect chunk size better
                if len(para) > self.chunk_size * 2:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    cur = ""
                    for s in sentences:
                        if len(cur) + len(s) + 1 > self.chunk_size:
                            if cur.strip():
                                para_buffer.append(cur.strip())
                                flush_buffer()
                                cur = ""
                        cur = (cur + " " + s).strip()
                    if cur:
                        para_buffer.append(cur)
                        flush_buffer()
                else:
                    para_buffer.append(para)

        # Flush remaining
        flush_buffer()

        # Filter tiny chunks
        filtered_chunks: List[str] = []
        filtered_paths: List[List[Dict]] = []
        for c, p in zip(chunks, paths):
            if len(c) >= self.min_chunk or (not filtered_chunks):
                filtered_chunks.append(c)
                filtered_paths.append(p)
        return filtered_chunks, filtered_paths
