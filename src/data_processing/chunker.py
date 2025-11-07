# c:\Users\harsh\Documents\chat application\src\data_processing\chunker.py

class Chunker:
    def __init__(self, text, chunk_size=1000, overlap=200):
        self.text = text
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self):
        """Chunks the text into smaller pieces."""
        chunks = []
        start = 0
        while start < len(self.text):
            end = start + self.chunk_size
            chunks.append(self.text[start:end])
            start += self.chunk_size - self.overlap
        return chunks
