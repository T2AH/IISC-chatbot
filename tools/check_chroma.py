"""
Quick ChromaDB check script using the project's DatabaseManager.

Run:
  python tools/check_chroma.py

This prints basic ChromaDB collection stats (name and count).
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` package imports work when running the script directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.database.manager import DatabaseManager


def main():
    print("Checking ChromaDB via DatabaseManager...")
    db = DatabaseManager()
    try:
        stats = db.get_stats()
        chroma = stats.get('chromadb') if stats else None
        print("ChromaDB stats:", chroma)
    except Exception as e:
        print("Error checking ChromaDB:", e)
    finally:
        db.close()


if __name__ == '__main__':
    main()
