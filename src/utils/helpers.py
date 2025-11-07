# c:\Users\harsh\Documents\chat application\src\utils\helpers.py
import os
from typing import Dict

from config import CRAWLER_CONFIG, CHUNKER_CONFIG, VECTOR_STORE_CONFIG, GEMINI_API_KEY


def get_config() -> Dict:
    """Return unified config dict sourced from config.py."""
    return {
        "crawler": {
            "start_urls": CRAWLER_CONFIG["start_urls"],
            "output_file": CRAWLER_CONFIG["output_file"],
            "max_depth": CRAWLER_CONFIG["max_depth"],
            "allowed_domains": CRAWLER_CONFIG.get("allowed_domains", []),
        },
        "chunker": CHUNKER_CONFIG,
        "vector_store": VECTOR_STORE_CONFIG,
        "llm": {
            "api_key": GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        }
    }
