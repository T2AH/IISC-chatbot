import re


def basic_clean(text: str) -> str:
    """Normalize whitespace and strip odd characters.

    - Replace non-breaking spaces with regular spaces
    - Collapse multiple whitespace to a single space
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""
    # Replace non-breaking spaces and other unicode space-like chars
    text = text.replace("\xa0", " ")
    # Collapse consecutive whitespace, including newlines/tabs
    text = re.sub(r"\s+", " ", text)
    return text.strip()
