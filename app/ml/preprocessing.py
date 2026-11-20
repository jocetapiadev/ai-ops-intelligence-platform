import re


def clean_text(text: str) -> str:
    if not text:
        return ""
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized
