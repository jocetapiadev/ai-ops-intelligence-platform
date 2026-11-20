import hashlib
import math

from app.core.config import settings
from app.ml.preprocessing import clean_text


def text_to_embedding(text: str, dim: int | None = None) -> list[float]:
    target_dim = dim or settings.embedding_dim
    cleaned = clean_text(text)
    if not cleaned:
        return [0.0] * target_dim

    values: list[float] = []
    counter = 0
    while len(values) < target_dim:
        digest = hashlib.sha256(f"{cleaned}:{counter}".encode("utf-8")).digest()
        for byte in digest:
            if len(values) < target_dim:
                values.append((byte / 127.5) - 1.0)
        counter += 1

    norm = math.sqrt(sum(v * v for v in values))
    if norm > 0:
        values = [v / norm for v in values]
    return values
