from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
CATEGORIES = ("metrics_spike", "error_screen", "healthy_dashboard")


def load_rgb_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("El archivo de imagen está vacío.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("La imagen supera el tamaño máximo permitido (5 MB).")
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            return img.convert("RGB").resize((128, 128))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("El archivo no es una imagen válida o está corrupto.") from exc


def extract_visual_features(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]

    mean_rgb = arr.mean(axis=(0, 1))
    std_rgb = arr.std(axis=(0, 1))

    red_dominance = float(np.mean(np.maximum(0.0, red - (green + blue) * 0.5)))
    green_dominance = float(np.mean(np.maximum(0.0, green - (red + blue) * 0.5)))

    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge_density = float(gx + gy)

    top_half_red = float(red[:64, :].mean())
    bottom_half_red = float(red[64:, :].mean())

    features = np.concatenate([
        mean_rgb,
        std_rgb,
        np.array([red_dominance, green_dominance, edge_density, top_half_red, bottom_half_red], dtype=np.float32),
    ])
    return features


def build_visual_embedding(img: Image.Image, dim: int = 32) -> List[float]:
    arr = np.asarray(img.resize((32, 32)), dtype=np.float32).flatten() / 255.0
    if len(arr) >= dim:
        chunks = np.array_split(arr[: (len(arr) // dim) * dim], dim)
        vec = np.array([c.mean() for c in chunks], dtype=np.float32)
    else:
        vec = np.pad(arr, (0, dim - len(arr)), mode="constant")
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return [round(float(x), 6) for x in vec]


def _synthetic_sample(category: str, rng: np.random.Generator) -> Image.Image:
    if category == "metrics_spike":
        img = Image.new("RGB", (128, 128), (24, 28, 36))
        draw = ImageDraw.Draw(img)
        # Línea base y pico pronunciado en rojo/naranja
        base_y = int(rng.integers(75, 95))
        draw.line([(8, base_y), (55, base_y)], fill=(120, 140, 170), width=2)
        peak_x = int(rng.integers(60, 100))
        peak_y = int(rng.integers(14, 34))
        draw.line([(55, base_y), (peak_x, peak_y), (120, base_y - 10)], fill=(235, 64, 52), width=4)
        draw.rectangle([peak_x - 8, peak_y - 8, peak_x + 8, peak_y + 8], fill=(255, 90, 60))
    elif category == "error_screen":
        img = Image.new("RGB", (128, 128), (58, 14, 18))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 16, 118, 42], fill=(190, 32, 42))
        for y in range(54, 112, 12):
            w = int(rng.integers(65, 112))
            draw.rectangle([14, y, 14 + w, y + 6], fill=(220, 75, 85))
    else:  # healthy_dashboard
        img = Image.new("RGB", (128, 128), (18, 32, 28))
        draw = ImageDraw.Draw(img)
        y_level = int(rng.integers(58, 76))
        draw.line([(10, y_level), (118, y_level + int(rng.integers(-5, 6)))], fill=(46, 204, 113), width=3)
        draw.rectangle([14, 92, 56, 110], fill=(39, 174, 96))
        draw.rectangle([66, 92, 108, 110], fill=(46, 204, 113))

    noise = rng.normal(0.0, 4.0, size=(128, 128, 3))
    noisy_arr = np.clip(np.asarray(img, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr, mode="RGB").filter(ImageFilter.GaussianBlur(radius=0.4))


class VisualIncidentClassifier:
    def __init__(self, samples_per_class: int = 35, seed: int = 42) -> None:
        self.pipeline: Pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ])
        self._fit(samples_per_class=samples_per_class, seed=seed)

    def _fit(self, samples_per_class: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x_train: List[np.ndarray] = []
        y_train: List[str] = []
        for category in CATEGORIES:
            for _ in range(samples_per_class):
                img = _synthetic_sample(category, rng)
                x_train.append(extract_visual_features(img))
                y_train.append(category)
        self.pipeline.fit(np.vstack(x_train), y_train)

    def predict(self, image_bytes: bytes) -> Dict[str, object]:
        img = load_rgb_image(image_bytes)
        feats = extract_visual_features(img).reshape(1, -1)
        probs = self.pipeline.predict_proba(feats)[0]
        classes = list(self.pipeline.classes_)
        best_idx = int(np.argmax(probs))
        category = str(classes[best_idx])
        confidence = round(float(probs[best_idx]), 3)
        embedding = build_visual_embedding(img)
        return {
            "category": category,
            "confidence": confidence,
            "embedding": embedding,
        }


_classifier_instance: VisualIncidentClassifier | None = None


def get_visual_classifier() -> VisualIncidentClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = VisualIncidentClassifier()
    return _classifier_instance
