from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.main import app
from app.db.session import Base, get_db


def make_client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def _png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_visual_incident_metrics_spike():
    client = make_client()
    img = Image.new("RGB", (128, 128), (24, 28, 36))
    draw = ImageDraw.Draw(img)
    draw.line([(8, 85), (55, 85)], fill=(120, 140, 170), width=2)
    draw.line([(55, 85), (85, 20), (120, 75)], fill=(235, 64, 52), width=4)
    draw.rectangle([77, 12, 93, 28], fill=(255, 90, 60))

    resp = client.post(
        "/incidents/visual",
        data={"title": "Pico de latencia en panel"},
        files={"file": ("spike.png", _png_bytes(img), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_category"] == "metrics_spike"
    assert isinstance(data["confidence"], float)
    assert len(data["recommended_action"]) > 0


def test_visual_incident_error_screen():
    client = make_client()
    img = Image.new("RGB", (128, 128), (58, 14, 18))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 16, 118, 42], fill=(190, 32, 42))
    for y in range(54, 112, 12):
        draw.rectangle([14, y, 108, y + 6], fill=(220, 75, 85))

    resp = client.post(
        "/incidents/visual",
        data={"title": "Pantalla de error 500"},
        files={"file": ("error.png", _png_bytes(img), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_category"] == "error_screen"
    assert len(data["recommended_action"]) > 0


def test_visual_incident_invalid_file():
    client = make_client()
    resp = client.post(
        "/incidents/visual",
        files={"file": ("bad.png", b"not-an-image", "image/png")},
    )
    assert resp.status_code == 400
