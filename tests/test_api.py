from fastapi.testclient import TestClient
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


def test_health():
    client = make_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_and_list_incident():
    client = make_client()
    payload = {
        "title": "Timeout en consultas SQL",
        "description": "Connection pool exhausted and slow sql query on postgres",
    }
    create_resp = client.post("/incidents", json=payload)
    assert create_resp.status_code in (200, 201)

    data = create_resp.json()
    assert data["title"] == payload["title"]
    assert data["description"] == payload["description"]
    assert data["predicted_category"] == "database"
    assert isinstance(data["confidence"], float)
    assert isinstance(data["recommended_action"], str)
    assert len(data["recommended_action"]) > 0
    assert "id" in data

    list_resp = client.get("/incidents")
    assert list_resp.status_code == 200
    incidents = list_resp.json()
    assert isinstance(incidents, list)
    assert len(incidents) >= 1
    assert any(item["id"] == data["id"] for item in incidents)
