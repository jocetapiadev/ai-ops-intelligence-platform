from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent.graph import incident_agent
from app.db.init_db import init_db
from app.db.models import Incident
from app.db.session import get_db
from app.ml.vision import ALLOWED_CONTENT_TYPES, get_visual_classifier


class IncidentCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=200)
    description: str = Field(..., min_length=3)


class IncidentResponse(BaseModel):
    id: int
    title: str
    description: str
    predicted_category: str | None
    confidence: float | None
    recommended_action: str | None

    model_config = {"from_attributes": True}


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Ops Intelligence Platform", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/incidents", response_model=IncidentResponse)
def create_incident(payload: IncidentCreate, db: Session = Depends(get_db)):
    state_result = incident_agent.invoke(
        {
            "title": payload.title,
            "description": payload.description,
            "category": "",
            "confidence": 0.0,
            "recommended_action": "",
        }
    )

    incident = Incident(
        title=payload.title,
        description=payload.description,
        predicted_category=state_result.get("category"),
        confidence=state_result.get("confidence"),
        recommended_action=state_result.get("recommended_action"),
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return incident


@app.post("/incidents/visual", response_model=IncidentResponse)
async def create_visual_incident(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {file.content_type}. Usa PNG, JPEG o WEBP.",
        )

    raw_bytes = await file.read()
    try:
        vision_result = get_visual_classifier().predict(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_title = (title or file.filename or "Incidente visual").strip()[:200] or "Incidente visual"
    visual_category = str(vision_result["category"])
    visual_confidence = float(vision_result["confidence"])
    description = f"Análisis visual ({file.filename or 'imagen'}): categoría detectada {visual_category}."

    state_result = incident_agent.invoke(
        {
            "title": resolved_title,
            "description": description,
            "category": visual_category,
            "confidence": visual_confidence,
            "recommended_action": "",
        }
    )

    incident = Incident(
        title=resolved_title,
        description=description,
        predicted_category=state_result.get("category"),
        confidence=state_result.get("confidence"),
        recommended_action=state_result.get("recommended_action"),
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return incident


@app.get("/incidents", response_model=list[IncidentResponse])
def list_incidents(db: Session = Depends(get_db)):
    return db.query(Incident).order_by(Incident.id.desc()).all()
