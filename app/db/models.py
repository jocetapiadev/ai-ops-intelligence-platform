from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.core.config import settings
from app.db.session import Base


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    predicted_category = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    recommended_action = Column(Text, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )


class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    category = Column(String(100), nullable=False)
    resolution = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding_dim), nullable=True)
