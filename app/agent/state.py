from typing import TypedDict


class IncidentState(TypedDict):
    title: str
    description: str
    category: str
    confidence: float
    recommended_action: str
