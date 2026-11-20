from langgraph.graph import END, StateGraph

from app.agent.state import IncidentState
from app.ml.classifier import classifier

ACTION_PLAYBOOKS: dict[str, str] = {
    "database": (
        "Revisar conexiones activas, bloqueos (locks) y consultas lentas en "
        "PostgreSQL; verificar espacio en disco y pool de conexiones."
    ),
    "kubernetes": (
        "Inspeccionar eventos y logs del pod (`kubectl describe/logs`), "
        "revisar límites de memoria/CPU (OOMKilled) y estado de probes."
    ),
    "network": (
        "Verificar latencia, resolución DNS, reglas de firewall/security groups "
        "y estado del balanceador o certificado TLS."
    ),
    "auth": (
        "Validar expiración y firma de tokens JWT/OAuth, configuración de SSO "
        "y permisos/roles del usuario o servicio."
    ),
    "metrics_spike": (
        "Analizar el pico en el panel de métricas: revisar CPU, memoria, latencia "
        "p95/p99 y despliegues recientes asociados al intervalo del pico."
    ),
    "error_screen": (
        "Inspeccionar la pantalla o traza de error: revisar logs de aplicación, "
        "códigos HTTP/excepciones recientes y reproducir el fallo en staging."
    ),
    "healthy_dashboard": (
        "Panel en estado normal sin anomalías críticas: mantener monitoreo "
        "continuo y documentar la línea base si corresponde."
    ),
}


def classify_node(state: IncidentState) -> IncidentState:
    existing_category = state.get("category", "").strip()
    existing_confidence = float(state.get("confidence", 0.0) or 0.0)
    if existing_category and existing_confidence > 0.0:
        return {
            **state,
            "category": existing_category,
            "confidence": existing_confidence,
        }

    combined_text = f"{state['title']} {state['description']}".strip()
    category, confidence = classifier.predict(combined_text)
    return {
        **state,
        "category": category,
        "confidence": confidence,
    }


def recommend_node(state: IncidentState) -> IncidentState:
    category = state.get("category", "network")
    action = ACTION_PLAYBOOKS.get(
        category,
        "Revisar métricas recientes, logs del servicio y escalar al equipo on-call.",
    )
    return {
        **state,
        "recommended_action": action,
    }


def build_incident_graph():
    graph = StateGraph(IncidentState)
    graph.add_node("classify", classify_node)
    graph.add_node("recommend", recommend_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "recommend")
    graph.add_edge("recommend", END)

    return graph.compile()


incident_agent = build_incident_graph()
