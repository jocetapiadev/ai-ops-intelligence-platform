# AI Ops Intelligence Platform

Plataforma modular de inteligencia operativa construida con **FastAPI**, **LangGraph**, **PostgreSQL (`pgvector`)** y **Scikit-learn** para clasificar incidentes tecnicos, sugerir acciones operativas y persistir registros.

## Arquitectura

- `app/core/`: Configuracion de entorno (`config.py`).
- `app/db/`: Conexion SQLAlchemy (`session.py`), modelos (`models.py`) e inicializacion de extension `pgvector` y tablas (`init_db.py`).
- `app/ml/`: Limpieza de texto (`preprocessing.py`), generacion de embeddings (`embeddings.py`) y clasificador con `scikit-learn` (`classifier.py`).
- `app/agent/`: Estado (`state.py`) y grafo de orquestacion (`graph.py`) con **LangGraph**.
- `app/api/`: Endpoints REST (`main.py`) con **FastAPI**.
- `tests/`: Pruebas automatizadas con `pytest`.

## Levantar el entorno con Docker Compose

```bash
docker compose up --build -d
```

Una vez iniciado, la API queda disponible en `http://localhost:8000` y la documentacion interactiva en `http://localhost:8000/docs`.

## Ejemplo de uso

```bash
curl -X POST http://localhost:8000/incidents \
  -H "Content-Type: application/json" \
  -d "{\"title\": \"Timeout en consultas SQL\", \"description\": \"Connection pool exhausted and slow sql query on postgres\"}"
```
