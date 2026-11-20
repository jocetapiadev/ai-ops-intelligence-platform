# AI Ops Intelligence Platform

Plataforma modular de inteligencia operativa construida con **FastAPI**, **LangGraph**, **PostgreSQL (`pgvector`)** y **Scikit-learn** para clasificar incidentes técnicos y anomalías visuales, sugerir acciones operativas y persistir registros.

## Arquitectura

- `app/core/`: Configuración de entorno (`config.py`).
- `app/db/`: Conexión SQLAlchemy (`session.py`), modelos (`models.py`) e inicialización de extensión `pgvector` y tablas (`init_db.py`).
- `app/ml/`: Limpieza de texto (`preprocessing.py`), generación de embeddings (`embeddings.py`), clasificador de texto con `scikit-learn` (`classifier.py`) y clasificador visual (`vision.py`).
- `app/agent/`: Estado (`state.py`) y grafo de orquestación (`graph.py`) con **LangGraph**.
- `app/api/`: Endpoints REST (`main.py`) con **FastAPI**.
- `tests/`: Pruebas automatizadas con `pytest`.

## Ejecución en local (entorno virtual)

### 1. Crear y activar el entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
2. Instalar dependencias
Bash
pip install --upgrade pip
pip install -r requirements.txt
Levantar la API con Uvicorn
Con base de datos local SQLite (ideal para pruebas rápidas sin depender de PostgreSQL):
O con PostgreSQL configurado en tu entorno o .env:

Bash
uvicorn app.api.main:app --reload
Una vez iniciado, la API queda disponible en http://127.0.0.1:8000 y la documentación interactiva en http://127.0.0.1:8000/docs.

4. Ejecutar pruebas
Bash
pytest -v
Levantar el entorno con Docker Compose
Bash
docker compose up --build -d
Una vez iniciado, la API queda disponible en http://localhost:8000 y la documentación interactiva en http://localhost:8000/docs.

Ejemplos de uso
Clasificación de incidente por texto
Bash
curl -X POST [http://127.0.0.1:8000/incidents](http://127.0.0.1:8000/incidents) \
  -H "Content-Type: application/json" \
  -d '{"title": "Timeout en consultas SQL", "description": "Connection pool exhausted and slow sql query on postgres"}'
Clasificación de anomalía visual
Bash
curl -X POST [http://127.0.0.1:8000/incidents/visual](http://127.0.0.1:8000/incidents/visual) \
  -F "file=@ruta/a/imagen.png" \
  -F "title=Dashboard con hora de latencia"

Bash
DATABASE_URL="sqlite:///./local_ops.db" uvicorn app.api.main:app --reload
