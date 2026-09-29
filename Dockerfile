FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt requirements-postgres.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-postgres.txt
COPY ml ./ml
COPY backend ./backend
COPY scripts ./scripts
COPY frontend ./frontend
COPY data/eval ./data/eval
EXPOSE 8000
# models/ and data/ are mounted as volumes (trained with scripts/run_all.py)
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
