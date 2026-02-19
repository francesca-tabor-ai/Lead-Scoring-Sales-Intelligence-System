# ─── Build Stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# ─── API Image ───────────────────────────────────────────────────────────────
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ─── Dashboard Image ─────────────────────────────────────────────────────────
FROM base AS dashboard
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
