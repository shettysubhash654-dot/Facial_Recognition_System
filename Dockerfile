FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY frontend ./frontend

ENV VISIONID_EPHEMERAL=1 \
    VISIONID_RESET_ON_STARTUP=1 \
    VISIONID_IDLE_RESET_MINUTES=120

EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
