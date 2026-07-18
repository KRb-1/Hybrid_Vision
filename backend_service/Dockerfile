FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INFERENCE_DEVICE=cpu

WORKDIR /app

RUN apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt requirements-docker.txt ./
RUN pip install -r requirements-docker.txt

COPY backend ./backend
COPY fred_two_stream ./fred_two_stream
COPY app.py .
COPY models ./models
COPY team_b ./team_b

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
