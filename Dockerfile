FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# ffmpeg for transcoding reliability
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all app code (your current root layout)
COPY . /app

# Default to API; worker overrides via docker-compose command
EXPOSE 8000
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]
