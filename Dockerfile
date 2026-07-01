# ── Build stage ───────────────────────────────────────────────────
# python:3.11-slim (Debian bookworm) gives us glibc so MediaPipe wheels work.
FROM python:3.11-slim AS builder

# System libraries needed by MediaPipe and OpenCV on a headless Debian image
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
        libgpiod2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps in a separate layer so rebuilds after code changes
# don't reinstall the ML packages.
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# ── Runtime stage ─────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="PalmGate" \
      org.opencontainers.image.description="Palm biometric access system"

# Re-install the same system runtime libs in the final layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
        libgpiod2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Carry over site-packages from the builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Application source
COPY app/ ./app/

# Writable directory for the SQLite database.
# Mount a named volume here (see docker-compose.yml) so data survives
# container restarts and image rebuilds.
RUN mkdir -p /data

EXPOSE 8000

# DB_PATH is overridden by docker-compose so the DB lands on the volume,
# not inside the container's writable layer.
ENV DB_PATH=/data/palmprint.db
ENV ORT_LOG_SEVERITY_LEVEL=3

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
