FROM python:3.9-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    git patch build-essential ripgrep ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /bench
COPY harness ./harness
COPY scripts ./scripts
COPY run.py ./
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1
ENTRYPOINT ["python", "run.py"]
