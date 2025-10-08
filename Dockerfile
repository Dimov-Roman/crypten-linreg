FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

COPY tasks/ ./tasks/
COPY config.py .
COPY worker.py .

RUN mkdir -p /data/linreg/homework

ENV PYTHONUNBUFFERED=1
ENV LOGLEVEL=INFO

CMD ["python", "-m", "worker", "call", "tasks.mpc:linreg_homework"]