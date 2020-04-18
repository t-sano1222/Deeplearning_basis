FROM python:3.7.0-slim

WORKDIR /app
COPY requirements.txt /app/

RUN apt-get update \
  && apt-get -y upgrade \
  && pip install --no-cache-dir -r requirements.txt \
  && rm -rf /tmp/* /var/tmp/* \
  && rm -rf /var/lib/apt/lists/* \
  rm -rf /var/lib/apt/lists/*

WORKDIR /work