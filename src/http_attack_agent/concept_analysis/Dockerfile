# CPU-only offline WAF runner; use the host's A100 environment for encode/analyze.
FROM python:3.11-slim-bookworm
ARG CRS_VERSION=4.10.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmodsecurity3 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/crs && curl -fL \
    https://github.com/coreruleset/coreruleset/archive/refs/tags/v${CRS_VERSION}.tar.gz \
    | tar -xz --strip-components=1 -C /opt/crs
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "http_attack_agent.concept_analysis"]
