# Build stage — install Python deps + download supercronic
FROM python:3.12-slim AS builder

ARG TARGETARCH

WORKDIR /app

# Install supercronic (cron replacement that logs to stdout)
ARG SUPERCRONIC_VERSION=v0.2.33
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && ARCH=$(case "${TARGETARCH}" in amd64) echo "amd64" ;; arm64) echo "arm64" ;; *) echo "amd64" ;; esac) \
    && curl -fsSL "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-${ARCH}" \
       -o /usr/local/bin/supercronic \
    && chmod +x /usr/local/bin/supercronic \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CLI only needs a subset)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 scheduler

# Copy supercronic binary
COPY --from=builder /usr/local/bin/supercronic /usr/local/bin/supercronic

# Copy Python dependencies
COPY --from=builder /root/.local /home/scheduler/.local

# Copy CLI code and project config
COPY cli/ /app/cli/
COPY pyproject.toml /app/

# Copy scheduler assets
COPY scheduler/crontab /etc/supercronic/crontab
COPY scheduler/fetch.sh /app/fetch.sh
RUN chmod +x /app/fetch.sh

# Set ownership
RUN chown -R scheduler:scheduler /app /home/scheduler/.local

# Switch to non-root user
USER scheduler

ENV PATH=/home/scheduler/.local/bin:$PATH
ENV PYTHONPATH=/home/scheduler/.local/lib/python3.12/site-packages

CMD ["supercronic", "/etc/supercronic/crontab"]
