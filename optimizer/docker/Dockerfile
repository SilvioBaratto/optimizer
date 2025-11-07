# Dockerfile for Portfolio Optimizer - Daily Analysis Runner
# Runs macro regime and stock signal analysis at midnight daily
# syntax=docker/dockerfile:1.4

# Use full python:3.11-bullseye (not slim) for better package support
FROM python:3.11-bullseye

# Set working directory
WORKDIR /app

# Install cron for scheduled tasks
RUN apt-get update && \
    apt-get install -y --no-install-recommends cron && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /app/

# Test if pip install works with Docker Desktop macOS DNS
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 60 -r /app/requirements.txt

# Copy entire optimizer application (build context is parent directory)
COPY . /app/

# Create logs directory and make scripts executable
RUN mkdir -p /var/log/optimizer && \
    chmod +x /app/docker/run_daily_analysis.sh /app/docker/entrypoint.sh

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Use entrypoint script
ENTRYPOINT ["/app/docker/entrypoint.sh"]

# Start cron in foreground mode (keeps container running)
CMD ["cron", "-f"]
