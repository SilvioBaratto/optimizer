# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy and install API requirements
COPY api/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy and install the optimizer library from project root
COPY pyproject.toml README.md ./
COPY optimizer/ optimizer/
RUN pip install --no-cache-dir --user -e .

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user first
RUN useradd -m -u 1000 appuser

# Copy Python dependencies from builder to appuser's home
COPY --from=builder /root/.local /home/appuser/.local

# Copy optimizer library (needed for editable install)
COPY --from=builder /app/optimizer /app/optimizer
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Copy application code
COPY api/ ./api/

# Generate BAML client (before changing to non-root user)
ENV PYTHONPATH=/home/appuser/.local/lib/python3.12/site-packages
WORKDIR /app/api
RUN /home/appuser/.local/bin/baml-cli generate

# Copy and make entrypoint script executable
COPY api/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Change ownership
RUN chown -R appuser:appuser /app && \
    chown -R appuser:appuser /home/appuser/.local && \
    chown appuser:appuser /entrypoint.sh

# Switch to non-root user
USER appuser

# Make sure scripts in .local are usable
ENV PATH=/home/appuser/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

# Use entrypoint script to run migrations and setup before starting the app
ENTRYPOINT ["/entrypoint.sh"]

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
