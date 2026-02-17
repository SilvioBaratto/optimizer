#!/bin/bash
set -e

echo "Starting FastAPI Template API..."

# Wait for database to be ready
echo "Waiting for database to be ready..."
until pg_isready -h db -p 5432 -U postgres; do
    echo "Database is unavailable - sleeping"
    sleep 2
done
echo "Database is ready!"

# Run migrations using DATABASE_URL from environment
echo "Running database migrations..."
alembic upgrade head
echo "Migrations completed!"

echo "Setup complete! Starting application..."

# Execute the main command (start uvicorn)
exec "$@"
