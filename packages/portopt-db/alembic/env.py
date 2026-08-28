import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The URL is injected via the DATABASE_URL environment variable (the consumer —
# ingestion, a future fund/, or a CI/dev shell — sets it). portopt_db does not
# import any application settings, so it cannot read the daemon's .env. To avoid
# silently migrating the credentialed alembic.ini default DB when the caller
# forgot to export DATABASE_URL, online migrations refuse to run without it.
# Offline mode (`--sql`) only renders SQL against the ini URL and never touches a
# real database, so the fallback is harmless there.
_db_url = os.getenv("DATABASE_URL")
if _db_url:
    config.set_main_option("sqlalchemy.url", _db_url)
elif not context.is_offline_mode():
    raise RuntimeError(
        "DATABASE_URL is not set — refusing to run online migrations against "
        "the alembic.ini default database. Export DATABASE_URL (or run via "
        "Docker, which sets it) before `alembic upgrade`."
    )

# Importing the models package registers every model on Base.metadata, so
# autogenerate/upgrade see the complete schema.
from portopt_db.models import Base

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
