"""API-key management command group — direct DB access."""

from __future__ import annotations

import hashlib
import secrets
import sys
import uuid
from pathlib import Path

import typer

# Ensure api/ is on the path (mirrors direct_fetch.py pattern)
_API_DIR = Path(__file__).resolve().parent.parent / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from app.database import DatabaseManager  # noqa: E402
from app.models.api_key import ApiKey  # noqa: E402

auth_app = typer.Typer(name="auth", help="API-key issuance and revocation.")


def _db() -> DatabaseManager:
    manager = DatabaseManager()
    manager.initialize()
    return manager


# ------------------------------------------------------------------
# issue
# ------------------------------------------------------------------


@auth_app.command()
def issue(
    name: str = typer.Option(..., "--name", "-n", help="Human-readable label for the key."),
) -> None:
    """Generate a new API key, store its hash, and print the plaintext once."""
    plaintext = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(plaintext.encode()).hexdigest()

    manager = _db()
    try:
        with manager.get_session() as session:
            api_key = ApiKey(key_hash=key_hash, name=name)
            session.add(api_key)
            session.commit()
            key_id = str(api_key.id)
    finally:
        manager.close()

    typer.echo(f"Key ID  : {key_id}")
    typer.echo(f"Name    : {name}")
    typer.echo(f"API Key : {plaintext}")
    typer.echo("")
    typer.echo("Store this key safely — it will not be shown again.")


# ------------------------------------------------------------------
# revoke
# ------------------------------------------------------------------


@auth_app.command()
def revoke(
    key_id: str = typer.Argument(..., help="UUID of the key to revoke."),
) -> None:
    """Soft-delete an API key by setting is_active=False."""
    try:
        parsed_id = uuid.UUID(key_id)
    except ValueError:
        typer.echo(f"Error: '{key_id}' is not a valid UUID.", err=True)
        raise typer.Exit(code=1)

    manager = _db()
    try:
        with manager.get_session() as session:
            api_key = session.query(ApiKey).filter_by(id=parsed_id).first()
            if api_key is None:
                typer.echo(f"Error: key '{key_id}' not found.", err=True)
                raise typer.Exit(code=1)
            api_key.is_active = False
            session.commit()
    finally:
        manager.close()

    typer.echo(f"Key '{key_id}' revoked successfully.")
