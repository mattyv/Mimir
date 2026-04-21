"""Lightweight HTTP /health endpoint for external monitoring (§13.2).

Exposes a minimal WSGI application that returns a JSON health payload.
Can be served standalone via wsgiref for local dev, or mounted under
any WSGI-compatible server (gunicorn, uvicorn with wsgi middleware) in prod.

Usage (standalone):
    python -m mimir.mcp.health
"""

from __future__ import annotations

import json
import os
from typing import Any
from wsgiref.simple_server import make_server

import psycopg
from psycopg.rows import dict_row


def _health_payload() -> dict[str, Any]:
    dsn = os.environ.get("DATABASE_URL", "dbname=mimir user=root")
    try:
        with psycopg.connect(dsn, row_factory=dict_row, autocommit=True) as conn:
            meta = conn.execute("SELECT version, updated_at FROM graph_meta WHERE id = 1").fetchone()
            entity_row = conn.execute(
                "SELECT COUNT(*) AS n FROM entities WHERE valid_until IS NULL"
            ).fetchone()
        return {
            "status": "ok",
            "graph_version": int(meta["version"]) if meta else 0,
            "active_entities": int(entity_row["n"]) if entity_row else 0,
            "last_update": meta["updated_at"].isoformat() if meta else None,
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def health_app(
    environ: dict[str, Any],
    start_response: Any,
) -> list[bytes]:
    """WSGI application returning a JSON health check."""
    path = environ.get("PATH_INFO", "/")
    if path != "/health":
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"Not Found"]

    payload = _health_payload()
    body = json.dumps(payload).encode()
    status_code = "200 OK" if payload.get("status") == "ok" else "503 Service Unavailable"
    start_response(status_code, [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
    return [body]


if __name__ == "__main__":
    host = os.environ.get("HEALTH_HOST", "0.0.0.0")
    port = int(os.environ.get("HEALTH_PORT", "8080"))
    with make_server(host, port, health_app) as httpd:
        print(f"Mimir health endpoint: http://{host}:{port}/health")
        httpd.serve_forever()
