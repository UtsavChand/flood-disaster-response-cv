import sqlite3
from pathlib import Path

from backend.app.schemas import Zone


DB_PATH = Path(__file__).resolve().parents[2] / "backend" / "app" / "data" / "app.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS zones (
                id TEXT PRIMARY KEY,
                data_json TEXT NOT NULL
            )
            """
        )


def list_zones() -> list[Zone]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT data_json FROM zones ORDER BY id"
        ).fetchall()

    return [Zone.model_validate_json(row["data_json"]) for row in rows]


def add_zone(zone: Zone) -> None:
    with _connect() as connection:
        connection.execute(
            "INSERT INTO zones (id, data_json) VALUES (?, ?)",
            (zone.id, zone.model_dump_json()),
        )