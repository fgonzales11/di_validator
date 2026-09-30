"""Cloud-only database and process support; local workspaces continue using SQLite."""

from __future__ import annotations

import os
from contextlib import contextmanager


def database_url():
    return os.environ.get("DI_DATABASE_URL", "")


class Row(dict):
    def __getitem__(self, key):
        return list(self.values())[key] if isinstance(key, int) else super().__getitem__(key)


def row_factory(cursor):
    columns = [column.name for column in cursor.description] if cursor.description else []
    return lambda values: Row(zip(columns, values))


class Database:
    def __init__(self, connection):
        self.raw = connection

    def execute(self, query, values=()):
        if query == "BEGIN IMMEDIATE":
            query = "BEGIN"
        return self.raw.execute(query.replace("?", "%s"), values)

    def executescript(self, script):
        for statement in script.split(";"):
            if statement.strip():
                self.raw.execute(statement)


@contextmanager
def connection():
    import psycopg

    with psycopg.connect(database_url(), row_factory=row_factory, connect_timeout=15) as raw:
        yield Database(raw)


@contextmanager
def worker_lease():
    """One worker across old/new Cloud Run revisions, held on a live DB session."""
    import psycopg

    with psycopg.connect(database_url(), autocommit=True, connect_timeout=15) as raw:
        acquired = raw.execute("SELECT pg_try_advisory_lock(741946251)").fetchone()[0]
        try:
            yield raw if acquired else None
        finally:
            if acquired and not raw.closed:
                raw.execute("SELECT pg_advisory_unlock(741946251)")
