from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def workspace() -> Path:
    path = Path(os.environ.get("DI_WORKSPACE", ROOT / "runtime")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return uuid.uuid4().hex


def encode(value):
    return json.dumps(value, default=str, allow_nan=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def connection():
    if os.environ.get("DI_DATABASE_URL"):
        from .cloud import connection as cloud_connection

        with cloud_connection() as db:
            yield db
        return
    db = sqlite3.connect(workspace() / "catalog.sqlite", timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def initialize():
    with connection() as db:
        if not os.environ.get("DI_DATABASE_URL"):
            db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS objects (
            kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
            created_at TEXT NOT NULL, PRIMARY KEY(kind,id)
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
            config TEXT NOT NULL, progress DOUBLE PRECISION NOT NULL DEFAULT 0,
            message TEXT NOT NULL DEFAULT '', result TEXT, error TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            cancel_requested INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS logs (
            job_id TEXT NOT NULL, time TEXT NOT NULL, message TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS objects_kind ON objects(kind,created_at);
        """)
    for name in ["datasets", "experiments", "uploads", "tmp", "exports"]:
        (workspace() / name).mkdir(exist_ok=True)


def put(kind, body, object_id=None):
    body = dict(body)
    object_id = object_id or body.get("id") or uid()
    body.update(id=object_id)
    body.setdefault("created_at", now())
    with connection() as db:
        db.execute(
            "INSERT INTO objects VALUES (?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body",
            (kind, object_id, encode(body), body["created_at"]),
        )
    return body


def get(kind, object_id):
    with connection() as db:
        row = db.execute("SELECT body FROM objects WHERE kind=? AND id=?", (kind, object_id)).fetchone()
    if not row:
        raise ValueError(f"{kind.title()} not found: {object_id}")
    return json.loads(row[0])


def listing(kind):
    with connection() as db:
        rows = db.execute(
            "SELECT body FROM objects WHERE kind=? ORDER BY created_at DESC", (kind,)
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def enqueue(kind, config):
    job_id, stamp = uid(), now()
    with connection() as db:
        db.execute(
            "INSERT INTO jobs(id,kind,status,config,created_at,updated_at) VALUES (?,?,?,?,?,?)",
            (job_id, kind, "queued", encode(config), stamp, stamp),
        )
    return job(job_id)


def job(job_id):
    with connection() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise ValueError("Job not found")
    result = dict(row)
    for field in ["config", "result"]:
        if result[field]:
            result[field] = json.loads(result[field])
    return result


def jobs():
    with connection() as db:
        ids = [r[0] for r in db.execute("SELECT id FROM jobs ORDER BY created_at DESC LIMIT 100")]
    return [job(j) for j in ids]


class Cancelled(Exception):
    pass


class Progress:
    def __init__(self, job_id=None):
        self.job_id = job_id

    def check(self):
        if self.job_id and job(self.job_id)["cancel_requested"]:
            raise Cancelled("Cancelled by user")

    def __call__(self, fraction, message):
        self.check()
        if self.job_id:
            with connection() as db:
                db.execute(
                    "UPDATE jobs SET progress=?,message=?,updated_at=? WHERE id=?",
                    (min(1, max(0, fraction)), message, now(), self.job_id),
                )
                db.execute("INSERT INTO logs VALUES (?,?,?)", (self.job_id, now(), message))
