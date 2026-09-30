"""Job lifecycle operations shared by HTTP endpoints and the worker.

Keep queue state transitions here so both SQLite and hosted PostgreSQL follow
the same rules. Job execution and process ownership belong to the worker.
"""

from . import store


def detail(job_id: str) -> dict:
    result = store.job(job_id)
    with store.connection() as db:
        result["logs"] = [
            dict(row)
            for row in db.execute("SELECT time,message FROM logs WHERE job_id=? ORDER BY time", (job_id,))
        ]
    return result


def cancel(job_id: str) -> dict:
    """Cancel queued work immediately; running work is stopped by the worker."""
    store.job(job_id)
    with store.connection() as db:
        db.execute(
            "UPDATE jobs SET cancel_requested=1,"
            "status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END WHERE id=?",
            (job_id,),
        )
    return store.job(job_id)


def rerun(job_id: str) -> dict:
    previous = store.job(job_id)
    return store.enqueue(previous["kind"], previous["config"])


def claim() -> str | None:
    """Claim the oldest available job inside a transaction.

    The hosted worker's advisory lease provides exclusive ownership when the
    database adapter translates BEGIN IMMEDIATE to PostgreSQL's BEGIN.
    """
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT id FROM jobs WHERE status='queued' AND cancel_requested=0 ORDER BY created_at LIMIT 1"
        ).fetchone()
        if not row:
            return None
        db.execute("UPDATE jobs SET status='running',updated_at=? WHERE id=?", (store.now(), row[0]))
        return row[0]


def complete(job_id: str, result: dict, kind: str) -> None:
    summary = {"id": result["id"], "name": result.get("name", kind)}
    with store.connection() as db:
        db.execute(
            "UPDATE jobs SET status='completed',progress=1,message='Completed',result=?,updated_at=? WHERE id=?",
            (store.encode(summary), store.now(), job_id),
        )


def fail(job_id: str, error: BaseException, trace: str) -> None:
    status = "cancelled" if isinstance(error, store.Cancelled) else "failed"
    with store.connection() as db:
        db.execute(
            "UPDATE jobs SET status=?,message=?,error=?,updated_at=? WHERE id=?",
            (status, str(error), trace, store.now(), job_id),
        )
        db.execute("INSERT INTO logs VALUES (?,?,?)", (job_id, store.now(), trace))


def recover_interrupted() -> None:
    with store.connection() as db:
        db.execute(
            "UPDATE jobs SET status='interrupted',"
            "message='Worker restarted; rerun the saved configuration',updated_at=? WHERE status='running'",
            (store.now(),),
        )


def finish_cancelled(job_id: str) -> None:
    with store.connection() as db:
        db.execute(
            "UPDATE jobs SET status='cancelled',message='Cancelled',updated_at=? WHERE id=? AND status='running'",
            (store.now(), job_id),
        )


def fail_abandoned(job_id: str) -> None:
    """Record an unexpected child exit without overwriting a terminal status."""
    if store.job(job_id)["status"] == "running":
        with store.connection() as db:
            db.execute(
                "UPDATE jobs SET status='failed',message='Worker process exited before completion',"
                "updated_at=? WHERE id=?",
                (store.now(), job_id),
            )
