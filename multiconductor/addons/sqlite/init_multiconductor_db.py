import sqlite3
from pathlib import Path


def init_db(db_path: Path, schema_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        with schema_path.open("r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    schema_path = here / "multiconductor_schema.sql"
    db_path = here / "multiconductor.db"
    init_db(db_path, schema_path)
    print(f"Initialized SQLite DB at {db_path}")
