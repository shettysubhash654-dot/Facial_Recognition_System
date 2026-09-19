"""Single-file SQLite persistence for VisionID.

All application state lives in one SQLite database:
    backend/data/face_recognition.db

There is no JSON persistence and no browser localStorage for application data.
Face images and embeddings are stored directly in SQLite BLOB columns.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

EPHEMERAL_MODE = os.getenv("VISIONID_EPHEMERAL", "0").strip().lower() in {"1", "true", "yes", "on"}
RESET_ON_STARTUP = os.getenv("VISIONID_RESET_ON_STARTUP", "1" if EPHEMERAL_MODE else "0").strip().lower() in {"1", "true", "yes", "on"}

if os.getenv("VISIONID_DATA_DIR"):
    DATA_DIR = Path(os.environ["VISIONID_DATA_DIR"]).expanduser().resolve()
elif EPHEMERAL_MODE:
    DATA_DIR = Path(tempfile.gettempdir()) / "visionid-runtime"
else:
    DATA_DIR = Path(__file__).resolve().parent / "data"

DB_PATH = DATA_DIR / "face_recognition.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_THRESHOLD = 0.65
# Duplicate protection is intentionally stricter than recognition. It blocks a
# new name when the incoming face is sufficiently similar to any existing identity.
DEFAULT_DUPLICATE_THRESHOLD = 0.80
ACTIVITY_LIMIT = 500
DEFAULT_IDLE_RESET_MINUTES = 120

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
    person_id TEXT DEFAULT '',
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL DEFAULT 'image/jpeg',
    image_sha256 TEXT NOT NULL UNIQUE,
    image_phash TEXT NOT NULL,
    image_blob BLOB NOT NULL,
    embedding_blob BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    source TEXT NOT NULL DEFAULT 'enrollment',
    sample_role TEXT NOT NULL DEFAULT 'reference',
    expected_label TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_face_samples_person ON face_samples(person_id);
CREATE INDEX IF NOT EXISTS idx_face_samples_phash ON face_samples(image_phash);

CREATE TABLE IF NOT EXISTS activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    name TEXT NOT NULL,
    similarity REAL NOT NULL,
    known INTEGER NOT NULL,
    face_count INTEGER NOT NULL DEFAULT 1,
    source TEXT NOT NULL DEFAULT 'upload'
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    threshold REAL NOT NULL,
    samples INTEGER NOT NULL,
    evaluated_known INTEGER NOT NULL,
    evaluated_unknown INTEGER NOT NULL,
    correct INTEGER NOT NULL,
    false_accepts INTEGER NOT NULL,
    false_rejects INTEGER NOT NULL,
    detection_failures INTEGER NOT NULL,
    accuracy REAL NOT NULL,
    false_accept_rate REAL NOT NULL,
    false_reject_rate REAL NOT NULL,
    detection_failure_rate REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    sample_id INTEGER,
    expected_label TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    similarity REAL NOT NULL,
    status TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    FOREIGN KEY(sample_id) REFERENCES face_samples(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS demo_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    source_url TEXT NOT NULL,
    attribution TEXT NOT NULL DEFAULT '',
    license_text TEXT NOT NULL DEFAULT '',
    loaded INTEGER NOT NULL DEFAULT 0
);

INSERT OR IGNORE INTO settings(key, value) VALUES ('threshold', '0.65');
INSERT OR IGNORE INTO settings(key, value) VALUES ('duplicate_threshold', '0.80');
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def touch_interactive_activity() -> None:
    """Update the timestamp used by ephemeral-demo idle expiry."""
    if not EPHEMERAL_MODE:
        return
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO settings(key,value) VALUES('last_interactive_at',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (now_iso(),),
        )


def maybe_reset_for_idle() -> bool:
    """Reset the ephemeral DB after a period without interactive activity."""
    if not EPHEMERAL_MODE:
        return False
    try:
        idle_minutes = float(os.getenv("VISIONID_IDLE_RESET_MINUTES", str(DEFAULT_IDLE_RESET_MINUTES)))
    except ValueError:
        idle_minutes = float(DEFAULT_IDLE_RESET_MINUTES)
    if idle_minutes <= 0 or not DB_PATH.exists():
        return False

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        row = conn.execute("SELECT value FROM settings WHERE key='last_interactive_at'").fetchone()
    if not row:
        touch_interactive_activity()
        return False

    from datetime import datetime
    try:
        last = datetime.fromisoformat(str(row[0]))
        elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 60.0
    except Exception:
        elapsed = 0.0

    if elapsed >= idle_minutes:
        reset_runtime_database()
        touch_interactive_activity()
        return True
    return False


def reset_runtime_database() -> None:
    """Delete and recreate the current SQLite database in-place."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def init_db() -> None:
    if RESET_ON_STARTUP and DB_PATH.exists():
        try:
            DB_PATH.unlink()
        except PermissionError as exc:
            raise RuntimeError(f"Could not reset ephemeral database at {DB_PATH}: {exc}") from exc
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def _embedding_blob(embedding: np.ndarray) -> bytes:
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return arr.tobytes()


def _embedding_from_blob(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32).copy()
    if arr.size != dim:
        raise ValueError(f"Invalid embedding size: expected {dim}, got {arr.size}")
    norm = np.linalg.norm(arr)
    return arr / max(norm, 1e-12)


def get_threshold() -> float:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key='threshold'").fetchone()
        return float(row[0]) if row else DEFAULT_THRESHOLD


def set_threshold(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO settings(key,value) VALUES('threshold',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(value),),
        )
    return value


def get_duplicate_threshold() -> float:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key='duplicate_threshold'").fetchone()
        return float(row[0]) if row else DEFAULT_DUPLICATE_THRESHOLD


def get_person(name: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM people WHERE name=?", (name.strip(),)).fetchone()


def list_people() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.name, p.person_id, p.description, p.created_at, p.updated_at,
                   COUNT(s.id) AS num_images
            FROM people p
            LEFT JOIN face_samples s ON s.person_id = p.id
            GROUP BY p.id
            ORDER BY p.name COLLATE NOCASE
            """
        ).fetchall()
    return [dict(row) for row in rows]


def person_count() -> int:
    with get_conn() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM people").fetchone()[0])


def create_person(name: str, person_id: str = "", description: str = "") -> int:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Name cannot be empty")
    stamp = now_iso()
    with get_conn() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO people(name,person_id,description,created_at,updated_at) VALUES(?,?,?,?,?)",
                (cleaned, person_id.strip(), description.strip(), stamp, stamp),
            )
        except sqlite3.IntegrityError as exc:
            if "people.name" in str(exc).lower() or "unique" in str(exc).lower():
                raise ValueError(f"Person '{cleaned}' already exists") from exc
            raise
        return int(cur.lastrowid)


def touch_person(person_id: int, description: str | None = None, person_code: str | None = None) -> None:
    updates = ["updated_at=?"]
    params: list[Any] = [now_iso()]
    if description is not None:
        updates.append("description=?")
        params.append(description.strip())
    if person_code is not None:
        updates.append("person_id=?")
        params.append(person_code.strip())
    params.append(person_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE people SET {', '.join(updates)} WHERE id=?", params)


def delete_person(name: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM people WHERE name=?", (name.strip(),))
        return cur.rowcount > 0


def add_sample(
    *,
    person_id: int | None,
    filename: str,
    mime_type: str,
    image_sha256: str,
    image_phash: str,
    image_blob: bytes,
    embedding: np.ndarray,
    source: str,
    sample_role: str,
    expected_label: str | None,
) -> int:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO face_samples(
                person_id,filename,mime_type,image_sha256,image_phash,image_blob,
                embedding_blob,embedding_dim,source,sample_role,expected_label,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                person_id,
                filename,
                mime_type,
                image_sha256,
                image_phash,
                sqlite3.Binary(image_blob),
                sqlite3.Binary(_embedding_blob(vector)),
                int(vector.size),
                source,
                sample_role,
                expected_label,
                now_iso(),
            ),
        )
        if person_id is not None:
            conn.execute("UPDATE people SET updated_at=? WHERE id=?", (now_iso(), person_id))
        return int(cur.lastrowid)


def sha256_exists(image_sha256: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT s.id, s.filename, p.name
            FROM face_samples s
            LEFT JOIN people p ON p.id=s.person_id
            WHERE s.image_sha256=?
            """,
            (image_sha256,),
        ).fetchone()
    return dict(row) if row else None


def list_phash_candidates(image_phash: str, max_distance: int = 6) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT s.id,s.person_id,p.name,s.image_phash
            FROM face_samples s
            LEFT JOIN people p ON p.id=s.person_id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def all_embeddings() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT s.id,s.person_id,p.name,s.embedding_blob,s.embedding_dim,s.sample_role,s.expected_label
            FROM face_samples s
            LEFT JOIN people p ON p.id=s.person_id
            WHERE s.embedding_blob IS NOT NULL
            ORDER BY s.id
            """
        ).fetchall()
    out = []
    for row in rows:
        out.append({
            "id": row["id"],
            "person_id": row["person_id"],
            "name": row["name"],
            "embedding": _embedding_from_blob(row["embedding_blob"], row["embedding_dim"]),
            "sample_role": row["sample_role"],
            "expected_label": row["expected_label"],
        })
    return out


def samples_for_evaluation() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT s.id,s.person_id,p.name,s.filename,s.image_blob,s.embedding_blob,s.embedding_dim,
                   s.sample_role,s.expected_label,s.source
            FROM face_samples s
            LEFT JOIN people p ON p.id=s.person_id
            WHERE s.sample_role IN ('reference','demo_reference','demo_validation','unknown')
            ORDER BY s.id
            """
        ).fetchall()
    result=[]
    for row in rows:
        result.append({
            "id": row["id"], "person_id": row["person_id"], "name": row["name"],
            "filename": row["filename"], "image_blob": row["image_blob"],
            "embedding": _embedding_from_blob(row["embedding_blob"], row["embedding_dim"]),
            "sample_role": row["sample_role"], "expected_label": row["expected_label"],
            "source": row["source"],
        })
    return result


def log_activity(name: str, similarity: float, known: bool, face_count: int, source: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO activity(timestamp,name,similarity,known,face_count,source) VALUES(?,?,?,?,?,?)",
            (now_iso(), name, float(similarity), int(bool(known)), int(face_count), source),
        )
        conn.execute(
            "DELETE FROM activity WHERE id NOT IN (SELECT id FROM activity ORDER BY id DESC LIMIT ?)",
            (ACTIVITY_LIMIT,),
        )


def recent_activity(limit: int = 12) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT timestamp,name,similarity,known,face_count,source FROM activity ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def activity_stats() -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) total, COALESCE(SUM(known),0) known, "
            "COALESCE(SUM(CASE WHEN known=0 THEN 1 ELSE 0 END),0) unknown FROM activity"
        ).fetchone()
    total, known, unknown = int(row[0]), int(row[1]), int(row[2])
    return {
        "recognition_attempts": total,
        "successful_matches": known,
        "unknown_faces": unknown,
        "acceptance_rate": round((known / total) * 100, 1) if total else 0.0,
    }


def record_evaluation_run(summary: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO evaluation_runs(
                created_at,threshold,samples,evaluated_known,evaluated_unknown,correct,
                false_accepts,false_rejects,detection_failures,accuracy,false_accept_rate,
                false_reject_rate,detection_failure_rate
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                now_iso(), summary["threshold"], summary["samples"], summary["evaluated_known"],
                summary["evaluated_unknown"], summary["correct"], summary["false_accepts"],
                summary["false_rejects"], summary["detection_failures"], summary["accuracy"],
                summary["false_accept_rate"], summary["false_reject_rate"], summary["detection_failure_rate"],
            ),
        )
        run_id = int(cur.lastrowid)
        for row in rows:
            conn.execute(
                """
                INSERT INTO evaluation_results(run_id,sample_id,expected_label,predicted_label,similarity,status)
                VALUES(?,?,?,?,?,?)
                """,
                (run_id, row.get("sample_id"), row["expected"], row["predicted"], row["similarity"], row.get("status", row.get("result", "unknown"))),
            )
        return run_id


def latest_evaluation() -> dict[str, Any] | None:
    with get_conn() as conn:
        run = conn.execute("SELECT * FROM evaluation_runs ORDER BY id DESC LIMIT 1").fetchone()
        if not run:
            return None
        rows = conn.execute(
            "SELECT expected_label AS expected,predicted_label AS predicted,similarity,status "
            "FROM evaluation_results WHERE run_id=? ORDER BY id", (run["id"],)
        ).fetchall()
    out = dict(run)
    out["results"] = [dict(r) for r in rows]
    return out


def seed_demo_sources(sources: list[dict[str, str]]) -> None:
    with get_conn() as conn:
        for item in sources:
            conn.execute(
                "INSERT OR IGNORE INTO demo_sources(name,source_url,attribution,license_text) VALUES(?,?,?,?)",
                (item["name"], item["url"], item.get("attribution", ""), item.get("license", "")),
            )


def demo_sources() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT name,source_url,attribution,license_text,loaded FROM demo_sources ORDER BY id").fetchall()
    return [dict(r) for r in rows]


init_db()
