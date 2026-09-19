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
