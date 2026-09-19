import tempfile
from pathlib import Path

import numpy as np

from backend import database, main


def test_evaluation_records_results_without_500(monkeypatch):
    original_db = database.DB_PATH
    tmp_db = Path(tempfile.mkdtemp()) / "evaluation-test.db"
    database.DB_PATH = tmp_db
    try:
        database.reset_runtime_database()
        person_id = database.create_person("Alice")
        database.add_sample(
            person_id=person_id,
            filename="a1.jpg",
            mime_type="image/jpeg",
            image_sha256="a1",
            image_phash="0",
            image_blob=b"A1",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            source="test",
            sample_role="reference",
            expected_label="Alice",
        )
        database.add_sample(
            person_id=person_id,
            filename="a2.jpg",
            mime_type="image/jpeg",
            image_sha256="a2",
            image_phash="0",
            image_blob=b"A2",
            embedding=np.array([0.99, 0.1], dtype=np.float32),
            source="test",
            sample_role="reference",
            expected_label="Alice",
        )
        database.add_sample(
            person_id=None,
            filename="unknown.jpg",
            mime_type="image/jpeg",
            image_sha256="unknown",
            image_phash="0",
            image_blob=b"U1",
            embedding=np.array([0.0, 1.0], dtype=np.float32),
            source="test",
            sample_role="unknown",
            expected_label="Unknown",
        )

        vectors = {
            b"A1": np.array([1.0, 0.0], dtype=np.float32),
            b"A2": np.array([0.99, 0.1], dtype=np.float32),
            b"U1": np.array([0.0, 1.0], dtype=np.float32),
        }
        monkeypatch.setattr(main, "load_image", lambda blob: blob)
        monkeypatch.setattr(main, "extract_single_face", lambda image: (image, {}))
        monkeypatch.setattr(main, "get_embedding", lambda face: vectors[face])

        summary = main._evaluate()

        assert summary["available"] is True
        assert summary["evaluated_samples"] == 3
        assert summary["accuracy"] == 100.0
        assert summary["run_id"] is not None
        assert all("status" in row for row in summary["rows"])
    finally:
        database.DB_PATH = original_db
