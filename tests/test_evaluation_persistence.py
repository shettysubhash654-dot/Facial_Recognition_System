import numpy as np


def test_evaluation_run_accepts_result_field_without_status(tmp_path, monkeypatch):
    from backend import database

    db_path = tmp_path / "eval.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()

    summary = {
        "threshold": 0.65,
        "samples": 1,
        "evaluated_known": 1,
        "evaluated_unknown": 0,
        "correct": 1,
        "false_accepts": 0,
        "false_rejects": 0,
        "detection_failures": 0,
        "accuracy": 100.0,
        "false_accept_rate": 0.0,
        "false_reject_rate": 0.0,
        "detection_failure_rate": 0.0,
    }
    rows = [{
        "sample_id": None,
        "expected": "Alice",
        "predicted": "Alice",
        "similarity": 0.91,
        "result": "correct",
    }]

    run_id = database.record_evaluation_run(summary, rows)
    assert run_id == 1
    latest = database.latest_evaluation()
    assert latest is not None
    assert latest["results"][0]["status"] == "correct"
