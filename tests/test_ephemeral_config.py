def test_duplicate_threshold_documented_as_strict():
    from backend import database
    assert 0.65 < database.DEFAULT_DUPLICATE_THRESHOLD <= 0.90


def test_database_path_is_sqlite_file():
    from backend import database
    assert database.DB_PATH.name == "face_recognition.db"
    assert database.DB_PATH.suffix == ".db"
