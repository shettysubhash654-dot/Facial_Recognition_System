import numpy as np
from backend.face_engine import cosine, find_identity_duplicate


def unit(v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_same_person_different_photo_hits_duplicate_threshold():
    stored = [
        {"id": 1, "name": "A", "embedding": unit([1.0, 0.0, 0.0, 0.0])},
        {"id": 2, "name": "B", "embedding": unit([0.0, 1.0, 0.0, 0.0])},
    ]
    query = unit([0.98, 0.20, 0.0, 0.0])
    hit = find_identity_duplicate(query, stored, 0.82, exclude_name="C")
    assert hit is not None
    assert hit["name"] == "A"
    assert hit["similarity"] >= 0.82


def test_different_person_does_not_hit_duplicate():
    stored = [{"id": 1, "name": "A", "embedding": unit([1.0, 0.0, 0.0, 0.0])}]
    query = unit([0.0, 1.0, 0.0, 0.0])
    assert find_identity_duplicate(query, stored, 0.82, exclude_name="B") is None


def test_same_name_is_excluded_from_cross_name_duplicate_check():
    stored = [{"id": 1, "name": "A", "embedding": unit([1.0, 0.0, 0.0, 0.0])}]
    query = unit([1.0, 0.0, 0.0, 0.0])
    assert find_identity_duplicate(query, stored, 0.82, exclude_name="A") is None
