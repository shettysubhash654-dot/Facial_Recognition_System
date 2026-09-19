"""VisionID FastAPI application.

The web UI, API, evaluation engine, and database all live in this compact
project. Persistent application state is kept only in SQLite.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import database
from .face_engine import (
    best_match,
    canonicalize_image,
    cosine,
    detect_faces,
    extract_single_face,
    get_embedding,
    get_embeddings,
    find_identity_duplicate,
    hamming_distance,
    image_hash,
    load_image,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "frontend"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_ENROLL_IMAGES = 8
DUPLICATE_PHASH_DISTANCE = 8

app = FastAPI(
    title="VisionID — Face Recognition Command Center",
    version="4.1.0",
    description="Real-time face recognition with MTCNN, FaceNet embeddings, SQLite persistence, duplicate rejection, ephemeral demo mode, and evaluation.",
)


@app.middleware("http")
async def ephemeral_idle_guard(request, call_next):
    if database.maybe_reset_for_idle():
        # The next request sees a fresh SQLite database automatically.
        pass
    return await call_next(request)


class ThresholdPayload(BaseModel):
    value: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_upload(upload: UploadFile) -> bytes:
    data = upload.file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
    return data


def _find_duplicate(
    new_bytes: bytes,
    new_image,
    new_embedding: np.ndarray,
    requested_name: str,
    pending: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Reject exact/visual duplicates and cross-name identity duplicates.

    A new person name is blocked if the face embedding is similar to ANY existing
    identity's stored samples above the duplicate threshold. Additional photos
    for the SAME name are allowed (unless the image itself is a duplicate).
    """
    canonical_hash = sha256(new_bytes)
    exact = database.sha256_exists(canonical_hash)
    if exact:
        owner = exact.get("name") or "an existing record"
        return {
            "type": "image",
            "name": owner,
            "message": f"This image is already registered to '{owner}'.",
        }

    p_hash = image_hash(new_image)
    duplicate_threshold = database.get_duplicate_threshold()
    stored = database.all_embeddings()

    identity_hit = find_identity_duplicate(
        new_embedding, stored, duplicate_threshold, exclude_name=requested_name
    )
    if identity_hit:
        score = float(identity_hit["similarity"])
        name = identity_hit["name"]
        return {
            "type": "face",
            "name": name,
            "similarity": round(score, 4),
            "message": (
                f"Person already exists as '{name}'. A different photo of the same "
                f"face matched the existing identity at {score:.1%}. Please use the "
                "existing name instead of creating a new identity."
            ),
        }

    # Perceptual-hash fallback catches resized/re-encoded copies where the exact
    # SHA-256 changes. We only block when the hash is close AND face similarity
    # also supports the duplicate decision.
    for candidate in database.list_phash_candidates(p_hash, max_distance=DUPLICATE_PHASH_DISTANCE):
        candidate_name = (candidate.get("name") or "").strip()
        if not candidate_name or candidate_name.casefold() == requested_name.casefold():
            continue
        if hamming_distance(p_hash, candidate.get("image_phash", "")) > DUPLICATE_PHASH_DISTANCE:
            continue
        for sample in stored:
            if sample.get("id") != candidate.get("id"):
                continue
            score = cosine(new_embedding, sample["embedding"])
            if score >= duplicate_threshold:
                return {
                    "type": "image",
                    "name": candidate_name,
                    "similarity": round(score, 4),
                    "message": (
                        f"Person already exists as '{candidate_name}'. The uploaded image "
                        "appears to be the same face already registered."
                    ),
                }

    # Also compare against earlier images in the SAME request. This prevents
    # uploading two different photos of the same person under a new name in one
    # enrollment submission.
    for item in pending or []:
        pending_name = (item.get("name") or "").strip()
        if pending_name.casefold() == requested_name.casefold():
            continue
        score = cosine(new_embedding, item["embedding"])
        if score >= duplicate_threshold:
            return {
                "type": "face",
                "name": pending_name or "this enrollment request",
                "similarity": round(float(score), 4),
                "message": (
                    f"This face is already represented by '{pending_name}' in the current "
                    "enrollment. Duplicate identities are not allowed."
                ),
            }
    return None


def _normalized_faces_from_upload(upload: UploadFile):
    original = _read_upload(upload)
    image = load_image(original)
    canonical = canonicalize_image(image)
    canonical_image = load_image(canonical)
    face, meta = extract_single_face(canonical_image)
    if face is None:
        raise HTTPException(status_code=400, detail=meta["reason"])
    embedding = get_embedding(face)
    return original, canonical, canonical_image, meta, embedding


# ---------------------------------------------------------------------------
# System / dashboard
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "backend": "online",
        "database": "sqlite connected",
        "database_path": str(database.DB_PATH),
        "enrolled_people": database.person_count(),
        "threshold": database.get_threshold(),
        "duplicate_threshold": database.get_duplicate_threshold(),
        "model": "ready (lazy loaded)",
        "mode": "CPU",
        "storage_mode": "ephemeral" if database.EPHEMERAL_MODE else "persistent",
        "reset_on_startup": database.RESET_ON_STARTUP,
        "idle_reset_minutes": float(os.getenv("VISIONID_IDLE_RESET_MINUTES", str(database.DEFAULT_IDLE_RESET_MINUTES))),
    }


@app.get("/api/stats")
def stats():
    activity = database.activity_stats()
    return {
        "enrolled_people": database.person_count(),
        **activity,
        "threshold": database.get_threshold(),
        "duplicate_threshold": database.get_duplicate_threshold(),
    }


@app.get("/api/activity")
def activity(limit: int = Query(default=12, ge=1, le=200)):
    return {"activity": database.recent_activity(limit)}


# ---------------------------------------------------------------------------
# People / enrollment
# ---------------------------------------------------------------------------


@app.get("/api/people")
def people():
    return {"people": database.list_people()}


@app.delete("/api/people/{name}")
def delete_person(name: str):
    database.touch_interactive_activity()
    if not database.delete_person(name):
        raise HTTPException(status_code=404, detail="Person not found.")
    return {"message": f"Deleted {name}", "database": "updated"}


@app.post("/api/people")
async def enroll_person(
    name: str = Form(...),
    person_id: str = Form(""),
    description: str = Form(""),
    images: list[UploadFile] = File(...),
):
    database.touch_interactive_activity()
    cleaned_name = name.strip()
    if not cleaned_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    if len(images) < 1:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    if len(images) > MAX_ENROLL_IMAGES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_ENROLL_IMAGES} images per enrollment.")
    existing_person = database.get_person(cleaned_name)

    accepted: list[tuple[str, bytes, str, np.ndarray]] = []
    accepted_hashes: set[str] = set()
    pending: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for upload in images:
        try:
            original = _read_upload(upload)
            image = load_image(original)
            canonical = canonicalize_image(image)
            canonical_image = load_image(canonical)
            canonical_hash = sha256(canonical)
            if canonical_hash in accepted_hashes:
                skipped.append({"file": upload.filename or "image", "reason": "Duplicate image repeated in this enrollment request."})
                continue
            if database.sha256_exists(canonical_hash):
                skipped.append({"file": upload.filename or "image", "reason": "Exact image already exists in the database."})
                continue
            face, meta = extract_single_face(canonical_image)
            if face is None:
                skipped.append({"file": upload.filename or "image", "reason": meta["reason"]})
                continue
            embedding = get_embedding(face)
            duplicate = _find_duplicate(canonical, canonical_image, embedding, cleaned_name, pending=pending)
            if duplicate:
                skipped.append({"file": upload.filename or "image", "reason": duplicate["message"]})
                continue
            accepted.append((upload.filename or "capture.jpg", canonical, canonical_hash, embedding))
            accepted_hashes.add(canonical_hash)
            pending.append({"name": cleaned_name, "embedding": embedding})
        except HTTPException as exc:
            skipped.append({"file": upload.filename or "image", "reason": str(exc.detail)})
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            skipped.append({"file": upload.filename or "image", "reason": str(exc)})

    if not accepted:
        message = "No new image was enrolled."
        if skipped and all("already" in item["reason"].lower() or "duplicate" in item["reason"].lower() for item in skipped):
            message = skipped[0]["reason"]
        raise HTTPException(status_code=409, detail={"message": message, "skipped": skipped})

    try:
        if existing_person:
            person_db_id = int(existing_person["id"])
            database.touch_person(person_db_id, description=description or None, person_code=person_id or None)
        else:
            person_db_id = database.create_person(cleaned_name, person_id, description)
        for filename, canonical, canonical_hash, embedding in accepted:
            image = load_image(canonical)
            p_hash = image_hash(image)
            database.add_sample(
                person_id=person_db_id,
                filename=filename,
                mime_type="image/jpeg",
                image_sha256=canonical_hash,
                image_phash=p_hash,
                image_blob=canonical,
                embedding=embedding,
                source="enrollment",
                sample_role="reference",
                expected_label=cleaned_name,
            )
    except Exception as exc:
        # The database layer uses a transaction for each insert; surface a clean API error.
        raise HTTPException(status_code=500, detail=f"Enrollment could not be committed: {exc}") from exc

    return {
        "message": f"{'Updated' if existing_person else 'Enrolled'} {cleaned_name}",
        "person_id": person_db_id,
        "images_used": len(accepted),
        "images_skipped": skipped,
        "embedding_dimension": 512,
        "storage": "SQLite BLOBs",
    }


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------


@app.post("/api/recognize")
async def recognize(image: UploadFile = File(...), source: str = "upload"):
    database.touch_interactive_activity()
    raw = _read_upload(image)
    try:
        pil = load_image(raw)
        boxes, probabilities, faces = detect_faces(pil)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    threshold = database.get_threshold()
    if not boxes or faces is None:
        return {"image_width": pil.width, "image_height": pil.height, "threshold": threshold, "faces": []}

    embeddings = get_embeddings(faces)
    stored = database.all_embeddings()
    results = []
    for box, probability, embedding in zip(boxes, probabilities, embeddings):
        candidates = stored
        name, similarity, _ = best_match(embedding, candidates, threshold)
        known = name != "Unknown"
        database.log_activity(name, similarity, known, len(boxes), source)
        results.append({
            "box": [round(v, 1) for v in box],
            "name": name,
            "similarity": round(float(similarity), 4),
            "known": known,
            "detection_confidence": round(float(probability), 4),
        })

    return {"image_width": pil.width, "image_height": pil.height, "threshold": threshold, "faces": results}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@app.get("/api/settings/threshold")
def get_threshold():
    return {"threshold": database.get_threshold()}


@app.post("/api/settings/threshold")
def set_threshold(payload: ThresholdPayload):
    database.touch_interactive_activity()
    return {"threshold": database.set_threshold(payload.value)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate() -> dict[str, Any]:
    """Run a real leave-one-out evaluation directly from SQLite samples.

    Every run re-reads the current database. Adding/deleting enrollment data
    therefore changes the evaluation population and results automatically.
    """
    threshold = database.get_threshold()
    samples = database.samples_for_evaluation()
    all_known_embeddings = [s for s in samples if s.get("person_id") is not None and s["sample_role"] != "unknown"]
    rows: list[dict[str, Any]] = []
    known_total = 0
    unknown_total = 0
    correct = 0
    false_accepts = 0
    false_rejects = 0
    detection_failures = 0
    per_class: dict[str, dict[str, int]] = {}
    skipped = 0

    for sample in samples:
        expected = sample["expected_label"] or sample["name"] or "Unknown"
        is_unknown = expected.casefold() == "unknown" or sample["person_id"] is None

        try:
            image = load_image(sample["image_blob"])
            face, meta = extract_single_face(image)
            if face is None:
                detection_failures += 1
                status = "detection_failed"
                rows.append({"sample_id": sample["id"], "image": sample["filename"], "expected": expected, "predicted": "Unknown", "similarity": 0.0, "result": status, "status": status})
                continue
            query_embedding = get_embedding(face)
        except Exception as exc:
            detection_failures += 1
            rows.append({"sample_id": sample["id"], "image": sample["filename"], "expected": expected, "predicted": "Unknown", "similarity": 0.0, "result": f"error: {exc}", "status": f"error: {exc}"})
            continue

        if is_unknown:
            candidates = all_known_embeddings
        else:
            candidates = [x for x in all_known_embeddings if x["id"] != sample["id"]]
            # A person with one image is intentionally not self-tested.
            same_person = [x for x in candidates if x.get("name", "").casefold() == str(expected).casefold()]
            if not same_person:
                skipped += 1
                rows.append({"sample_id": sample["id"], "image": sample["filename"], "expected": expected, "predicted": "Not evaluated", "similarity": 0.0, "result": "not_evaluable_single_reference", "status": "not_evaluable_single_reference"})
                continue

        predicted, similarity, _ = best_match(query_embedding, candidates, threshold)
        if is_unknown:
            unknown_total += 1
            is_correct = predicted == "Unknown"
            if is_correct:
                correct += 1
            else:
                false_accepts += 1
        else:
            known_total += 1
            is_correct = predicted.casefold() == str(expected).casefold()
            if is_correct:
                correct += 1
            elif predicted == "Unknown":
                false_rejects += 1

        cls = str(expected)
        entry = per_class.setdefault(cls, {"total": 0, "correct": 0})
        entry["total"] += 1
        entry["correct"] += int(is_correct)
        rows.append({"sample_id": sample["id"], "image": sample["filename"], "expected": expected, "predicted": predicted, "similarity": round(float(similarity), 4), "result": "correct" if is_correct else "incorrect", "status": "correct" if is_correct else "incorrect"})

    evaluated = known_total + unknown_total
    accuracy = (correct / evaluated * 100) if evaluated else 0.0
    far = (false_accepts / unknown_total * 100) if unknown_total else 0.0
    frr = (false_rejects / known_total * 100) if known_total else 0.0
    dfr = (detection_failures / len(samples) * 100) if samples else 0.0
    summary = {
        "available": bool(samples),
        "samples": len(samples),
        "evaluated_samples": evaluated,
        "evaluated_known": known_total,
        "evaluated_unknown": unknown_total,
        "skipped_single_reference": skipped,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "false_accept_rate": round(far, 2),
        "false_reject_rate": round(frr, 2),
        "detection_failure_rate": round(dfr, 2),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "detection_failures": detection_failures,
        "threshold": threshold,
        "per_class": {
            name: {"total": values["total"], "correct": values["correct"], "accuracy": round(values["correct"] / values["total"] * 100, 1) if values["total"] else 0.0}
            for name, values in sorted(per_class.items())
        },
        "rows": rows[:500],
        "storage": "SQLite face_samples table",
        "method": "Leave-one-out for known identities + full enrolled set for Unknown samples.",
    }
    if samples:
        run_id = database.record_evaluation_run(summary, rows)
        summary["run_id"] = run_id
    return summary


@app.get("/api/evaluation/status")
def evaluation_status():
    samples = database.samples_for_evaluation()
    return {
        "database_samples": len(samples),
        "known_samples": sum(1 for s in samples if s["person_id"] is not None),
        "unknown_samples": sum(1 for s in samples if s["person_id"] is None),
        "people": database.person_count(),
        "message": "Evaluation reads the current SQLite database every time it runs.",
    }


@app.post("/api/evaluate")
def evaluate():
    database.touch_interactive_activity()
    return _evaluate()


@app.get("/api/evaluation/latest")
def latest_evaluation():
    return database.latest_evaluation() or {"available": False, "message": "No evaluation run yet."}


# ---------------------------------------------------------------------------
# Demo dataset — optional, runtime-only
# ---------------------------------------------------------------------------

# Source records were selected from Wikimedia Commons pages. The app stores
# the downloaded bytes in SQLite; nothing is written to a dataset directory.
DEMO_SOURCES = [
    {"name": "Narendra Modi", "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Official_portrait_of_Narendra_Modi_2022.jpg", "attribution": "Wikimedia Commons / Government of India", "license": "Government Open Data License - India"},
    {"name": "Droupadi Murmu", "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Droupadi_Murmu_official_portrait.jpg", "attribution": "Wikimedia Commons / President's Secretariat", "license": "Government Open Data License - India"},
    {"name": "Mamata Banerjee", "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Official_portrait_of_Mamata_Banerjee.jpg", "attribution": "Wikimedia Commons / Chief Minister's Office, Government of West Bengal", "license": "Government Open Data License - India"},
    {"name": "Yogi Adityanath", "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Yogi_Adityanath.jpg", "attribution": "Wikimedia Commons / Prime Minister's Office", "license": "See source page"},
    {"name": "M.K. Stalin", "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/MK_Stalin.jpg", "attribution": "Wikimedia Commons / Press Information Bureau", "license": "See source page"},
    {"name": "Pinarayi Vijayan", "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Pinarayi_Vijayan.jpg", "attribution": "Wikimedia Commons / bodhicommons", "license": "CC BY-SA 3.0"},
    {"name": "Himanta Biswa Sarma", "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Himanta_Biswa_Sarma,_2024.jpg", "attribution": "Wikimedia Commons / Prime Minister's Office, India", "license": "Government Open Data License - India"},
    {"name": "N. Chandrababu Naidu", "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/N._Chandrababu_Naidu.jpg", "attribution": "Wikimedia Commons / Naralokesh", "license": "Public domain"},
    {"name": "Siddaramaiah", "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/CM_Siddaramaiah_in_July_2025.png", "attribution": "Wikimedia Commons / MandukRao", "license": "CC0 1.0"},
    {"name": "Revanth Reddy", "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Portrait_of_Telangana_CM_Revanth_Reddy.png", "attribution": "Wikimedia Commons / PMO India", "license": "See source page"},
]

database.seed_demo_sources(DEMO_SOURCES)


@app.get("/api/demo/status")
def demo_status():
    return {"sources": database.demo_sources()}


@app.post("/api/demo/load")
def load_demo():
    database.touch_interactive_activity()
    """Load ten optional demo identities into SQLite.

    The downloaded images are converted to JPEG and inserted into SQLite. The
    route is intentionally explicit — opening the project never silently
    downloads or stores public figures.
    """
    sources = database.demo_sources()
    added = []
    skipped = []
    for source in sources[:10]:
        if source["loaded"]:
            skipped.append({"name": source["name"], "reason": "already loaded"})
            continue
        if database.get_person(source["name"]):
            skipped.append({"name": source["name"], "reason": "person already exists"})
            continue
        try:
            req = Request(source["source_url"], headers={"User-Agent": "VisionID-Demo/1.0"})
            with urlopen(req, timeout=20) as response:
                raw = response.read(MAX_IMAGE_BYTES + 1)
            if len(raw) > MAX_IMAGE_BYTES:
                raise ValueError("source image exceeded 10 MB")
            image = load_image(raw)
            canonical = canonicalize_image(image)
            canonical_img = load_image(canonical)
            face, meta = extract_single_face(canonical_img)
            if face is None:
                raise ValueError(meta["reason"])
            embedding = get_embedding(face)
            person_db_id = database.create_person(source["name"], description="Demo identity — public-source image")
            database.add_sample(
                person_id=person_db_id,
                filename=f"demo-{hashlib.sha1(source['name'].encode()).hexdigest()[:8]}.jpg",
                mime_type="image/jpeg",
                image_sha256=sha256(canonical),
                image_phash=image_hash(canonical_img),
                image_blob=canonical,
                embedding=embedding,
                source="demo",
                sample_role="demo_reference",
                expected_label=source["name"],
            )
            # Create a mild validation variant in memory and store it directly in SQLite.
            arr = np.asarray(canonical_img)
            variant = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            variant = cv2.convertScaleAbs(variant, alpha=1.04, beta=7)
            variant = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
            validation_img = load_image(canonicalize_image(Image.fromarray(variant)))
            v_face, v_meta = extract_single_face(validation_img)
            if v_face is not None:
                v_embedding = get_embedding(v_face)
                v_bytes = canonicalize_image(validation_img)
                database.add_sample(
                    person_id=person_db_id,
                    filename=f"demo-validation-{hashlib.sha1(source['name'].encode()).hexdigest()[:8]}.jpg",
                    mime_type="image/jpeg",
                    image_sha256=sha256(v_bytes),
                    image_phash=image_hash(validation_img),
                    image_blob=v_bytes,
                    embedding=v_embedding,
                    source="demo",
                    sample_role="demo_validation",
                    expected_label=source["name"],
                )
            with database.get_conn() as conn:
                conn.execute("UPDATE demo_sources SET loaded=1 WHERE name=?", (source["name"],))
            added.append(source["name"])
        except Exception as exc:
            skipped.append({"name": source["name"], "reason": str(exc)})
    return {"added": added, "skipped": skipped, "people": database.list_people()}


# ---------------------------------------------------------------------------
# Ephemeral demo reset
# ---------------------------------------------------------------------------

@app.get("/api/demo/mode")
def demo_mode():
    return {
        "ephemeral": database.EPHEMERAL_MODE,
        "reset_on_startup": database.RESET_ON_STARTUP,
        "idle_reset_minutes": float(os.getenv("VISIONID_IDLE_RESET_MINUTES", str(database.DEFAULT_IDLE_RESET_MINUTES))),
        "database_path": str(database.DB_PATH),
        "message": (
            "Demo data is temporary and will reset when the hosted process restarts."
            if database.EPHEMERAL_MODE
            else "Database is persistent on this machine/server."
        ),
    }

@app.post("/api/demo/reset")
def reset_demo():
    """Explicit reset is available only in ephemeral demo mode."""
    if not database.EPHEMERAL_MODE:
        raise HTTPException(status_code=403, detail="Manual reset is available only in ephemeral demo mode.")
    database.reset_runtime_database()
    database.touch_interactive_activity()
    return {"message": "Demo storage reset. The next request starts with an empty database."}


# ---------------------------------------------------------------------------
# Model information
# ---------------------------------------------------------------------------


@app.get("/api/model-info")
def model_info():
    return {
        "pipeline": [
            {"step": "Face detection", "detail": "MTCNN locates faces and returns confidence-scored bounding boxes."},
            {"step": "Alignment / crop", "detail": "facenet-pytorch extracts a standardized 160×160 face tensor."},
            {"step": "Face embedding", "detail": "InceptionResnetV1 pretrained on VGGFace2 produces a normalized 512-D vector."},
            {"step": "Similarity", "detail": "Cosine similarity compares the query embedding with stored SQLite embeddings."},
            {"step": "Unknown rejection", "detail": "The highest score must be at least the configured 0.65 threshold to be accepted."},
            {"step": "Duplicate protection", "detail": "Exact image hash + perceptual hash candidate search + face-embedding similarity prevent duplicate identities."},
        ],
        "runtime": {
            "recognition_model": "InceptionResnetV1 (VGGFace2)",
            "detector": "MTCNN",
            "embedding_dimension": 512,
            "execution_provider": "CPU",
            "similarity_metric": "cosine",
            "threshold": database.get_threshold(),
            "duplicate_threshold": database.get_duplicate_threshold(),
            "database": str(database.DB_PATH),
        },
        "storage": "SQLite only: people, face samples/images, embeddings, activity, settings, and evaluation runs are persisted in face_recognition.db.",
        "limitations": "The system does not implement liveness detection. Validate thresholds and permissions before operational use.",
    }


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")
