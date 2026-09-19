"""Face detection, embeddings, matching, duplicate protection, and image tools."""
from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

_MTCNN: Any | None = None
_MODEL: Any | None = None
_DEVICE: Any | None = None


def load_image(data: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            rgb = image.convert("RGB")
    except Exception as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    # OpenCV is used to validate that the decoded pixels are usable by the vision pipeline.
    frame = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    if frame.size == 0:
        raise ValueError("The image contains no pixels.")
    return rgb


def canonicalize_image(image: Image.Image) -> bytes:
    image = image.convert("RGB").copy()
    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_hash(image: Image.Image) -> str:
    """A compact perceptual hash used only as a duplicate candidate filter."""
    gray = np.asarray(image.convert("L").resize((16, 16), Image.Resampling.LANCZOS), dtype=np.float32)
    mean = float(gray.mean())
    bits = (gray >= mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:064x}"


def hamming_distance(a: str, b: str) -> int:
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except ValueError:
        return 999


def get_detector():
    global _MTCNN
    if _MTCNN is None:
        try:
            from facenet_pytorch import MTCNN
        except ImportError as exc:
            raise RuntimeError("facenet-pytorch is missing. Install requirements.txt first.") from exc
        _MTCNN = MTCNN(
            keep_all=True,
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device="cpu",
        )
    return _MTCNN


def get_model():
    global _MODEL, _DEVICE
    if _MODEL is None:
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
        except ImportError as exc:
            raise RuntimeError("torch/facenet-pytorch is missing. Install requirements.txt first.") from exc
        _DEVICE = torch.device("cpu")
        _MODEL = InceptionResnetV1(pretrained="vggface2").eval().to(_DEVICE)
    return _MODEL


def detect_faces(image: Image.Image):
    detector = get_detector()
    boxes, probabilities = detector.detect(image)
    if boxes is None or probabilities is None:
        return [], [], None
    valid_boxes = []
    valid_probs = []
    for box, probability in zip(boxes, probabilities):
        if probability is None:
            continue
        valid_boxes.append([float(v) for v in box])
        valid_probs.append(float(probability))
    if not valid_boxes:
        return [], [], None
    faces = detector.extract(image, np.asarray(valid_boxes, dtype=np.float32), save_path=None)
    return valid_boxes, valid_probs, faces


def extract_single_face(image: Image.Image):
    boxes, probs, faces = detect_faces(image)
    if not boxes:
        return None, {"reason": "No face detected.", "face_count": 0}
    if len(boxes) != 1:
        return None, {"reason": f"Expected exactly one face; detected {len(boxes)}.", "face_count": len(boxes)}
    face = faces[0] if getattr(faces, "ndim", 0) == 4 else faces
    return face, {"reason": "ok", "face_count": 1, "detection_confidence": probs[0], "box": boxes[0]}


def get_embeddings(faces) -> np.ndarray:
    import torch

    if faces is None:
        raise ValueError("No face tensors supplied.")
    if faces.ndim == 3:
        faces = faces.unsqueeze(0)
    model = get_model()
    with torch.no_grad():
        output = model(faces.to(_DEVICE))
    vectors = output.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def get_embedding(face) -> np.ndarray:
    return get_embeddings(face)[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)



def find_identity_duplicate(
    embedding: np.ndarray,
    samples: list[dict[str, Any]],
    threshold: float,
    exclude_name: str | None = None,
) -> dict[str, Any] | None:
    """Find whether an embedding already belongs to another enrolled identity.

    The comparison is against every stored face sample, not just the original
    enrollment image. The highest similarity for a person is effectively used
    because every sample is examined.
    """
    excluded = (exclude_name or "").strip().casefold()
    best: dict[str, Any] | None = None
    for sample in samples:
        name = (sample.get("name") or "").strip()
        if not name or name.casefold() == excluded:
            continue
        score = cosine(embedding, sample["embedding"])
        if score < threshold:
            continue
        if best is None or score > float(best["similarity"]):
            best = {
                "name": name,
                "similarity": float(score),
                "sample_id": sample.get("id"),
            }
    return best


def best_match(embedding: np.ndarray, samples: list[dict[str, Any]], threshold: float) -> tuple[str, float, int | None]:
    best_name = "Unknown"
    best_score = -1.0
    best_sample_id = None
    for sample in samples:
        score = cosine(embedding, sample["embedding"])
        if score > best_score:
            best_score = score
            best_name = sample["name"] or "Unknown"
            best_sample_id = sample["id"]
    if best_score < threshold:
        return "Unknown", max(0.0, best_score), best_sample_id
    return best_name, best_score, best_sample_id
