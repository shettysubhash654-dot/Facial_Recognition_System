# VisionID — Face Recognition Command Center

A FastAPI + browser-webcam face-recognition system using MTCNN, InceptionResnetV1/VGGFace2, cosine similarity, configurable Unknown rejection, SQLite persistence, duplicate-identity protection, and a live Evaluation Lab.

The URL you get inside the codespace =https://studious-chainsaw-x5qr96jv9q7xc964w-8000.app.github.dev/

## Storage architecture

This version intentionally removes JSON persistence and browser `localStorage` for application data.

All persistent state is stored in one SQLite database:

```text
backend/
└── data/
    └── face_recognition.db
```

The database stores:

- `people` — identities and metadata
- `face_samples` — canonical JPEG bytes, SHA-256, perceptual hash, face embeddings, sample role, and expected label
- `activity` — recognition events
- `settings` — matching threshold and duplicate threshold
- `evaluation_runs` — metric summaries for each run
- `evaluation_results` — per-sample evaluation decisions
- `demo_sources` — optional demo dataset source registry

There are no `people.json`, `activity.json`, `settings.json`, `embeddings.json`, or `evaluation/dataset/` runtime stores.

## Database path

The connection is deliberately explicit in `backend/database.py`:

```python
DB_PATH = Path(__file__).resolve().parent / "data" / "face_recognition.db"
```

and every database operation uses `sqlite3.connect(DB_PATH)`.

This means the application's persistence is server-side SQLite. When the application is hosted on a persistent server/volume, the database stays with the deployed application rather than in the browser.

## Duplicate protection

The enrollment pipeline rejects duplicates using three layers and checks the incoming face against **all previous face samples** in SQLite, not only the first photo:

1. **Exact duplicate:** SHA-256 of the canonical JPEG is unique in SQLite.
2. **Visual duplicate candidate:** a perceptual hash catches common re-encoded/resized copies.
3. **Identity duplicate:** a face-embedding cosine similarity of `>= 0.80` blocks the same face from being registered under another name.

A person can add new reference images under the same existing name. A new/different name is blocked when the incoming face matches an existing identity at or above the duplicate threshold. The same photo, including re-encoded copies, is also blocked.

## Recognition

```text
Browser webcam / uploaded image
          ↓
      MTCNN detection
          ↓
     Face extraction
          ↓
 InceptionResnetV1
          ↓
  512-D normalized vector
          ↓
   Cosine similarity
          ↓
 threshold = 0.65
          ↓
     Known / Unknown
```

## Evaluation Lab

Evaluation does **not** read a local folder. Each run reads the current contents of `face_recognition.db`.

For known identities, the evaluator uses **leave-one-out matching** so an image is not scored against its own embedding. A person with only one reference image is marked `not_evaluable_single_reference` instead of being counted as an artificial perfect match.

Unknown evaluation samples can also live directly in `face_samples` with `person_id=NULL` and `sample_role='unknown'`.

Every evaluation run stores:

- Accuracy
- False accept rate (FAR)
- False reject rate (FRR)
- Detection failure rate
- Per-identity accuracy
- Per-sample decisions

The UI also renders a visual metric chart after each run.

## Optional demo dataset

The Evaluation Lab has a **Load 10 demo identities** button. The application downloads the configured public-source images at runtime, converts them to JPEG, computes embeddings, and stores the resulting image bytes/embeddings directly in SQLite. Nothing is written to an evaluation folder.

The source pages/attribution information are stored in the `demo_sources` table.

## Run on Windows

### Recommended

Double-click:

```text
run_windows.bat
```

Or run from Command Prompt:

```cmd
cd C:\path\to\visionid-sqlite
run_windows.bat
```

The script creates `.venv`, installs the pinned CPU build, initializes SQLite automatically, and starts FastAPI.

Open:

```text
http://127.0.0.1:8000/
```

Swagger/API documentation:

```text
http://127.0.0.1:8000/docs
```

## Core UI

- **Dashboard** — current identity count, recognition activity, threshold
- **Recognize** — image upload and live browser webcam recognition
- **Enroll Identity** — image upload and webcam capture
- **People Directory** — list and delete identities
- **Evaluation Lab** — run database-backed evaluation and charts
- **Model Notes** — explain the inference pipeline

## Public demo mode (ephemeral storage)

For a public demo link where you do NOT want enrollment data to persist, run with:

```text
VISIONID_EPHEMERAL=1
VISIONID_RESET_ON_STARTUP=1
```

In this mode SQLite is still used directly, but the database is placed in a temporary runtime directory and recreated when the process starts. A hosted restart therefore gives the next visitor a fresh demo. The public demo does not depend on browser localStorage.

For local development, `run_windows.bat` uses persistent SQLite under `backend/data/face_recognition.db`. For a hosted demo, use `run_public_demo.bat` or equivalent environment variables.

By default, ephemeral mode also resets after 120 minutes without an interactive action. Recognition, enrollment, deletion, demo loading, threshold changes, and evaluation count as interactive actions; passive dashboard polling does not. Configure `VISIONID_IDLE_RESET_MINUTES` if needed.

Do not commit a production `face_recognition.db` containing real biometric data to a public GitHub repository. The runtime DB is ignored by `.gitignore`.


## Interview explanation

### Why SQLite?

The project has a small enrollment population and needs zero-cost, portable, transparent persistence. SQLite provides transactions, uniqueness constraints, foreign keys, and BLOB storage without a separate database server.

### Why store images in SQLite?

It removes a second data store. The current runtime dataset, face embeddings, activity, and evaluation records can all be managed through the same database path.

### Why both image hashing and face similarity?

An image hash detects duplicate files, while embedding similarity detects the more important case where the same face is re-uploaded or saved under a different filename/name.

### Why 0.65?

It is the initial configured Unknown-rejection threshold from the assignment design. The Evaluation Lab can be used to validate or tune it against an actual dataset.

## Security / privacy

Face embeddings and face images are biometric data. Keep production database files private, restrict access to the management UI/API, and add authentication/authorization before using the system beyond an internship demonstration.

## Project files

```text
visionid-sqlite/
├── backend/
│   ├── __init__.py
│   ├── database.py
│   ├── face_engine.py
│   ├── main.py
│   └── data/
│       └── face_recognition.db   # created automatically; ignored by Git
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── requirements.txt
├── run_windows.bat
├── .gitignore
└── README.md
```

There is intentionally no `launch.py`: FastAPI is started directly with Uvicorn.

## Recommended public-demo behavior

Use the included `Dockerfile` or equivalent hosted environment with:

```text
VISIONID_EPHEMERAL=1
VISIONID_RESET_ON_STARTUP=1
VISIONID_IDLE_RESET_MINUTES=120
```

This intentionally makes the hosted demo temporary. Data is still stored through SQLite and is available to the application during the live runtime, but a process restart recreates the database. If there is no interactive action for 120 minutes, the next request resets the runtime database as well.

This is appropriate for an internship demo where reviewers should be able to test enrollment and recognition without your personal enrollment data becoming permanent.

## Duplicate identity decision

- Recognition acceptance threshold: **0.65**
- Duplicate identity threshold: **0.80**

The duplicate threshold is checked against **every previously stored face sample**. A new name is rejected when the incoming face reaches the duplicate threshold for an already enrolled identity. The API returns a clear `409` response containing the existing identity and similarity score.

The check also runs against images already accepted earlier in the same enrollment request, so two photos of the same person cannot be submitted together under different names.

## GitHub / Codespaces demo

This repository is safe to publish without runtime biometric data. The public-demo configuration uses an ephemeral SQLite database created at runtime. See `GITHUB_DEPLOYMENT.md` for Codespaces setup.


## Major Limitations and Future Improvements
**1. Recognition accuracy can vary**

Recognition performance may decrease under poor lighting, different face angles, motion blur, low-resolution images, or partial occlusion.

**Future improvement:** Improve face alignment and image preprocessing, use multiple reference images for each person, and evaluate the model under different lighting, angles, distances, and expressions.

**2. Fixed similarity threshold**

The system currently uses a fixed recognition threshold, which may not provide the same performance in every environment.

**Future improvement:** Use a validation dataset to tune the threshold and analyze FAR (False Accept Rate) and FRR (False Reject Rate) at different threshold values.

**3. No liveness detection**

The current system recognizes a face but does not determine whether it belongs to a real person or is being presented through a photograph or video.

**Future improvement:** Add an anti-spoofing or liveness-detection module using techniques such as blink detection, facial movement analysis, depth sensing, or a dedicated anti-spoofing model.

**4. CPU-based inference can affect real-time performance**

Face detection and embedding generation can be computationally expensive, especially during continuous webcam recognition.

**Future improvement:** Use GPU acceleration, ONNX/TensorRT optimization, frame skipping, face tracking, and asynchronous inference to reduce latency and improve FPS.

**5. SQLite is suitable mainly for small-scale deployments**

SQLite is simple and efficient for a small number of identities but is not ideal for large-scale systems with many concurrent users and large amounts of biometric data.

**Future improvement:** Migrate to PostgreSQL or another server-based database and use FAISS or a vector database for faster similarity searches with large numbers of embeddings.

**6. Evaluation dataset is limited**

The evaluation results depend on the size and diversity of the available test dataset.

**Future improvement:** Build a larger evaluation dataset containing multiple identities, known and unknown faces, different lighting conditions, poses, expressions, distances, and occlusions, and use additional metrics such as confusion matrices, ROC curves, FAR, FRR, and inference latency.
