VisionID — Face Recognition Command Center

GitHub Repository: https://github.com/shettysubhash654-dot/Facial_Recognition_System

Live demo: The project can be run from GitHub Codespaces. Open the repository → Code → Codespaces → Create codespace on main → start FastAPI on port 8000 → open the forwarded port from the PORTS panel.

A FastAPI + browser-webcam face-recognition system using MTCNN, InceptionResnetV1/VGGFace2, cosine similarity, configurable Unknown rejection, SQLite persistence, duplicate-identity protection, and a database-backed Evaluation Lab.

🚀 Quick Start / Live Demo with GitHub Codespaces

The easiest way to try the project without installing Python locally is GitHub Codespaces.

1. Open the repository

https://github.com/shettysubhash654-dot/Facial_Recognition_System

2. Start a Codespace

On GitHub:

Code
  → Codespaces
  → Create codespace on main

Wait for the browser-based VS Code environment to finish loading.

3. Start VisionID

Open the Codespace terminal and run:

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

You should see:

Uvicorn running on http://0.0.0.0:8000
Application startup complete.

4. Open the public/demo URL

At the bottom of Codespaces:

PORTS
  → 8000
  → Open in Browser

For a recruiter-facing demo, set the forwarded port visibility to Public before sharing it.

The URL will look similar to:

https://<codespace-name>-8000.app.github.dev/

5. Test the application

Use the VisionID dashboard to try:

Dashboard
→ Recognize
→ Enroll Identity
→ People Directory
→ Evaluation Lab
→ Model Notes

Important: the forwarded Codespaces URL is a live development/demo URL. It only works while the Codespace and Uvicorn application are running. If the Codespace stops, start the Codespace and Uvicorn again, then reopen the forwarded port.

Project overview

VisionID provides a client-style face-recognition interface rather than requiring users to interact directly with API endpoints.

Main capabilities

Browser webcam recognition

Uploaded-image recognition

Face detection using MTCNN

Face embeddings using InceptionResnetV1 pretrained on VGGFace2

Cosine-similarity identity matching

Configurable Unknown rejection threshold

Duplicate-identity protection

Multiple reference samples per identity

Enrollment from uploaded images or webcam capture

People Directory with deletion

SQLite-backed activity and identity storage

Database-driven Evaluation Lab

Accuracy, FAR, FRR and detection-failure metrics

Visual evaluation chart

Optional 10-identity demo dataset loader

FastAPI backend and OpenAPI/Swagger documentation

GitHub Codespaces support

Ephemeral public-demo mode

Storage architecture

This version intentionally removes JSON persistence and browser localStorage for application data.

All application state is stored through one SQLite database:

backend/
└── data/
    └── face_recognition.db

The database stores:

people — identities and metadata

face_samples — canonical JPEG bytes, SHA-256, perceptual hash, face embeddings, sample role, and expected label

activity — recognition events

settings — matching threshold and duplicate threshold

evaluation_runs — metric summaries for each run

evaluation_results — per-sample evaluation decisions

demo_sources — optional demo dataset source registry

There are no people.json, activity.json, settings.json, embeddings.json, or evaluation/dataset/ runtime stores.

Database path

The connection is deliberately explicit in backend/database.py:

DB_PATH = Path(__file__).resolve().parent / "data" / "face_recognition.db"

and database operations use:

sqlite3.connect(DB_PATH)

The database therefore belongs to the machine/server running VisionID. If the project is hosted with persistent server storage, the database persists with that deployment. In the public-demo configuration described below, the database is intentionally temporary.

Duplicate identity protection

VisionID checks an incoming face against all previous face samples, not only the first image stored for a person.

The enrollment pipeline uses three layers:

Exact duplicate: SHA-256 of the canonical JPEG.

Visual duplicate candidate: perceptual hashing for re-encoded/resized copies.

Identity duplicate: face-embedding cosine similarity of 0.80 or higher blocks the same face from being registered under a different name.

The configured thresholds are:

Recognition / Unknown threshold: 0.65
Duplicate identity threshold:     0.80

Example:

Photo 1 → enroll as Person A       ✅
Different photo of same person → B ❌ Person already exists as A

A genuinely different reference image can be added under the same existing identity to improve recognition robustness.

The API returns a clear conflict (HTTP 409) when a duplicate identity is detected.

Duplicate detection is threshold-based, not mathematically perfect. Very different images, heavy occlusion, extreme pose, or poor lighting can reduce embedding similarity. The threshold should therefore be validated on representative data before production deployment.

Recognition pipeline

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

If the best similarity is below the configured recognition threshold, the system returns Unknown instead of forcing the closest enrolled identity.

Evaluation Lab

Evaluation does not depend on a separate evaluation folder. Each run reads the current contents of face_recognition.db.

Evaluation behavior

Current SQLite data
       ↓
Current enrolled identities/samples
       ↓
Current unknown samples
       ↓
Face detection + embedding + matching
       ↓
Metrics
       ↓
Stored evaluation run + per-sample results
       ↓
Visual chart in the UI

For known identities, the evaluator uses leave-one-out matching, so an image is not scored against its own embedding. A person with only one reference sample is marked not_evaluable_single_reference instead of being treated as a false-perfect match.

Unknown evaluation samples can be stored directly in face_samples with person_id = NULL and sample_role = 'unknown'.

Each run can report:

Accuracy

False Accept Rate (FAR)

False Reject Rate (FRR)

Detection failure rate

Per-identity accuracy

Per-sample decisions

Current dataset/sample counts

The results are recalculated from the current database state every time the evaluation button is used.

Therefore:

Enroll new data
      ↓
Database changes
      ↓
Run Evaluation
      ↓
Current evaluation results

and:

Delete identity
      ↓
Database changes
      ↓
Run Evaluation
      ↓
Current evaluation results

No fixed accuracy number is hard-coded into the application.

Optional 10-identity demo dataset

The Evaluation Lab provides a Load 10 demo identities action.

When used, the application can fetch configured public-source images at runtime, convert them to JPEG, generate embeddings, and store image bytes/embeddings directly in SQLite.

Nothing is written to a separate evaluation/dataset/ runtime folder.

The public source/attribution information is stored in the demo_sources table.

For an actual submission, demo images should be treated as evaluation/demo material rather than as the model's training data.

Run on Windows

Recommended

Double-click:

run_windows.bat

Or from Command Prompt:

cd C:\path\to\Facial_Recognition_System
run_windows.bat

The script creates .venv, installs the pinned CPU build, initializes SQLite, and starts FastAPI.

Open:

http://127.0.0.1:8000/

Swagger/API documentation:

http://127.0.0.1:8000/docs

Manual start

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

For a local browser, use:

http://127.0.0.1:8000/

0.0.0.0 is the server bind address; it is not the URL you should type into the browser.

Core UI

Dashboard — current identity count, recognition activity, threshold

Recognize — image upload and live browser-webcam recognition

Enroll Identity — image upload and webcam capture

People Directory — list and delete identities

Evaluation Lab — run database-backed evaluation and charts

Model Notes — explain the inference pipeline

Public demo mode — temporary storage

For a public demo where enrollment data should not become permanent, run with:

VISIONID_EPHEMERAL=1
VISIONID_RESET_ON_STARTUP=1

Optional idle reset:

VISIONID_IDLE_RESET_MINUTES=120

In this mode, SQLite is still used directly, but the database is placed in a temporary runtime location and recreated when the process starts. A hosted restart therefore gives the next visitor a fresh demo environment.

By default, ephemeral mode can also reset after the configured inactivity period. Interactive actions such as recognition, enrollment, deletion, demo loading, threshold changes, and evaluation count as activity; passive dashboard polling does not.

This mode is intended for an internship/demo environment where reviewers can test the application without your personal enrollment database becoming permanent.

Hosting note: ephemeral storage means data is intentionally disposable. Do not use this configuration for a production deployment that requires long-term records.

Limitations

1. Recognition accuracy depends on image quality

Extreme head angles, poor lighting, motion blur, low resolution, sunglasses/masks, partial faces, and heavy occlusion can reduce detection and matching reliability.

Future improvement: use stronger face detection/alignment, quality scoring, image normalization, pose handling, and a broader validation set representing real deployment conditions.

2. The similarity threshold is not universal

The current recognition threshold is 0.65 and the duplicate threshold is 0.80. These are configuration values, not guarantees of accuracy across all cameras, demographics, environments, or datasets.

Future improvement: calibrate thresholds using validation data, report ROC/DET curves, and choose thresholds based on the application's acceptable false-accept/false-reject trade-off.

3. Duplicate detection can produce edge cases

Embedding similarity is much stronger than filename/image hashing for detecting the same person across different photographs, but no fixed threshold can perfectly separate every identity pair.

Future improvement: use multiple reference embeddings per identity, score aggregation, adaptive thresholds, image-quality gates, and a stronger recognition model.

4. CPU inference can be slow

The current setup is designed to run without a dedicated GPU. Face detection and embedding generation can therefore have noticeable latency, particularly with multiple faces or repeated evaluation.

Future improvement: ONNX Runtime, model quantization, batching, GPU inference, asynchronous processing, and optimized face detectors/embedding models.

5. SQLite is designed for a small/simple deployment

SQLite is portable and appropriate for a small internship project, but it is not the ideal choice for a large multi-user production system with heavy concurrent writes.

Future improvement: migrate to PostgreSQL or another managed relational database, while keeping embeddings in a dedicated vector index/vector database when the dataset becomes large.

6. No authentication/authorization layer

The current demo interface is designed for evaluation and demonstration. It does not provide full production user-management controls.

Future improvement: add authentication, role-based access control, protected admin actions, API tokens/session security, rate limiting, audit controls, and secure secret management.

7. No liveness / anti-spoofing protection

A face-recognition system can potentially be fooled by a photograph or replayed video if liveness checks are not added.

Future improvement: add presentation-attack detection/liveness models, challenge-response mechanisms, depth/IR sensing, or other anti-spoofing controls appropriate to the hardware.

8. Browser webcam and server deployment are different from a desktop camera

The local desktop/web version uses the browser camera interface. A cloud-hosted deployment does not automatically have access to the server machine's physical webcam; it must use the reviewer's browser camera through web APIs.

Future improvement: keep the browser-camera workflow as the primary web deployment path and optionally add a dedicated native desktop client for controlled camera hardware.

9. Codespaces is a development/demo environment

A public Codespaces URL is useful for demonstrating the project, but it is not equivalent to a permanent production hosting platform. The Codespace can stop, and the Uvicorn process stops with it.

Future improvement: deploy the same application to a persistent cloud service with managed storage/database, authentication, monitoring, HTTPS, and production process management.

10. Privacy and biometric-data requirements

Face images and face embeddings are biometric information and require careful handling.

Future improvement: implement explicit consent, data-retention policies, encryption at rest/in transit, access controls, audit logs, deletion workflows, and jurisdiction-appropriate privacy/compliance review before real-world deployment.

Future roadmap

A production-oriented roadmap could be:

Current
  ↓
FastAPI + MTCNN + FaceNet + SQLite
  ↓
Quality / threshold calibration
  ↓
ONNX / optimized inference
  ↓
PostgreSQL + vector search for scale
  ↓
Authentication + authorization
  ↓
Liveness / anti-spoofing
  ↓
Monitoring + structured audit logs
  ↓
Persistent cloud deployment
  ↓
Production security/privacy controls

Potential future additions:

ONNX Runtime or optimized inference

Better face detector / recognition model

GPU acceleration and batching

Vector database for large-scale matching

Liveness detection

Role-based access control

PostgreSQL for multi-user deployments

Background evaluation jobs

ROC/DET and threshold calibration tools

Prometheus/OpenTelemetry/Grafana monitoring

Automated model and dependency updates

Interview explanation

Why SQLite?

The project has a small enrollment population and needs zero-cost, portable, transparent persistence. SQLite provides transactions, uniqueness constraints, foreign keys, and BLOB storage without a separate database server.

Why store images in SQLite?

It removes a second runtime data store. The current images, face embeddings, activity, and evaluation records can all be managed through the same database path.

Why both image hashing and face similarity?

An image hash detects duplicate files, while embedding similarity handles the more important case where the same face is re-uploaded or saved under a different filename or with different compression.

Why 0.65?

It is the initial configured Unknown-rejection threshold from the assignment design. The Evaluation Lab should be used to validate or tune it against representative data before production use.

Why 0.80 for duplicate identities?

Duplicate protection uses a stricter threshold than recognition acceptance so that the enrollment workflow can be conservative about registering the same person under a new name.

Security / privacy

Face embeddings and face images are biometric data. Keep production database files private, restrict access to the management UI/API, and add authentication/authorization before using the system beyond an internship demonstration.

Never commit a production face_recognition.db containing real biometric data to a public GitHub repository. The runtime database is ignored by .gitignore.

Project files

Facial_Recognition_System/
├── backend/
│   ├── __init__.py
│   ├── database.py
│   ├── face_engine.py
│   ├── main.py
│   └── data/
│       └── face_recognition.db   # created at runtime; ignored by Git
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── database/
│   └── schema.sql
├── tests/
├── .devcontainer/
├── .github/
├── requirements.txt
├── Dockerfile
├── run_windows.bat
├── run_public_demo.bat
├── README.md
├── VERIFICATION.md
└── GITHUB_DEPLOYMENT.md

There is intentionally no launch.py: FastAPI is started directly with Uvicorn.

GitHub / Codespaces demo guide

Repository

https://github.com/shettysubhash654-dot/Facial_Recognition_System

Launch

GitHub repository
   ↓
Code
   ↓
Codespaces
   ↓
Create codespace on main
   ↓
Terminal
   ↓
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ↓
PORTS
   ↓
8000
   ↓
Open in Browser

For a recruiter-facing URL, set port 8000 to Public in the PORTS panel.

If the page shows HTTP 502

A 502 in Codespaces usually means the forwarded port is available but the application is not successfully running on port 8000.

Check the terminal for:

Application startup complete.

If the server is not running, restart it with:

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

The Codespace URL should then be refreshed.

Local versus hosted storage

Local development uses:

backend/data/face_recognition.db

The public-demo mode can use temporary runtime storage so reviewer enrollment data is disposable.

Verification

The project includes automated tests covering the main persistence, duplicate-protection, configuration, and evaluation-regression paths.

Run:

python -m pytest -q

Then optionally run a manual smoke test through the UI.

Do not publish fabricated accuracy numbers. Evaluation metrics should be generated from the actual dataset used for testing.

License / third-party models

The project uses third-party open-source libraries listed in requirements.txt. The face embedding model is the pretrained InceptionResnetV1 implementation provided by facenet-pytorch and its VGGFace2 weights. Review the respective project/model licenses and terms before any commercial deployment.
