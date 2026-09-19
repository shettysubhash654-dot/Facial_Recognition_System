# Verification

## Automated checks

- Python compilation: passed
- Duplicate-logic unit tests: passed (5/5)
- FastAPI backend import: passed
- Deterministic API duplicate test: passed

## Duplicate enrollment test

Scenario:

1. Enroll image of person A as `A`.
2. Submit a different JPEG of the same face as `B`.
3. The API returns HTTP 409 and a clear message such as:
   `Person already exists as 'A'...`

The check uses the stored face embeddings for all existing identities and a duplicate threshold of `0.80`.

## Ephemeral demo mode

With:

```text
VISIONID_EPHEMERAL=1
VISIONID_RESET_ON_STARTUP=1
VISIONID_IDLE_RESET_MINUTES=120
```

the application uses SQLite in the temporary runtime directory. The database is recreated at process startup and is reset after the configured period of inactivity. Passive dashboard polling does not keep the runtime alive; recognition/enrollment/deletion/evaluation do.
