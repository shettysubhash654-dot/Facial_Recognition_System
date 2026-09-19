# GitHub + Codespaces deployment

## Publish the repository

1. Create a public GitHub repository.
2. Push the project with `.gitignore` intact. Never commit a runtime database, `.env`, `.venv`, or biometric images.
3. The database is created at runtime from `database/schema.sql` / `backend/database.py`.

## Run in Codespaces

Create a Codespace from the `main` branch. The devcontainer installs dependencies, starts FastAPI on port 8000, and forwards that port.

In the Codespace **Ports** panel, set port 8000 to **Public** and copy the generated `https://...app.github.dev` URL.

The deployment uses ephemeral mode, so the demo database is runtime-only and resets on process start. GitHub Codespaces may stop after inactivity; if it restarts, the application needs to start again and the forwarded port may need to be made public again.
