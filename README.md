# Meta-LLM Hallucination Detection Using Claim Graph Decomposition

This repository currently contains the deployable project foundation. LLM generation, claim extraction, claim graphs, retrieval, verification, and scoring will be added in later stages.

## Setup

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` only when local configuration needs to change. `DATABASE_PATH` is relative by default and can be replaced with another local path.

## Run the API

```powershell
python -m uvicorn backend.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Check `http://127.0.0.1:8000/health` for service status.

## Run the frontend

```powershell
Set-Location frontend
npm.cmd install
npm.cmd run dev
```

The Next.js frontend is available at `http://localhost:3000`.

## Test and validate imports

```powershell
python -m compileall backend frontend modules database utils tests
python -c "from backend.main import app; print(app.title)"
pytest
```
