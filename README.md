# Meta-LLM Hallucination Detection Using Claim Graph Decomposition

The project accepts a question, generates an answer, extracts factual claims,
retrieves evidence, and verifies each claim with an evidence-preserving
deterministic verifier. The API returns per-claim `SUPPORTED`, `CONTRADICTED`,
or `UNSUPPORTED` results, plus an aggregate `overall_result`. `UNSUPPORTED`
means that the available evidence is insufficient; the verifier never treats
retrieval alone as proof.

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
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
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
.\.venv\Scripts\python.exe -m compileall backend frontend modules database utils tests
.\.venv\Scripts\python.exe -c "from backend.main import app; print(app.title)"
.\.venv\Scripts\python.exe -m pytest -q
```

The frontend can be built from its directory with `npm.cmd run build`.

Example request:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/analyze `
  -ContentType "application/json" `
  -Body '{"question":"When did Chandrayaan-3 land on the Moon?"}'
```
