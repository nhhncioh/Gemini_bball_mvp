from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from celery.result import AsyncResult

from processor.tasks import process_video, celery_app

app = FastAPI(title="Gemini Basketball MVP", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class SubmitResponse(BaseModel):
    job_id: str
    file: str

class JobStatus(BaseModel):
    state: str
    result: Dict[str, Any] | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload", response_model=SubmitResponse)
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Missing filename")
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        f.write(await file.read())

    task = process_video.delay(str(dest))
    return SubmitResponse(job_id=task.id, file=str(dest))

@app.get("/result/{job_id}", response_model=JobStatus)
def result(job_id: str):
    res = AsyncResult(job_id, app=celery_app)
    payload = res.result if res.successful() else None
    return JobStatus(state=res.state, result=payload)
