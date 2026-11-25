"""HTTP API (FastAPI) untuk VPS.

    POST /runs                (multipart, field "files" berulang)  -> {"run_id", "status_url", "zip_url"}
    GET  /runs/{run_id}       status + manifest bila selesai
    GET  /runs/{run_id}/zip   unduh paket ZIP
    GET  /runs/{run_id}/laporan/{nama}   unduh satu berkas laporan (mis. ringkasan.html)
    GET  /health

Jalankan: uvicorn app.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import config
from . import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = FastAPI(title="Checklist TOR & RAB", version="1.0")
_pool = ThreadPoolExecutor(max_workers=max(1, config.PIPELINE_WORKERS))
_RUN_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9a-f]{6}$")


def _run_dir(run_id: str) -> Path:
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(400, "run_id tidak valid")
    d = config.RUNS_DIR / run_id
    if not d.is_dir():
        raise HTTPException(404, "run tidak ditemukan")
    return d


@app.get("/health")
def health():
    import tor_extractor as tor
    ok, why = tor.ocr_status()
    return {"status": "ok", "ocr": ok, "ocr_info": why or "siap",
            "vlm_endpoint": bool(config.VLM_ENDPOINT_URL),
            "sft_endpoint": pipeline.ctor.SFT_ENDPOINT_URL is not None}


@app.post("/runs", status_code=202)
async def create_run(files: list[UploadFile] = File(...), vlm: bool = True):
    if not files:
        raise HTTPException(400, "minimal satu berkas")
    run_dir = pipeline.create_run()
    limit = int(config.MAX_UPLOAD_MB * 1024 * 1024)
    for uf in files:
        name = pipeline.safe_filename(uf.filename or "berkas.pdf")
        data = await uf.read()
        if len(data) > limit:
            raise HTTPException(413, f"{name} melebihi {config.MAX_UPLOAD_MB:g} MB")
        (run_dir / "upload" / name).write_bytes(data)
    _pool.submit(pipeline.process_run, run_dir, vlm)
    rid = run_dir.name
    return {"run_id": rid, "status_url": f"/runs/{rid}", "zip_url": f"/runs/{rid}/zip"}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    d = _run_dir(run_id)
    out = pipeline.read_status(d)
    out.pop("trace", None)
    if (d / "manifest.json").exists():
        out["manifest"] = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return out


@app.get("/runs/{run_id}/zip")
def get_zip(run_id: str):
    d = _run_dir(run_id)
    z = d / f"hasil_{run_id}.zip"
    if not z.exists():
        raise HTTPException(409, "run belum selesai")
    return FileResponse(z, media_type="application/zip", filename=z.name)


@app.get("/runs/{run_id}/laporan/{nama}")
def get_laporan(run_id: str, nama: str):
    d = _run_dir(run_id) / "laporan"
    p = (d / nama).resolve()
    if p.parent != d.resolve() or not p.is_file():
        raise HTTPException(404, "berkas tidak ditemukan")
    return FileResponse(p, filename=p.name)
