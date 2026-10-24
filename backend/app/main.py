"""TruthLens AI — unified FastAPI application."""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import registry
from .config import POLICY, ROOT, SETTINGS, Policy
from .services import compare as compare_svc, evaluation, integrity, live as live_svc, report as report_svc, transcript as transcript_svc
from .services.assistant import ask as assistant_ask
from .services.jobs import JobManager
from .services.media import MediaError, cleanup, detect_kind, ffprobe, temp_dir, validate_limits
from .services.store import Store

STORE = Store()
JOBS = JobManager(STORE)
TASKS: dict[str, dict] = {}
_rate: dict[str, deque] = defaultdict(deque)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]")


def _purge_loop():
    while True:
        try:
            STORE.purge_older_than(SETTINGS.retention_days)
        except Exception:
            pass
        time.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_purge_loop, daemon=True).start()
    yield


app = FastAPI(title="TruthLens AI", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def guard(request: Request, call_next):
    p = request.url.path
    if p.startswith("/api") and p != "/api/health":
        if SETTINGS.api_token:
            tok = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or request.headers.get("x-api-token", "")
            if tok != SETTINGS.api_token:
                return JSONResponse({"error": "unauthorized", "message": "Missing or invalid API token."}, status_code=401)
        if request.method == "POST":
            live = p.startswith("/api/live")
            ip, now = (request.client.host if request.client else "?"), time.time()
            q = _rate[ip + ("|live" if live else "")]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= (SETTINGS.live_rate_limit_per_min if live else SETTINGS.rate_limit_per_min):
                return JSONResponse({"error": "rate_limited", "message": "Too many requests; slow down."}, status_code=429)
            q.append(now)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cache-Control"] = "no-store" if p.startswith("/api") else resp.headers.get("Cache-Control", "no-cache")
    resp.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; media-src 'self' blob:"
    return resp


@app.exception_handler(MediaError)
async def media_error_handler(request: Request, exc: MediaError):
    status = {"too_large": 413, "too_long": 413, "unsupported": 415, "unreadable": 422, "no_session": 404,
              "busy": 503, "model_unavailable": 503}.get(exc.code, 400)
    return JSONResponse({"error": exc.code, "message": str(exc)}, status_code=status)


async def _save_upload(file: UploadFile, d: Path, name: str = "upload") -> Path:
    """Stream to disk with a hard size limit; filename is never used as a path."""
    dest, limit, size = d / name, SETTINGS.max_upload_mb * 1024 * 1024, 0
    with open(dest, "wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                out.close(); dest.unlink(missing_ok=True)
                raise MediaError(f"File exceeds the {SETTINGS.max_upload_mb} MB limit.", "too_large")
            out.write(chunk)
    if size == 0:
        raise MediaError("Uploaded file is empty.", "empty")
    return dest


def _clean_name(n: str | None) -> str:
    return _SAFE_NAME.sub("_", Path(n or "upload").name)[:120] or "upload"


# ----------------------------- info -----------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "fa3_available": registry.get_fa3().info()["available"],
            "audio_available": registry.get_audio().info()["available"]}


@app.get("/api/models")
def models():
    return {"fa3": registry.get_fa3().info(), "audio": registry.get_audio().info(),
            "asr": transcript_svc.get_transcriber().info()}


@app.get("/api/config")
def config():
    return {"policy": asdict(POLICY.get()),
            "limits": {"max_upload_mb": SETTINGS.max_upload_mb, "max_media_seconds": SETTINGS.max_media_seconds,
                       "rate_limit_per_min": SETTINGS.rate_limit_per_min},
            "privacy": {"uploaded_media_stored": False, "results_stored": True, "store_thumbnails": SETTINGS.store_thumbnails,
                        "retention_days": SETTINGS.retention_days, "external_services": False,
                        "stored_where": "local SQLite file in the data directory"},
            "auth_required": bool(SETTINGS.api_token)}


@app.put("/api/policy")
async def set_policy(request: Request):
    body = await request.json()
    try:
        return asdict(POLICY.update(body))
    except (ValueError, AssertionError, TypeError) as exc:
        raise HTTPException(400, f"Invalid policy: {exc}")


@app.get("/api/stats")
def stats():
    return STORE.stats()


# ----------------------------- analysis -----------------------------
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), expect: str = Form(""), integrity_check: bool = Form(True),
                  policy: str = Form("")):
    if expect not in ("", "image", "audio", "video"):
        raise HTTPException(400, "expect must be image, audio, video or empty")
    d = temp_dir()
    try:
        path = await _save_upload(file, d)
    except Exception:
        cleanup(d); raise
    opts = {"integrity": integrity_check}
    if policy:
        try:
            opts["policy"] = json.loads(policy)
            base = asdict(POLICY.get()); base.update(opts["policy"]); Policy(**base).validate()
        except Exception:
            cleanup(d); raise HTTPException(400, "Invalid policy override.")
    jid = JOBS.submit(d, path, _clean_name(file.filename), options=opts, force_kind=expect or None)
    return {"job_id": jid}


@app.post("/api/batch")
async def batch(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= 50:
        raise HTTPException(400, "Provide between 1 and 50 files.")
    bid, ids = uuid.uuid4().hex[:10], []
    for f in files:
        d = temp_dir()
        try:
            path = await _save_upload(f, d)
            ids.append(JOBS.submit(d, path, _clean_name(f.filename), batch_id=bid))
        except MediaError as exc:
            cleanup(d)
            jid = uuid.uuid4().hex[:12]
            JOBS.jobs[jid] = {"id": jid, "filename": _clean_name(f.filename), "status": "failed", "progress": 1.0,
                              "message": "rejected", "analysis_id": None, "error": str(exc), "error_code": exc.code,
                              "batch_id": bid, "attempts": 0, "created": time.time(), "_tmp": "", "_path": "", "_opts": {}, "_kind": None}
            ids.append(jid)
    return {"batch_id": bid, "job_ids": ids}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = JOBS.public(jid)
    if not j:
        raise HTTPException(404, "Unknown job.")
    return j


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str):
    if jid not in JOBS.jobs:
        raise HTTPException(404, "Unknown job.")
    return {"cancelled": JOBS.cancel(jid)}


@app.post("/api/jobs/{jid}/retry")
def job_retry(jid: str):
    if jid not in JOBS.jobs:
        raise HTTPException(404, "Unknown job.")
    ok = JOBS.retry(jid)
    if not ok:
        raise HTTPException(409, "This job cannot be retried (only recoverable failures such as timeouts can; media is not kept otherwise).")
    return {"retrying": True}


@app.get("/api/batches/{bid}")
def batch_status(bid: str):
    b = JOBS.batch(bid)
    if not b["jobs"]:
        raise HTTPException(404, "Unknown batch.")
    return b


@app.get("/api/batches/{bid}/report")
def batch_report(bid: str):
    b = JOBS.batch(bid)
    if not b["jobs"]:
        raise HTTPException(404, "Unknown batch.")
    rows = []
    for j in b["jobs"]:
        r = STORE.get(j["analysis_id"]) if j["analysis_id"] else None
        rows.append({"file": j["filename"], "status": j["status"], "error": j["error"],
                     "overall": r["assessment"]["overall_label"] if r else None, "analysis_id": j["analysis_id"],
                     "scenario": r["assessment"]["scenario"] if r else None})
    payload = {"report_type": "truthlens-ai-batch-report", "machine_generated": True, "batch_id": bid, "summary": b["summary"],
               "counts": b["counts"], "items": rows}
    return Response(json.dumps(payload, indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="batch_{bid}.json"'})


@app.get("/api/analyses")
def analyses(q: str = "", kind: str = "", overall: str = "", batch_id: str = "", limit: int = 200):
    return {"items": STORE.list(q, kind, overall, batch_id, min(max(limit, 1), 500))}


def _get(aid: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{8,32}", aid):
        raise HTTPException(400, "Invalid analysis id.")
    rec = STORE.get(aid)
    if not rec:
        raise HTTPException(404, "Analysis not found.")
    return rec


@app.get("/api/analyses/{aid}")
def analysis(aid: str):
    return _get(aid)


@app.delete("/api/analyses/{aid}")
def delete_analysis(aid: str):
    _get(aid)
    return {"deleted": STORE.delete(aid), "note": "Uploaded media was never stored; this removed the stored result."}


@app.get("/api/analyses/{aid}/report.json")
def rep_json(aid: str):
    return Response(report_svc.json_report(_get(aid)), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="truthlens_{aid}.json"'})


@app.get("/api/analyses/{aid}/report.pdf")
def rep_pdf(aid: str):
    return Response(report_svc.pdf_report(_get(aid)), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="truthlens_{aid}.pdf"'})


@app.post("/api/analyses/{aid}/ask")
async def ask(aid: str, request: Request):
    body = await request.json()
    q = str(body.get("question", ""))[:500]
    if not q.strip():
        raise HTTPException(400, "Empty question.")
    return assistant_ask(_get(aid), q)


# ----------------------------- generic tasks (inspect / compare / robustness / evaluation) ---------
def _run_task(tid: str, fn, dirs: list[Path]):
    t = TASKS[tid]; t["status"] = "running"
    try:
        t["result"] = fn(lambda f, m: t.update(progress=round(f, 3), message=m))
        t["status"] = "done"; t["progress"] = 1.0
    except MediaError as exc:
        t.update(status="failed", error=str(exc), error_code=exc.code)
    except Exception as exc:
        t.update(status="failed", error=f"Unexpected processing error ({type(exc).__name__}).", error_code="internal")
    finally:
        for d in dirs:
            cleanup(d)


def _start_task(fn, dirs=()) -> str:
    tid = uuid.uuid4().hex[:12]
    TASKS[tid] = {"id": tid, "status": "queued", "progress": 0.0, "message": "", "result": None, "error": None}
    JOBS.pool.submit(_run_task, tid, fn, list(dirs))
    return tid


@app.get("/api/tasks/{tid}")
def task(tid: str):
    t = TASKS.get(tid)
    if not t:
        raise HTTPException(404, "Unknown task.")
    return t


@app.post("/api/inspect")
async def inspect_(file: UploadFile = File(...)):
    d = temp_dir()
    try:
        p = await _save_upload(file, d)
    except Exception:
        cleanup(d); raise
    name = _clean_name(file.filename)

    def fn(prog):
        probe = ffprobe(p); kind = detect_kind(p, probe); validate_limits(p, probe, kind)
        return {"filename": name, "kind": kind, **integrity.inspect(p, kind, probe)}
    return {"task_id": _start_task(fn, [d])}


@app.post("/api/compare")
async def compare(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    da, db = temp_dir(), temp_dir()
    try:
        pa, pb = await _save_upload(file_a, da), await _save_upload(file_b, db)
    except Exception:
        cleanup(da); cleanup(db); raise
    na, nb = _clean_name(file_a.filename), _clean_name(file_b.filename)
    pol = POLICY.get()
    return {"task_id": _start_task(lambda prog: compare_svc.compare(pa, na, pb, nb, pol, prog), [da, db])}


@app.post("/api/robustness")
async def robustness(file: UploadFile = File(...)):
    d = temp_dir()
    try:
        p = await _save_upload(file, d)
    except Exception:
        cleanup(d); raise
    name, pol = _clean_name(file.filename), POLICY.get()
    return {"task_id": _start_task(lambda prog: compare_svc.robustness(p, name, pol, prog), [d])}


# ----------------------------- evaluation -----------------------------
@app.get("/api/evaluation")
def eval_status():
    layout = evaluation.scan()
    return {"dataset_counts": evaluation.counts(layout), "eval_data_dir_configured": SETTINGS.eval_data_dir,
            "latest": STORE.kv_get("evaluation_latest"),
            "requirements": "EVAL_DATA_DIR/<image|audio|video>/<test|val>/<real|fake>/ — see docs/EVALUATION.md"}


@app.post("/api/evaluation/run")
def eval_run():
    pol = POLICY.get()

    def fn(prog):
        r = evaluation.run(pol, prog)
        STORE.kv_set("evaluation_latest", r)
        return r
    return {"task_id": _start_task(fn)}


# ----------------------------- live detection -----------------------------
@app.post("/api/live/start")
def live_start():
    s = live_svc.start_session()
    return {"session_id": s.id, "rolling_seconds": live_svc.LIVE_ROLLING_S, "models": {
        "fa3": registry.get_fa3().info()["available"], "audio": registry.get_audio().info()["available"],
        "asr": transcript_svc.get_transcriber().info()["available"]},
        "privacy": "Audio chunks and frames are processed in memory/temp files and discarded; only scores and transcript text "
                   "are kept in server memory for this session and are not written to the history."}


@app.post("/api/live/{sid}/audio")
async def live_audio(sid: str, file: UploadFile = File(...)):
    s = live_svc.get_session(sid)
    d = temp_dir()
    try:
        limit, size = SETTINGS.live_max_chunk_mb * 1024 * 1024, 0
        dest = d / "chunk"
        with open(dest, "wb") as out:
            while chunk := await file.read(1 << 18):
                size += len(chunk)
                if size > limit:
                    raise MediaError(f"Chunk exceeds {SETTINGS.live_max_chunk_mb} MB.", "too_large")
                out.write(chunk)
        if size < 200:
            raise MediaError("Chunk is empty.", "empty")
        pol = POLICY.get()
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(None, live_svc.add_audio_chunk, s, dest, pol)
    finally:
        cleanup(d)


@app.post("/api/live/{sid}/frame")
async def live_frame(sid: str, file: UploadFile = File(...)):
    s = live_svc.get_session(sid)
    data = await file.read(SETTINGS.live_max_chunk_mb * 1024 * 1024 + 1)
    if not data or len(data) > SETTINGS.live_max_chunk_mb * 1024 * 1024:
        raise MediaError("Frame is empty or too large.", "too_large" if data else "empty")
    import asyncio
    return await asyncio.get_running_loop().run_in_executor(None, live_svc.add_frame, s, data, POLICY.get())


@app.delete("/api/live/{sid}")
def live_end(sid: str):
    return {"ended": live_svc.end_session(sid)}


frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
