"""In-process job queue (thread pool) with progress, cancel and retry. Uploaded media lives in a temp dir that is
deleted when the job ends (success, failure or cancel)."""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import SETTINGS, POLICY
from .media import MediaError, cleanup
from .pipeline import Cancelled, analyze_file, slim

RECOVERABLE = {"timeout", "tool_missing_transient"}


class JobManager:
    def __init__(self, store, workers: int | None = None):
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=workers or SETTINGS.workers)
        self.jobs: dict[str, dict] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def submit(self, tmpdir: Path, path: Path, filename: str, batch_id: str | None = None, options: dict | None = None,
               force_kind: str | None = None) -> str:
        jid = uuid.uuid4().hex[:12]
        ev = threading.Event()
        job = {"id": jid, "filename": filename, "status": "queued", "progress": 0.0, "message": "queued",
               "analysis_id": None, "error": None, "error_code": None, "batch_id": batch_id, "attempts": 0,
               "created": time.time(), "_tmp": str(tmpdir), "_path": str(path), "_opts": options or {}, "_kind": force_kind}
        with self._lock:
            self.jobs[jid] = job
            self._cancel[jid] = ev
        self.pool.submit(self._run, jid)
        return jid

    def _run(self, jid: str) -> None:
        job, ev = self.jobs[jid], self._cancel[jid]
        job["attempts"] += 1
        if ev.is_set():
            job.update(status="cancelled", message="cancelled before start")
            cleanup(Path(job["_tmp"])); return
        job.update(status="running", error=None, error_code=None)

        def prog(f, m):
            job["progress"], job["message"] = round(f, 3), m
        try:
            opts = job["_opts"]
            pol = POLICY.get()
            if opts.get("policy"):
                from dataclasses import asdict
                from ..config import Policy
                d = asdict(pol); d.update(opts["policy"]); pol = Policy(**d).validate()
            rec = analyze_file(Path(job["_path"]), job["filename"], pol, prog, ev,
                               do_integrity=opts.get("integrity", True), force_kind=job["_kind"])
            self.store.put(rec, job["batch_id"])
            job.update(status="done", progress=1.0, message="done", analysis_id=rec["id"])
            cleanup(Path(job["_tmp"]))
        except Cancelled:
            job.update(status="cancelled", message="cancelled")
            cleanup(Path(job["_tmp"]))
        except MediaError as exc:
            job.update(status="failed", error=str(exc), error_code=exc.code, message="failed")
            if exc.code not in RECOVERABLE:
                cleanup(Path(job["_tmp"]))
        except Exception as exc:  # never leak internals/paths
            job.update(status="failed", error=f"Unexpected processing error ({type(exc).__name__}).",
                       error_code="internal", message="failed")
            cleanup(Path(job["_tmp"]))

    def retry(self, jid: str) -> bool:
        job = self.jobs.get(jid)
        if not job or job["status"] != "failed" or job["error_code"] not in RECOVERABLE or not Path(job["_tmp"]).exists():
            return False
        self._cancel[jid] = threading.Event()
        job.update(status="queued", progress=0.0, message="queued (retry)")
        self.pool.submit(self._run, jid)
        return True

    def cancel(self, jid: str) -> bool:
        ev = self._cancel.get(jid)
        if not ev or self.jobs[jid]["status"] in ("done", "failed", "cancelled"):
            return False
        ev.set()
        if self.jobs[jid]["status"] == "queued":
            self.jobs[jid].update(status="cancelled", message="cancelled")
        return True

    def public(self, jid: str) -> dict | None:
        j = self.jobs.get(jid)
        return None if not j else {k: v for k, v in j.items() if not k.startswith("_")}

    def batch(self, batch_id: str) -> dict:
        js = [self.public(j) for j, v in self.jobs.items() if v["batch_id"] == batch_id]
        counts: dict = {}
        for j in js:
            counts[j["status"]] = counts.get(j["status"], 0) + 1
        done = [self.store.get(j["analysis_id"]) for j in js if j["analysis_id"]]
        summary: dict = {}
        for r in done:
            summary[r["assessment"]["overall"]] = summary.get(r["assessment"]["overall"], 0) + 1
        return {"batch_id": batch_id, "jobs": js, "counts": counts, "summary": summary,
                "finished": all(j["status"] in ("done", "failed", "cancelled") for j in js) if js else False}
