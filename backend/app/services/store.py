"""SQLite persistence for analysis records (results only — uploaded media is never stored)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from ..config import SETTINGS, data_path


class Store:
    def __init__(self, path: str | None = None):
        self.path = path or str(data_path("veritas.sqlite3"))
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript("""
            CREATE TABLE IF NOT EXISTS analyses(id TEXT PRIMARY KEY, filename TEXT, kind TEXT, created_at TEXT,
                overall TEXT, scenario TEXT, batch_id TEXT, sha256 TEXT, record TEXT);
            CREATE INDEX IF NOT EXISTS ix_created ON analyses(created_at);
            CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
            """)

    def put(self, rec: dict, batch_id: str | None = None) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO analyses VALUES(?,?,?,?,?,?,?,?,?)",
                             (rec["id"], rec["filename"], rec["kind"], rec["created_at"], rec["assessment"]["overall"],
                              rec["assessment"]["scenario"], batch_id, rec.get("sha256"), json.dumps(rec)))
            self._db.commit()

    def get(self, id_: str) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT record FROM analyses WHERE id=?", (id_,)).fetchone()
        return json.loads(r["record"]) if r else None

    def list(self, q: str = "", kind: str = "", overall: str = "", batch_id: str = "", limit: int = 200) -> list[dict]:
        sql, args = "SELECT id,filename,kind,created_at,overall,scenario,batch_id,sha256 FROM analyses WHERE 1=1", []
        if q:
            sql += " AND (filename LIKE ? OR id LIKE ? OR sha256 LIKE ?)"; args += [f"%{q}%"] * 3
        if kind:
            sql += " AND kind=?"; args.append(kind)
        if overall:
            sql += " AND overall=?"; args.append(overall)
        if batch_id:
            sql += " AND batch_id=?"; args.append(batch_id)
        sql += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def delete(self, id_: str) -> bool:
        with self._lock:
            n = self._db.execute("DELETE FROM analyses WHERE id=?", (id_,)).rowcount
            self._db.commit()
        return n > 0

    def purge_older_than(self, days: int) -> int:
        if days <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        with self._lock:
            n = self._db.execute("DELETE FROM analyses WHERE created_at < ?", (cutoff,)).rowcount
            self._db.commit()
        return n

    def stats(self) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT kind, overall, COUNT(*) c FROM analyses GROUP BY kind, overall").fetchall()
        total = sum(r["c"] for r in rows)
        return {"total": total, "by_overall": _group(rows, "overall"), "by_kind": _group(rows, "kind")}

    def kv_get(self, k: str, default=None):
        with self._lock:
            r = self._db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r["v"]) if r else default

    def kv_set(self, k: str, v) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", (k, json.dumps(v)))
            self._db.commit()


def _group(rows, key):
    out: dict = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + r["c"]
    return out
