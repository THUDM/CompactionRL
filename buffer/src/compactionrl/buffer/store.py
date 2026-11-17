"""Bounded persistent FIFO with idempotent insertion and expiring delivery leases."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class Buffer:
    def __init__(self, path, capacity=1024, lease_seconds=600):
        self.path = str(path)
        self.capacity = capacity
        self.lease_seconds = lease_seconds
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS samples (
                  sequence INTEGER PRIMARY KEY AUTOINCREMENT, uid TEXT UNIQUE NOT NULL,
                  payload TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'ready',
                  lease TEXT, expires REAL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, records):
        encoded = [
            (r["uid"], json.dumps(r, sort_keys=True, allow_nan=False)) for r in records
        ]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            novel = []
            for uid, payload in encoded:
                existing = db.execute(
                    "SELECT payload FROM samples WHERE uid=?", (uid,)
                ).fetchone()
                if existing:
                    if existing["payload"] != payload:
                        raise ValueError(f"UID reused with different contents: {uid}")
                else:
                    novel.append((uid, payload))
            active = db.execute(
                "SELECT count(*) FROM samples WHERE state != 'done'"
            ).fetchone()[0]
            if active + len(novel) > self.capacity:
                raise OverflowError("Buffer at capacity; retry after consumption")
            db.executemany("INSERT INTO samples(uid,payload) VALUES (?,?)", novel)
            db.commit()
        return len(novel)

    def take(self, count):
        if not 1 <= count <= self.capacity:
            raise ValueError("Invalid batch size")
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE samples SET state='ready',lease=NULL WHERE state='leased' AND expires<?",
                (now,),
            )
            rows = db.execute(
                "SELECT uid,payload FROM samples WHERE state='ready' ORDER BY sequence LIMIT ?",
                (count,),
            ).fetchall()
            if len(rows) < count:
                db.commit()
                return None, []
            lease = str(uuid.uuid4())
            db.executemany(
                "UPDATE samples SET state='leased',lease=?,expires=? WHERE uid=?",
                [(lease, now + self.lease_seconds, r["uid"]) for r in rows],
            )
            db.commit()
        return lease, [json.loads(r["payload"]) for r in rows]

    def ack(self, lease):
        with self.connect() as db:
            return db.execute(
                "UPDATE samples SET state='done' WHERE lease=? AND state='leased'",
                (lease,),
            ).rowcount

    def stats(self):
        with self.connect() as db:
            return dict(
                db.execute(
                    "SELECT state,count(*) FROM samples GROUP BY state"
                ).fetchall()
            )
