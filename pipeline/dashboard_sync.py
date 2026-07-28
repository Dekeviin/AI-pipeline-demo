"""Optional live link into AlgoDashboard's pipeline telemetry panel.

Writes the dashboard's own pipeline_meta / pipeline_stages / pipeline_details
tables (schema from AlgoDashboard/scripts/seed.mjs) as stages run, so the
"AI pipeline" card animates with real progress. Disabled by default; enable in
config.yaml or with --sync-dashboard. Every write is wrapped so a missing or
locked dashboard DB can never kill a pipeline run.
"""
import sqlite3
from datetime import datetime, timezone

# Dashboard DDL (mirrors AlgoDashboard/scripts/seed.mjs) — created if missing so
# the sync also works against a fresh, never-seeded dashboard DB.
DDL = """
CREATE TABLE IF NOT EXISTS pipeline_meta (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  current_task TEXT NOT NULL, new_data_fetched TEXT NOT NULL,
  data_range TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pipeline_stages (
  id INTEGER PRIMARY KEY, stage_order INTEGER NOT NULL,
  key TEXT NOT NULL, name TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('completed','running','standby','queued')),
  progress INTEGER NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pipeline_details (
  id INTEGER PRIMARY KEY, stage_id INTEGER NOT NULL REFERENCES pipeline_stages(id),
  label TEXT NOT NULL, value TEXT NOT NULL
);
"""

# our stage key → dashboard (stage_order, key, display name)
STAGE_MAP = {
    "ingestion": (1, "ingestion", "Data Ingestion"),
    "features": (2, "features", "Feature Engineering"),
    "training": (3, "training", "Model Training"),
    "testing": (4, "testing", "Advanced Testing"),
}


class DashboardSync:
    def __init__(self, cfg: dict):
        self.enabled = cfg["dashboard"]["enabled"]
        self.db_path = cfg["dashboard"]["db_path"]

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _exec(self, fn):
        if not self.enabled:
            return
        try:
            con = sqlite3.connect(self.db_path, timeout=2)
            try:
                con.executescript(DDL)
                fn(con)
                con.commit()
            finally:
                con.close()
        except sqlite3.Error as e:
            print(f"  [dashboard-sync] skipped: {e}")

    def _stage_id(self, con, key: str) -> int:
        order, k, name = STAGE_MAP[key]
        row = con.execute("SELECT id FROM pipeline_stages WHERE key = ?", (k,)).fetchone()
        if row:
            return row[0]
        con.execute(
            "INSERT INTO pipeline_stages (stage_order, key, name, status, progress, updated_at) "
            "VALUES (?,?,?,?,?,?)", (order, k, name, "queued", 0, self._now()))
        return con.execute("SELECT id FROM pipeline_stages WHERE key = ?", (k,)).fetchone()[0]

    def set_meta(self, current_task: str, data_range: str):
        def fn(con):
            con.execute(
                "INSERT INTO pipeline_meta (id, current_task, new_data_fetched, data_range, updated_at) "
                "VALUES (1,?,?,?,?) ON CONFLICT(id) DO UPDATE SET current_task=excluded.current_task, "
                "new_data_fetched=excluded.new_data_fetched, data_range=excluded.data_range, "
                "updated_at=excluded.updated_at",
                (current_task, self._now()[:10], data_range, self._now()))
        self._exec(fn)

    def update_stage(self, key: str, status: str, progress: int, details: dict | None = None):
        def fn(con):
            sid = self._stage_id(con, key)
            con.execute(
                "UPDATE pipeline_stages SET status=?, progress=?, updated_at=? WHERE id=?",
                (status, int(progress), self._now(), sid))
            if details is not None:
                con.execute("DELETE FROM pipeline_details WHERE stage_id=?", (sid,))
                con.executemany(
                    "INSERT INTO pipeline_details (stage_id, label, value) VALUES (?,?,?)",
                    [(sid, str(k), str(v)) for k, v in details.items()])
        self._exec(fn)

    def progress_cb(self, key: str):
        """Callback matching each stage's `progress(pct, details)` signature."""
        def cb(pct: int, details: dict):
            self.update_stage(key, "running" if pct < 100 else "completed", pct, details)
        return cb
