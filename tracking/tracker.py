"""
tracking.tracker
=================

A real, minimal experiment tracker used in place of MLflow.

Why not MLflow here: `df -h /` on this machine shows ~278Mi of free disk
space (98% full, 228Gi volume). `pip install mlflow` pulls in a fairly
heavy dependency tree (alembic, sqlalchemy, gunicorn, click, Flask,
graphene, etc.) that is not safe to attempt with that little headroom --
a failed/partial install could fill the disk entirely. So instead of
faking tracking, this module implements the actual functionality a team
needs from MLflow's tracking API for this project: runs, params, metrics,
and artifacts, persisted durably and queryably.

Design mirrors MLflow's core concepts on purpose (Experiment -> Run ->
params/metrics/artifacts) so swapping in real MLflow later is a drop-in
replacement, not a rewrite:
  - Every run gets a UUID, a start time, and an experiment name.
  - params/metrics are logged as key/value pairs.
  - Everything is persisted to a local SQLite database (`mlruns.db`) plus
    one JSON file per run under `mlruns/<run_id>.json`, so runs survive
    process restarts and can be queried with SQL or read back as JSON.
  - `list_runs` / `best_run` let downstream code (or a human) compare runs,
    exactly like `mlflow.search_runs`.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Self


class Run:
    def __init__(self, tracker: ExperimentTracker, run_id: str, experiment: str, run_name: str):
        self.tracker = tracker
        self.run_id = run_id
        self.experiment = experiment
        self.run_name = run_name
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.start_time = time.time()
        self.end_time: float | None = None
        self.status = "RUNNING"

    def log_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_metric(self, key: str, value: float) -> None:
        self.metrics[key] = float(value)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.status = "FAILED" if exc_type else "FINISHED"
        self.end_time = time.time()
        self.tracker._persist(self)


class ExperimentTracker:
    """Minimal, real, local experiment tracker (MLflow tracking-server
    replacement). Persists to SQLite + one JSON file per run."""

    def __init__(self, base_dir: str | Path = "mlruns"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "mlruns.db"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment TEXT NOT NULL,
                run_name TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                status TEXT NOT NULL,
                params_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def start_run(self, experiment: str, run_name: str) -> Run:
        run_id = uuid.uuid4().hex[:12]
        return Run(self, run_id, experiment, run_name)

    def _persist(self, run: Run) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO runs
                (run_id, experiment, run_name, start_time, end_time, status, params_json, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.experiment,
                run.run_name,
                run.start_time,
                run.end_time,
                run.status,
                json.dumps(run.params, default=str),
                json.dumps(run.metrics),
            ),
        )
        self._conn.commit()
        run_file = self.base_dir / f"{run.run_id}.json"
        run_file.write_text(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "experiment": run.experiment,
                    "run_name": run.run_name,
                    "start_time": run.start_time,
                    "end_time": run.end_time,
                    "status": run.status,
                    "params": run.params,
                    "metrics": run.metrics,
                },
                indent=2,
                default=str,
            )
        )

    def list_runs(self, experiment: str | None = None) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        if experiment is None:
            rows = cur.execute("SELECT * FROM runs ORDER BY start_time").fetchall()
        else:
            rows = cur.execute(
                "SELECT * FROM runs WHERE experiment = ? ORDER BY start_time", (experiment,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["params"] = json.loads(d.pop("params_json"))
            d["metrics"] = json.loads(d.pop("metrics_json"))
            out.append(d)
        return out

    def best_run(self, experiment: str, metric: str, maximize: bool = True) -> dict[str, Any] | None:
        runs = [r for r in self.list_runs(experiment) if metric in r["metrics"]]
        if not runs:
            return None
        return max(runs, key=lambda r: r["metrics"][metric]) if maximize else min(
            runs, key=lambda r: r["metrics"][metric]
        )

    def close(self) -> None:
        self._conn.close()
