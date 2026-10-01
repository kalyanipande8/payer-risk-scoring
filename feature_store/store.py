"""
feature_store.store
====================

A real, runnable local feature store standing in for a Databricks feature
store (Databricks itself is not provisioned here: no cluster, no
credentials, no cost). It implements the properties a payer risk-scoring
program actually needs from a feature store:

  1. Named feature tables with an explicit, versioned schema.
  2. Append-only version history -- a write never mutates a prior version,
     so any model trained against version K can always be reproduced.
  3. Point-in-time-correct retrieval: given a member id and an "as-of"
     timestamp (the moment a label/prediction event happened), return only
     the feature values that were VALID at or before that instant. This is
     the standard anti-leakage guarantee real feature stores provide.
  4. A metadata registry (owner/description/lineage) per feature,
     independent from the versioned data itself -- this is what lets 5
     different models discover and share the same standardized features.

Storage layer: SQLite for the registry/version index (ACID, single file,
real SQL point-in-time queries -- the same query shape as Delta's
time-travel/`AS OF`) plus flat CSV snapshots on disk for the actual rows
(append-only, one immutable file per version).
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FeatureMetadata:
    feature_name: str
    table_name: str
    owner: str
    description: str
    lineage: str
    dtype: str


class FeatureStore:
    """Local, file-backed feature store with real versioning and
    point-in-time-correct retrieval.

    Physical layout under `base_dir`:
      registry.db                        -- SQLite: schemas, version index, metadata
      snapshots/<table>/v<version>.csv   -- immutable feature rows for that version
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.snapshots_dir = self.base_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "registry.db"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS feature_tables (
                table_name TEXT PRIMARY KEY,
                entity_key TEXT NOT NULL,
                schema_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS table_versions (
                table_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                schema_json TEXT NOT NULL,
                snapshot_path TEXT NOT NULL,
                event_time REAL NOT NULL,
                ingested_at REAL NOT NULL,
                row_count INTEGER NOT NULL,
                PRIMARY KEY (table_name, version)
            );

            CREATE TABLE IF NOT EXISTS feature_metadata (
                table_name TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                owner TEXT,
                description TEXT,
                lineage TEXT,
                dtype TEXT,
                PRIMARY KEY (table_name, feature_name)
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register_table(self, table_name: str, entity_key: str, schema: dict[str, str]) -> None:
        if entity_key not in schema:
            raise ValueError(f"entity_key '{entity_key}' must appear in schema")
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT schema_json, entity_key FROM feature_tables WHERE table_name = ?",
            (table_name,),
        ).fetchone()
        if row is not None:
            existing_schema = json.loads(row["schema_json"])
            if existing_schema != schema or row["entity_key"] != entity_key:
                raise ValueError(f"Table '{table_name}' already registered with a different schema")
            return
        cur.execute(
            "INSERT INTO feature_tables (table_name, entity_key, schema_json, created_at) VALUES (?, ?, ?, ?)",
            (table_name, entity_key, json.dumps(schema), time.time()),
        )
        self._conn.commit()

    def register_feature_metadata(self, meta: FeatureMetadata) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO feature_metadata (table_name, feature_name, owner, description, lineage, dtype)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(table_name, feature_name) DO UPDATE SET
                owner=excluded.owner, description=excluded.description,
                lineage=excluded.lineage, dtype=excluded.dtype
            """,
            (meta.table_name, meta.feature_name, meta.owner, meta.description, meta.lineage, meta.dtype),
        )
        self._conn.commit()

    def get_metadata(self, table_name: str) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT * FROM feature_metadata WHERE table_name = ?", (table_name,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # writing new versions (append-only)
    # ------------------------------------------------------------------
    def write_version(self, table_name: str, rows: list[dict[str, Any]], event_time: float | None = None) -> int:
        cur = self._conn.cursor()
        table_row = cur.execute(
            "SELECT schema_json FROM feature_tables WHERE table_name = ?", (table_name,)
        ).fetchone()
        if table_row is None:
            raise ValueError(f"Table '{table_name}' is not registered")
        schema = json.loads(table_row["schema_json"])

        for r in rows:
            missing = set(schema) - set(r)
            if missing:
                raise ValueError(f"Row missing columns {missing} required by schema")

        max_version_row = cur.execute(
            "SELECT COALESCE(MAX(version), 0) AS mv FROM table_versions WHERE table_name = ?",
            (table_name,),
        ).fetchone()
        new_version = max_version_row["mv"] + 1

        table_dir = self.snapshots_dir / table_name
        table_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = table_dir / f"v{new_version}.csv"

        columns = list(schema.keys())
        with open(snapshot_path, "w", encoding="utf-8") as f:
            f.write(",".join(columns) + "\n")
            for r in rows:
                f.write(",".join(str(r[c]) for c in columns) + "\n")

        now = time.time()
        et = event_time if event_time is not None else now
        cur.execute(
            """
            INSERT INTO table_versions
                (table_name, version, schema_json, snapshot_path, event_time, ingested_at, row_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (table_name, new_version, json.dumps(schema), str(snapshot_path), et, now, len(rows)),
        )
        self._conn.commit()
        return new_version

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------
    def _load_snapshot(self, snapshot_path: str, schema: dict[str, str]) -> list[dict[str, Any]]:
        cast = {"int": int, "float": float, "str": str}
        rows: list[dict[str, Any]] = []
        with open(snapshot_path, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    continue
                values = line.split(",")
                row = {}
                for col, val in zip(header, values):
                    dtype = schema.get(col, "str")
                    row[col] = cast.get(dtype, str)(val)
                rows.append(row)
        return rows

    def list_versions(self, table_name: str) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT version, event_time, ingested_at, row_count FROM table_versions "
            "WHERE table_name = ? ORDER BY version",
            (table_name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_features(self, table_name: str, version: int | None = None) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        if version is None:
            row = cur.execute(
                "SELECT version, schema_json, snapshot_path FROM table_versions "
                "WHERE table_name = ? ORDER BY version DESC LIMIT 1",
                (table_name,),
            ).fetchone()
        else:
            row = cur.execute(
                "SELECT version, schema_json, snapshot_path FROM table_versions "
                "WHERE table_name = ? AND version = ?",
                (table_name, version),
            ).fetchone()
        if row is None:
            raise ValueError(f"No such version for table '{table_name}'")
        schema = json.loads(row["schema_json"])
        return self._load_snapshot(row["snapshot_path"], schema)

    def get_features_as_of(
        self, table_name: str, as_of: float, entity_ids: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Point-in-time-correct retrieval: latest version whose event_time
        is <= as_of. Never returns a version computed after `as_of` -- the
        core anti-leakage guarantee."""
        cur = self._conn.cursor()
        row = cur.execute(
            """
            SELECT version, schema_json, snapshot_path FROM table_versions
            WHERE table_name = ? AND event_time <= ?
            ORDER BY event_time DESC, version DESC LIMIT 1
            """,
            (table_name, as_of),
        ).fetchone()
        if row is None:
            return []
        schema = json.loads(row["schema_json"])
        data = self._load_snapshot(row["snapshot_path"], schema)
        if entity_ids is not None:
            table_meta = cur.execute(
                "SELECT entity_key FROM feature_tables WHERE table_name = ?", (table_name,)
            ).fetchone()
            ek = table_meta["entity_key"]
            wanted = set(entity_ids)
            data = [r for r in data if r[ek] in wanted]
        return data

    def point_in_time_join(
        self,
        table_name: str,
        entity_events: list[dict[str, Any]],
        entity_key: str,
        as_of_key: str = "event_time",
    ) -> list[dict[str, Any]]:
        """Build a leakage-free training set: for EACH labeled event,
        independently look up the feature version valid at that event's own
        timestamp."""
        cur = self._conn.cursor()
        versions = cur.execute(
            "SELECT version, schema_json, snapshot_path, event_time FROM table_versions "
            "WHERE table_name = ? ORDER BY event_time",
            (table_name,),
        ).fetchall()
        if not versions:
            raise ValueError(f"No versions available for table '{table_name}'")

        loaded = []
        for v in versions:
            schema = json.loads(v["schema_json"])
            rows = self._load_snapshot(v["snapshot_path"], schema)
            by_entity = {r[entity_key]: r for r in rows}
            loaded.append((v["event_time"], by_entity))

        results = []
        for event in entity_events:
            eid = event[entity_key]
            as_of = event[as_of_key]
            best = None
            for ev_time, by_entity in loaded:
                if ev_time <= as_of and eid in by_entity and (best is None or ev_time > best[0]):
                    best = (ev_time, by_entity[eid])
            joined = dict(event)
            if best is not None:
                for k, v in best[1].items():
                    if k != entity_key:
                        joined[f"feat_{k}"] = v
                joined["_feature_version_event_time"] = best[0]
            else:
                joined["_feature_version_event_time"] = None
            results.append(joined)
        return results

    def close(self) -> None:
        self._conn.close()
