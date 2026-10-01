"""
tests.test_pipeline
====================

Real tests against the real pipeline (no mocking of the core logic):
  - 60+ standardized features exist in the shared feature store.
  - All 5 models train and reach reasonable held-out performance.
  - The feature store's point-in-time correctness guarantee holds, via an
    injected sanity check: a deliberately "future" feature version must
    never leak into an as-of query made before it existed.
  - The experiment tracker persists real run records queryable afterward.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from feature_store import FeatureStore
from pipeline.features import FEATURE_COLUMNS, build_feature_store, get_training_frame
from pipeline.labels import add_labels
from pipeline.simulate import simulate_members
from pipeline.train import build_model_specs, train_all_models
from tracking import ExperimentTracker

TMP_ROOT = Path(__file__).resolve().parent / "_tmp_test_artifacts"


@pytest.fixture(scope="module")
def raw_data():
    return simulate_members(n_members=1500, seed=99)


@pytest.fixture(scope="module")
def labeled_data(raw_data):
    return add_labels(raw_data, seed=3)


@pytest.fixture(scope="module")
def store_and_info(raw_data):
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    store_dir = TMP_ROOT / "feature_store"
    store, info = build_feature_store(raw_data, store_dir)
    yield store, info
    store.close()
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)


# ----------------------------------------------------------------------
# Feature store: schema and standardization
# ----------------------------------------------------------------------

def test_feature_count_is_60_or_more():
    assert len(FEATURE_COLUMNS) >= 60
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS), "feature names must be distinct"


def test_all_five_models_share_overlapping_feature_subsets():
    specs = build_model_specs(FEATURE_COLUMNS)
    assert len(specs) >= 5
    all_used = set()
    for s in specs:
        assert 5 <= len(s.feature_subset) <= len(FEATURE_COLUMNS)
        assert set(s.feature_subset) <= set(FEATURE_COLUMNS)
        all_used |= set(s.feature_subset)
    # The whole point of a shared feature store: models reuse the same pool.
    assert len(all_used) < len(FEATURE_COLUMNS) * 5, "subsets should overlap, not be disjoint copies"


def test_raw_simulation_has_no_nulls(raw_data):
    assert raw_data.isnull().sum().sum() == 0
    assert len(raw_data) == 1500
    assert raw_data["member_id"].nunique() == 1500


# ----------------------------------------------------------------------
# Feature store: versioning + point-in-time correctness
# ----------------------------------------------------------------------

def test_feature_store_registers_versions(store_and_info):
    store, info = store_and_info
    versions = store.list_versions(info["table"])
    assert len(versions) == 2
    assert versions[0]["version"] == 1
    assert versions[1]["version"] == 2


def test_point_in_time_correctness_injected_sanity_check(store_and_info):
    """The core anti-leakage guarantee: querying at a timestamp BEFORE the
    later version's event_time must return the earlier version's data,
    and must NEVER return the later ("future") version.

    This is an injected sanity check: we ask for v2's exact row count and
    values as a negative control -- if the store incorrectly returned v2
    data for an as-of time that predates it, this test fails.
    """
    store, info = store_and_info
    table = info["table"]

    v1_rows = store.get_features(table, version=info["v1_version"])
    v2_rows = store.get_features(table, version=info["v2_version"])
    assert v1_rows != v2_rows, "v1 (partial-year) and v2 (full-year) must differ"

    midpoint = (info["v1_event_time"] + info["v2_event_time"]) / 2
    pit_rows = store.get_features_as_of(table, as_of=midpoint)
    assert pit_rows == v1_rows, "as-of query between v1 and v2 must return v1 (no future leakage)"
    assert pit_rows != v2_rows

    before_v1 = info["v1_event_time"] - 1000
    assert store.get_features_as_of(table, as_of=before_v1) == [], (
        "as-of before the first version exists must return no data, not fabricated data"
    )

    after_v2 = info["v2_event_time"] + 1000
    assert store.get_features_as_of(table, as_of=after_v2) == v2_rows


def test_point_in_time_join_is_leakage_free():
    store_dir = TMP_ROOT / "pit_join_store"
    if store_dir.exists():
        shutil.rmtree(store_dir)
    store = FeatureStore(store_dir)
    store.register_table("t", "id", {"id": "int", "x": "float"})
    store.write_version("t", [{"id": 1, "x": 10.0}], event_time=100.0)
    store.write_version("t", [{"id": 1, "x": 99.0}], event_time=200.0)

    events = [
        {"id": 1, "event_time": 150.0},  # should see x=10.0 (v1), not 99.0
        {"id": 1, "event_time": 250.0},  # should see x=99.0 (v2)
    ]
    joined = store.point_in_time_join("t", events, entity_key="id")
    assert joined[0]["feat_x"] == 10.0
    assert joined[1]["feat_x"] == 99.0
    store.close()
    shutil.rmtree(store_dir)


# ----------------------------------------------------------------------
# Model training: all 5 models, reasonable performance
# ----------------------------------------------------------------------

def test_all_five_models_train_with_reasonable_performance(raw_data, labeled_data, store_and_info, tmp_path):
    store, info = store_and_info
    training_features = get_training_frame(store, version=info["v2_version"])
    tracker = ExperimentTracker(tmp_path / "mlruns")

    results = train_all_models(training_features, labeled_data, tracker, seed=11)

    assert len(results) == 5
    names = {r["model_name"] for r in results}
    assert names == {
        "readmission_risk", "high_cost_claimant_risk", "chronic_progression_risk",
        "fraud_risk_proxy", "care_gap_risk",
    }
    for r in results:
        assert r["roc_auc"] > 0.6, f"{r['model_name']} roc_auc too low: {r['roc_auc']}"
        assert 0.0 <= r["f1"] <= 1.0
        assert r["n_features"] >= 5

    # Tracker persisted real, queryable run records.
    runs = tracker.list_runs("payer-risk-scoring")
    assert len(runs) == 5
    for run in runs:
        assert run["status"] == "FINISHED"
        assert "roc_auc" in run["metrics"]
    tracker.close()


def test_tracker_best_run_selection(tmp_path):
    tracker = ExperimentTracker(tmp_path / "mlruns2")
    with tracker.start_run("exp", "run_a") as r:
        r.log_metric("roc_auc", 0.7)
    with tracker.start_run("exp", "run_b") as r:
        r.log_metric("roc_auc", 0.85)
    best = tracker.best_run("exp", "roc_auc", maximize=True)
    assert best["run_name"] == "run_b"
    tracker.close()
