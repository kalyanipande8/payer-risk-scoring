#!/usr/bin/env python3
"""
run_all.py
==========

End-to-end pipeline: simulate claims/clinical data -> engineer 60+
standardized features -> write them into the versioned, point-in-time
correct feature store -> train and evaluate 5 real risk-scoring models ->
track every run -> print a results summary.

Run: python run_all.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from pipeline.features import FEATURE_COLUMNS, build_feature_store, get_training_frame
from pipeline.labels import add_labels
from pipeline.simulate import simulate_members
from pipeline.train import train_all_models
from tracking import ExperimentTracker

ROOT = Path(__file__).resolve().parent
STORE_DIR = ROOT / "data" / "feature_store_local"
MLRUNS_DIR = ROOT / "mlruns"


def main() -> dict:
    print("=" * 70)
    print("Payer Risk Scoring Framework -- run_all")
    print("=" * 70)

    # 1. Simulate raw claims + clinical data
    print("\n[1/5] Simulating claims + clinical data for members...")
    raw_df = simulate_members(n_members=6000, seed=42)
    print(f"    -> {len(raw_df)} members, {len(FEATURE_COLUMNS)} standardized features")
    assert len(FEATURE_COLUMNS) >= 60, "Feature store must expose 60+ standardized features"

    # 2. Generate the 5 model labels
    print("\n[2/5] Generating labels for 5 risk-scoring models...")
    labeled_df = add_labels(raw_df, seed=7)
    label_cols = [c for c in labeled_df.columns if c.startswith("label_")]
    for c in label_cols:
        print(f"    -> {c}: positive rate = {labeled_df[c].mean():.3f}")

    # 3. Build the versioned, point-in-time correct feature store
    print("\n[3/5] Writing standardized features into the versioned feature store...")
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR)
    store, info = build_feature_store(raw_df, STORE_DIR)
    print(f"    -> table '{info['table']}': v{info['v1_version']} (partial-year, "
          f"event_time={info['v1_event_time']:.0f}) and v{info['v2_version']} "
          f"(full-year, event_time={info['v2_event_time']:.0f})")

    # Point-in-time correctness sanity check: querying strictly between the
    # two event_times must return v1, never v2.
    midpoint = (info["v1_event_time"] + info["v2_event_time"]) / 2
    pit_rows = store.get_features_as_of(info["table"], as_of=midpoint)
    v1_rows = store.get_features(info["table"], version=info["v1_version"])
    v2_rows = store.get_features(info["table"], version=info["v2_version"])
    pit_matches_v1 = pit_rows == v1_rows
    pit_matches_v2 = pit_rows == v2_rows
    print(f"    -> point-in-time check @ midpoint: matches v1={pit_matches_v1}, "
          f"matches v2={pit_matches_v2} (expected True/False)")
    assert pit_matches_v1 and not pit_matches_v2, "Point-in-time retrieval leaked a future version!"

    # 4. Train all 5 models against the FULL (v2) feature snapshot
    print("\n[4/5] Training 5 models against the shared feature store (v2)...")
    training_features = get_training_frame(store, version=info["v2_version"])
    tracker = ExperimentTracker(MLRUNS_DIR)
    results = train_all_models(training_features, labeled_df, tracker)

    print("\n[5/5] Held-out evaluation results:")
    print(f"{'model':30s} {'algo':20s} {'#feat':>6s} {'roc_auc':>8s} {'pr_auc':>8s} {'f1':>6s} {'acc':>6s}")
    for r in results:
        print(f"{r['model_name']:30s} {r['algorithm']:20s} {r['n_features']:6d} "
              f"{r['roc_auc']:8.3f} {r['pr_auc']:8.3f} {r['f1']:6.3f} {r['accuracy']:6.3f}")

    tracker.close()
    store.close()

    summary = {
        "n_members": len(raw_df),
        "n_features": len(FEATURE_COLUMNS),
        "point_in_time_check_passed": bool(pit_matches_v1 and not pit_matches_v2),
        "models": results,
    }
    out_path = ROOT / "run_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {out_path}")
    return summary


if __name__ == "__main__":
    main()
