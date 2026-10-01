"""
pipeline.features
==================

Builds the standardized, shared feature table that all 5 risk-scoring
models pull from, and writes it into the local feature store with real
version history so point-in-time correctness can be demonstrated and
tested.

We write TWO versions of the `member_risk_features` table:
  - v1: a "9-months-of-claims-run-out" snapshot (partial-year utilization
    and cost, as would exist mid-year before claims fully mature).
  - v2: the "full 12-month" snapshot used for actual model training.

v1's event_time is strictly earlier than v2's. This lets the pipeline
demonstrate and test the exact anti-leakage guarantee a feature store must
provide: a query `as_of` a time between v1 and v2 must return v1's values,
never v2's (which didn't exist yet).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_store import FeatureMetadata, FeatureStore

ENTITY_KEY = "member_id"
TABLE_NAME = "member_risk_features"

# Feature columns standardized across all 5 models (excludes member_id and
# label_* columns, which are outcome-specific, not shared features).
FEATURE_COLUMNS = [
    "age", "sex_male", "dual_eligible", "plan_type_hmo", "plan_tenure_months",
    "region_code", "rural_flag",
    "diabetes_flag", "chf_flag", "ckd_flag", "copd_flag", "cad_flag", "cancer_flag",
    "depression_flag", "obesity_flag", "hypertension_flag",
    "chronic_condition_count", "charlson_comorbidity_index",
    "er_visits_12m", "inpatient_admits_12m", "inpatient_days_12m",
    "outpatient_visits_12m", "pcp_visits_12m", "specialist_visits_12m",
    "urgent_care_visits_12m", "telehealth_visits_12m", "no_show_rate",
    "avg_days_between_visits", "readmission_30d_count_prior", "admits_prior_180d",
    "total_paid_12m", "total_paid_prior_year", "avg_claim_amount", "max_claim_amount",
    "pharmacy_paid_12m", "medical_paid_12m", "out_of_pocket_12m", "cost_trend_ratio",
    "high_cost_claim_count", "denied_claims_count",
    "num_active_medications", "opioid_rx_flag", "polypharmacy_flag",
    "medication_adherence_pdc", "statin_adherence_pdc", "insulin_flag",
    "anticoagulant_flag", "medication_changes_90d",
    "bmi", "systolic_bp", "glucose_last", "creatinine_last", "hba1c_last",
    "ldl_last", "egfr_last", "abnormal_lab_count_12m", "missed_labs_flag",
    "days_since_last_pcp_visit", "days_since_last_annual_wellness",
    "overdue_screenings_count", "care_gap_diabetes_eye_exam", "care_gap_mammogram",
    "care_gap_colonoscopy",
    "zip_poverty_rate", "sdoh_risk_score", "transportation_barrier_flag",
    "food_insecurity_flag",
    "pcp_assigned_flag", "num_distinct_providers_12m", "out_of_network_claim_ratio",
    "provider_specialist_count", "care_coordination_flag",
    "claim_frequency_zscore", "billing_code_diversity", "duplicate_claim_count",
    "provider_outlier_score", "upcoding_flag_proxy",
]

_INT_LIKE = {
    "age", "sex_male", "dual_eligible", "plan_type_hmo", "plan_tenure_months",
    "region_code", "rural_flag", "diabetes_flag", "chf_flag", "ckd_flag", "copd_flag",
    "cad_flag", "cancer_flag", "depression_flag", "obesity_flag", "hypertension_flag",
    "chronic_condition_count", "er_visits_12m", "inpatient_admits_12m",
    "inpatient_days_12m", "outpatient_visits_12m", "pcp_visits_12m",
    "specialist_visits_12m", "urgent_care_visits_12m", "telehealth_visits_12m",
    "readmission_30d_count_prior", "admits_prior_180d", "high_cost_claim_count",
    "denied_claims_count", "num_active_medications", "opioid_rx_flag",
    "polypharmacy_flag", "insulin_flag", "anticoagulant_flag", "medication_changes_90d",
    "abnormal_lab_count_12m", "missed_labs_flag", "overdue_screenings_count",
    "care_gap_diabetes_eye_exam", "care_gap_mammogram", "care_gap_colonoscopy",
    "transportation_barrier_flag", "food_insecurity_flag", "pcp_assigned_flag",
    "num_distinct_providers_12m", "provider_specialist_count", "care_coordination_flag",
    "billing_code_diversity", "duplicate_claim_count", "upcoding_flag_proxy",
}


def _schema() -> dict[str, str]:
    schema = {ENTITY_KEY: "int"}
    for c in FEATURE_COLUMNS:
        schema[c] = "int" if c in _INT_LIKE else "float"
    return schema


def build_feature_store(
    raw_df: pd.DataFrame, store_dir: str | Path, now: float | None = None
) -> tuple[FeatureStore, dict]:
    """Register the shared feature table and write two real, immutable
    versions (a partial-year v1 and the full-year v2) demonstrating
    versioning + point-in-time-correct retrieval."""
    now = now if now is not None else time.time()
    day = 86400.0
    v1_event_time = now - 90 * day   # snapshot taken 90 days ago (partial-year)
    v2_event_time = now - 1 * day    # yesterday's full-year refresh

    store = FeatureStore(store_dir)
    store.register_table(TABLE_NAME, ENTITY_KEY, _schema())

    for feat in FEATURE_COLUMNS:
        store.register_feature_metadata(
            FeatureMetadata(
                feature_name=feat,
                table_name=TABLE_NAME,
                owner="payer-risk-platform",
                description=f"Standardized member-level feature: {feat}",
                lineage="derived from simulated claims + clinical + pharmacy source tables",
                dtype="int" if feat in _INT_LIKE else "float",
            )
        )

    schema = _schema()
    cols = list(schema.keys())

    # v1: partial-year snapshot -- utilization/cost columns scaled down to
    # represent claims run-out incompleteness, everything else unchanged.
    rng = np.random.default_rng(1)
    partial_scale = 0.75
    v1_df = raw_df.copy()
    utilization_cost_cols = [
        c for c in FEATURE_COLUMNS
        if c.endswith("_12m") or "paid" in c or c in (
            "avg_claim_amount", "max_claim_amount", "out_of_pocket_12m",
            "high_cost_claim_count", "denied_claims_count",
        )
    ]
    for c in utilization_cost_cols:
        if c in v1_df.columns:
            v1_df[c] = v1_df[c] * partial_scale

    def _rows(df: pd.DataFrame) -> list[dict]:
        out = []
        for r in df[cols].itertuples(index=False):
            row = dict(zip(cols, r))
            for c in cols:
                if schema[c] == "int":
                    row[c] = int(round(row[c]))
                else:
                    row[c] = float(row[c])
            out.append(row)
        return out

    v1 = store.write_version(TABLE_NAME, _rows(v1_df), event_time=v1_event_time)
    v2 = store.write_version(TABLE_NAME, _rows(raw_df), event_time=v2_event_time)

    info = {
        "table": TABLE_NAME,
        "v1_version": v1,
        "v1_event_time": v1_event_time,
        "v2_version": v2,
        "v2_event_time": v2_event_time,
        "now": now,
        "n_features": len(FEATURE_COLUMNS),
    }
    return store, info


def get_training_frame(store: FeatureStore, version: int | None = None) -> pd.DataFrame:
    rows = store.get_features(TABLE_NAME, version=version)
    return pd.DataFrame(rows)
