"""
pipeline.labels
================

Generates the 5 risk-scoring model labels from the raw simulated data.
Each label is a plausible, noisy function of a different (overlapping)
subset of the same underlying member state -- exactly the situation a
payer feature store is built for: 5 model teams each pull from the same
governed feature set but train against their own outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def add_labels(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()

    def z(col: str) -> np.ndarray:
        s = df[col].astype(float)
        std = s.std() if s.std() > 0 else 1.0
        return ((s - s.mean()) / std).to_numpy()

    # 1) Readmission risk (30-day readmission after an index admission)
    logit = (
        -2.4
        + 1.1 * z("inpatient_admits_12m")
        + 0.7 * z("chronic_condition_count")
        + 0.6 * df["chf_flag"] + 0.5 * df["copd_flag"]
        + 0.4 * z("er_visits_12m")
        + 0.5 * df["readmission_30d_count_prior"].clip(upper=3)
        - 0.3 * z("medication_adherence_pdc")
        + rng.normal(0, 0.6, len(df))
    )
    df["label_readmission_risk"] = (rng.random(len(df)) < _sigmoid(logit)).astype(int)

    # 2) High-cost-claimant risk (top decile of NEXT period spend, proxied)
    logit = (
        -2.7
        + 1.3 * z("total_paid_12m")
        + 0.6 * z("chronic_condition_count")
        + 0.5 * z("inpatient_admits_12m")
        + 0.3 * z("high_cost_claim_count")
        + 0.2 * df["cost_trend_ratio"].clip(upper=3)
        + rng.normal(0, 0.5, len(df))
    )
    df["label_high_cost_risk"] = (rng.random(len(df)) < _sigmoid(logit)).astype(int)

    # 3) Chronic-condition-progression risk (e.g. diabetic complication onset)
    logit = (
        -2.3
        + 0.9 * df["diabetes_flag"]
        + 0.6 * z("hba1c_last")
        + 0.5 * z("creatinine_last")
        - 0.4 * z("egfr_last")
        + 0.4 * z("charlson_comorbidity_index")
        - 0.5 * z("medication_adherence_pdc")
        + 0.3 * z("bmi")
        + rng.normal(0, 0.6, len(df))
    )
    df["label_progression_risk"] = (rng.random(len(df)) < _sigmoid(logit)).astype(int)

    # 4) Fraud-risk proxy (billing-pattern anomaly)
    logit = (
        -3.0
        + 1.6 * z("provider_outlier_score")
        + 0.9 * df["upcoding_flag_proxy"]
        + 0.5 * z("duplicate_claim_count")
        + 0.4 * z("billing_code_diversity")
        + 0.3 * z("out_of_network_claim_ratio")
        + rng.normal(0, 0.5, len(df))
    )
    df["label_fraud_risk"] = (rng.random(len(df)) < _sigmoid(logit)).astype(int)

    # 5) Care-gap risk (likely to miss a recommended preventive service)
    logit = (
        -1.6
        + 0.9 * z("days_since_last_pcp_visit")
        + 0.6 * z("overdue_screenings_count")
        + 0.5 * df["care_gap_diabetes_eye_exam"]
        + 0.4 * df["care_gap_mammogram"]
        + 0.4 * df["care_gap_colonoscopy"]
        + 0.4 * z("sdoh_risk_score")
        + 0.3 * df["transportation_barrier_flag"]
        - 0.4 * z("pcp_visits_12m")
        + rng.normal(0, 0.6, len(df))
    )
    df["label_care_gap_risk"] = (rng.random(len(df)) < _sigmoid(logit)).astype(int)

    return df
