"""
pipeline.train
===============

Trains the 5 real risk-scoring models against the shared, versioned
feature store, tracks every run with the local experiment tracker, and
returns held-out evaluation metrics for each model.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline.features import FEATURE_COLUMNS
from tracking import ExperimentTracker

EXPERIMENT = "payer-risk-scoring"


@dataclass
class ModelSpec:
    name: str
    label_col: str
    algo: str  # "random_forest" | "gradient_boosting" | "logistic_regression"
    feature_subset: list[str]


def build_model_specs(all_features: list[str]) -> list[ModelSpec]:
    """5 real models, each pulling a different (overlapping) subset of the
    same shared, standardized feature table -- exactly how one feature
    store serves many model teams."""
    clinical = [f for f in all_features if f in {
        "age", "diabetes_flag", "chf_flag", "ckd_flag", "copd_flag", "cad_flag",
        "cancer_flag", "hypertension_flag", "chronic_condition_count",
        "charlson_comorbidity_index", "hba1c_last", "ldl_last", "egfr_last",
        "creatinine_last", "bmi", "systolic_bp", "glucose_last",
        "medication_adherence_pdc", "num_active_medications", "insulin_flag",
    }]
    utilization = [f for f in all_features if f in {
        "er_visits_12m", "inpatient_admits_12m", "inpatient_days_12m",
        "outpatient_visits_12m", "pcp_visits_12m", "specialist_visits_12m",
        "urgent_care_visits_12m", "readmission_30d_count_prior", "admits_prior_180d",
        "chronic_condition_count", "chf_flag", "copd_flag", "age",
        "medication_adherence_pdc", "no_show_rate",
    }]
    cost = [f for f in all_features if f in {
        "total_paid_12m", "total_paid_prior_year", "avg_claim_amount",
        "max_claim_amount", "pharmacy_paid_12m", "medical_paid_12m",
        "out_of_pocket_12m", "cost_trend_ratio", "high_cost_claim_count",
        "chronic_condition_count", "inpatient_admits_12m", "age",
    }]
    fraud = [f for f in all_features if f in {
        "claim_frequency_zscore", "billing_code_diversity", "duplicate_claim_count",
        "provider_outlier_score", "upcoding_flag_proxy", "out_of_network_claim_ratio",
        "num_distinct_providers_12m", "high_cost_claim_count", "denied_claims_count",
    }]
    care_gap = [f for f in all_features if f in {
        "days_since_last_pcp_visit", "days_since_last_annual_wellness",
        "overdue_screenings_count", "care_gap_diabetes_eye_exam",
        "care_gap_mammogram", "care_gap_colonoscopy", "sdoh_risk_score",
        "transportation_barrier_flag", "food_insecurity_flag", "pcp_visits_12m",
        "pcp_assigned_flag", "no_show_rate", "age", "sex_male",
    }]

    return [
        ModelSpec("readmission_risk", "label_readmission_risk", "gradient_boosting", utilization),
        ModelSpec("high_cost_claimant_risk", "label_high_cost_risk", "random_forest", cost),
        ModelSpec("chronic_progression_risk", "label_progression_risk", "gradient_boosting", clinical),
        ModelSpec("fraud_risk_proxy", "label_fraud_risk", "random_forest", fraud),
        ModelSpec("care_gap_risk", "label_care_gap_risk", "logistic_regression", care_gap),
    ]


def _make_estimator(algo: str, seed: int):
    if algo == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=seed, n_jobs=-1, class_weight="balanced_subsample",
        )
    if algo == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed,
        )
    if algo == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    raise ValueError(f"Unknown algo {algo}")


def train_all_models(
    feature_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    tracker: ExperimentTracker,
    seed: int = 123,
) -> list[dict]:
    """feature_df comes from the feature store (point-in-time correct
    version); labeled_df supplies the label_* outcome columns aligned by
    member_id. Trains, evaluates on a held-out split, and logs every run."""
    merged = feature_df.merge(
        labeled_df[["member_id"] + [c for c in labeled_df.columns if c.startswith("label_")]],
        on="member_id", how="inner",
    )

    specs = build_model_specs(FEATURE_COLUMNS)
    results = []
    for spec in specs:
        X = merged[spec.feature_subset].astype(float).values
        y = merged[spec.label_col].astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y,
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = _make_estimator(spec.algo, seed)
        with tracker.start_run(EXPERIMENT, spec.name) as run:
            run.log_params({
                "model_name": spec.name,
                "algorithm": spec.algo,
                "n_features": len(spec.feature_subset),
                "features": ",".join(spec.feature_subset),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "positive_rate_train": float(y_train.mean()),
                "positive_rate_test": float(y_test.mean()),
                "random_seed": seed,
            })
            model.fit(X_train_s, y_train)
            proba = model.predict_proba(X_test_s)[:, 1]
            pred = (proba >= 0.5).astype(int)

            metrics = {
                "roc_auc": roc_auc_score(y_test, proba),
                "pr_auc": average_precision_score(y_test, proba),
                "f1": f1_score(y_test, pred, zero_division=0),
                "accuracy": accuracy_score(y_test, pred),
            }
            run.log_metrics(metrics)

            results.append({
                "model_name": spec.name,
                "label": spec.label_col,
                "algorithm": spec.algo,
                "n_features": len(spec.feature_subset),
                "run_id": run.run_id,
                **metrics,
            })
    return results
