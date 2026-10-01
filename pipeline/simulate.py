"""
pipeline.simulate
==================

Simulates raw payer claims + clinical data for a member population. This
stands in for the raw source tables (claims warehouse, EHR extract,
pharmacy feed) that a real feature-engineering pipeline would read from.

Everything is generated with a fixed RNG seed so the whole pipeline is
reproducible end to end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_MEMBERS_DEFAULT = 6000


def simulate_members(n_members: int = N_MEMBERS_DEFAULT, seed: int = 42) -> pd.DataFrame:
    """Simulate one row of raw claims/clinical signal per member, covering
    a trailing 12-month observation window. This is the raw input the
    feature-engineering step below standardizes into the shared feature
    store schema.
    """
    rng = np.random.default_rng(seed)
    n = n_members

    member_id = np.arange(1, n + 1)
    age = rng.integers(18, 90, size=n)
    sex_male = rng.integers(0, 2, size=n)
    dual_eligible = rng.binomial(1, 0.12, size=n)
    plan_type_hmo = rng.binomial(1, 0.55, size=n)
    plan_tenure_months = rng.integers(1, 180, size=n)
    region_code = rng.integers(0, 5, size=n)
    rural_flag = rng.binomial(1, 0.22, size=n)

    # Age-correlated chronic condition prevalence (older -> more comorbid)
    age_z = (age - age.mean()) / age.std()
    comorbid_propensity = 1 / (1 + np.exp(-(age_z * 1.1 - 1.0)))

    diabetes_flag = rng.binomial(1, np.clip(comorbid_propensity * 0.45, 0.02, 0.85))
    chf_flag = rng.binomial(1, np.clip(comorbid_propensity * 0.18, 0.01, 0.6))
    ckd_flag = rng.binomial(1, np.clip(comorbid_propensity * 0.16, 0.01, 0.55))
    copd_flag = rng.binomial(1, np.clip(comorbid_propensity * 0.15, 0.01, 0.5))
    cad_flag = rng.binomial(1, np.clip(comorbid_propensity * 0.2, 0.01, 0.6))
    cancer_flag = rng.binomial(1, np.clip(comorbid_propensity * 0.08, 0.005, 0.3))
    depression_flag = rng.binomial(1, 0.15 + 0.1 * dual_eligible)
    obesity_flag = rng.binomial(1, 0.30)
    hypertension_flag = rng.binomial(1, np.clip(comorbid_propensity * 0.55, 0.05, 0.9))

    chronic_condition_count = (
        diabetes_flag + chf_flag + ckd_flag + copd_flag + cad_flag
        + cancer_flag + depression_flag + obesity_flag + hypertension_flag
    )
    charlson_comorbidity_index = (
        chronic_condition_count + rng.poisson(0.4, size=n)
    ).astype(float)

    # Utilization -- driven by chronic burden + age + dual eligibility
    util_base = 0.5 + 0.35 * chronic_condition_count + 0.4 * dual_eligible + 0.15 * age_z.clip(0, None)
    er_visits_12m = rng.poisson(np.clip(util_base * 0.5, 0.05, None))
    inpatient_admits_12m = rng.poisson(np.clip(util_base * 0.18, 0.01, None))
    inpatient_days_12m = inpatient_admits_12m * rng.integers(1, 8, size=n)
    outpatient_visits_12m = rng.poisson(np.clip(util_base * 3.0, 0.2, None))
    pcp_visits_12m = rng.poisson(np.clip(2.0 + 0.4 * chronic_condition_count, 0.1, None))
    specialist_visits_12m = rng.poisson(np.clip(0.5 * chronic_condition_count, 0.05, None))
    urgent_care_visits_12m = rng.poisson(np.clip(util_base * 0.3, 0.02, None))
    telehealth_visits_12m = rng.poisson(1.2)
    no_show_rate = np.clip(rng.normal(0.08 + 0.05 * dual_eligible, 0.05, size=n), 0, 0.6)
    avg_days_between_visits = np.clip(365 / np.maximum(pcp_visits_12m + specialist_visits_12m, 1), 5, 365)
    readmission_30d_count_prior = rng.binomial(np.maximum(inpatient_admits_12m, 0), 0.15)
    admits_prior_180d = rng.poisson(np.clip(inpatient_admits_12m * 0.5, 0, None))

    # Cost -- correlated with utilization and chronic burden, right-skewed
    cost_base = (
        500
        + 4000 * chronic_condition_count
        + 8000 * inpatient_admits_12m
        + 300 * outpatient_visits_12m
        + 1500 * er_visits_12m
    )
    total_paid_12m = np.clip(rng.gamma(shape=2.0, scale=np.maximum(cost_base / 2, 50)), 100, None)
    total_paid_prior_year = np.clip(total_paid_12m * rng.normal(0.85, 0.25, size=n), 50, None)
    avg_claim_amount = total_paid_12m / np.maximum(outpatient_visits_12m + inpatient_admits_12m + er_visits_12m, 1)
    max_claim_amount = avg_claim_amount * rng.uniform(2, 12, size=n)
    pharmacy_paid_12m = np.clip(total_paid_12m * rng.uniform(0.05, 0.35, size=n), 0, None)
    medical_paid_12m = total_paid_12m - pharmacy_paid_12m
    out_of_pocket_12m = np.clip(total_paid_12m * rng.uniform(0.02, 0.15, size=n), 0, None)
    cost_trend_ratio = total_paid_12m / np.maximum(total_paid_prior_year, 1)
    high_cost_claim_count = rng.poisson(np.clip(total_paid_12m / 20000, 0, None))
    denied_claims_count = rng.poisson(0.3)

    # Medication
    num_active_medications = rng.poisson(np.clip(1 + 1.2 * chronic_condition_count, 0, None))
    opioid_rx_flag = rng.binomial(1, 0.08 + 0.05 * chf_flag)
    polypharmacy_flag = (num_active_medications >= 5).astype(int)
    medication_adherence_pdc = np.clip(rng.normal(0.78 - 0.05 * dual_eligible, 0.15, size=n), 0.1, 1.0)
    statin_adherence_pdc = np.clip(rng.normal(0.75, 0.18, size=n), 0.1, 1.0)
    insulin_flag = rng.binomial(1, np.clip(diabetes_flag * 0.35, 0, 1))
    anticoagulant_flag = rng.binomial(1, np.clip(0.1 * (chf_flag + cad_flag), 0, 1))
    medication_changes_90d = rng.poisson(0.5)

    # Labs / vitals
    bmi = np.clip(rng.normal(27 + 4 * obesity_flag, 5, size=n), 15, 60)
    systolic_bp = np.clip(rng.normal(122 + 12 * hypertension_flag, 14, size=n), 90, 210)
    glucose_last = np.clip(rng.normal(100 + 45 * diabetes_flag, 25, size=n), 60, 400)
    creatinine_last = np.clip(rng.normal(0.9 + 0.6 * ckd_flag, 0.3, size=n), 0.4, 6)
    hba1c_last = np.clip(rng.normal(5.6 + 2.0 * diabetes_flag, 1.0, size=n), 4.5, 14)
    ldl_last = np.clip(rng.normal(110 - 10 * (statin_adherence_pdc > 0.8), 30, size=n), 40, 260)
    egfr_last = np.clip(rng.normal(90 - 30 * ckd_flag, 18, size=n), 5, 130)
    abnormal_lab_count_12m = rng.poisson(np.clip(0.5 * chronic_condition_count, 0, None))
    missed_labs_flag = rng.binomial(1, 0.15 + 0.1 * (1 - medication_adherence_pdc > 0.3))

    # Care gaps
    days_since_last_pcp_visit = np.clip(rng.exponential(120 / np.maximum(pcp_visits_12m, 0.3)), 1, 720)
    days_since_last_annual_wellness = np.clip(rng.exponential(200), 1, 900)
    overdue_screenings_count = rng.poisson(np.clip(1.2 - pcp_visits_12m * 0.15, 0, None))
    care_gap_diabetes_eye_exam = ((diabetes_flag == 1) & (rng.random(n) < 0.35)).astype(int)
    care_gap_mammogram = ((sex_male == 0) & (age >= 40) & (rng.random(n) < 0.30)).astype(int)
    care_gap_colonoscopy = ((age >= 45) & (rng.random(n) < 0.40)).astype(int)

    # Social determinants proxies
    zip_poverty_rate = np.clip(rng.beta(2, 8, size=n), 0, 1)
    sdoh_risk_score = np.clip(zip_poverty_rate * 5 + dual_eligible * 1.5 + rng.normal(0, 0.5, size=n), 0, 10)
    transportation_barrier_flag = rng.binomial(1, np.clip(0.1 + 0.2 * rural_flag, 0, 1))
    food_insecurity_flag = rng.binomial(1, np.clip(0.05 + 0.25 * zip_poverty_rate, 0, 1))

    # Provider / network
    pcp_assigned_flag = rng.binomial(1, 0.88)
    num_distinct_providers_12m = rng.poisson(np.clip(1 + 0.6 * chronic_condition_count, 0, None)) + 1
    out_of_network_claim_ratio = np.clip(rng.beta(1, 12, size=n), 0, 1)
    provider_specialist_count = rng.poisson(np.clip(0.4 * chronic_condition_count, 0, None))
    care_coordination_flag = rng.binomial(1, np.clip(0.15 + 0.05 * chronic_condition_count, 0, 1))

    # Fraud-proxy signals (synthetic anomaly indicators)
    claim_frequency_zscore = rng.normal(0, 1, size=n)
    billing_code_diversity = rng.poisson(np.clip(2 + 0.3 * (outpatient_visits_12m), 0, None))
    duplicate_claim_count = rng.poisson(0.2)
    provider_outlier_score = np.clip(rng.beta(1.5, 10, size=n), 0, 1)
    # A small synthetic population of "billing anomaly" members drives the
    # upcoding proxy flag and fraud label downstream.
    anomaly_latent = (
        1.6 * provider_outlier_score
        + 0.8 * (duplicate_claim_count > 1)
        + 0.5 * (claim_frequency_zscore > 2)
        + 0.4 * (billing_code_diversity > 8)
        + rng.normal(0, 0.4, size=n)
    )
    upcoding_flag_proxy = (anomaly_latent > np.quantile(anomaly_latent, 0.9)).astype(int)

    df = pd.DataFrame({
        "member_id": member_id,
        "age": age,
        "sex_male": sex_male,
        "dual_eligible": dual_eligible,
        "plan_type_hmo": plan_type_hmo,
        "plan_tenure_months": plan_tenure_months,
        "region_code": region_code,
        "rural_flag": rural_flag,
        "diabetes_flag": diabetes_flag,
        "chf_flag": chf_flag,
        "ckd_flag": ckd_flag,
        "copd_flag": copd_flag,
        "cad_flag": cad_flag,
        "cancer_flag": cancer_flag,
        "depression_flag": depression_flag,
        "obesity_flag": obesity_flag,
        "hypertension_flag": hypertension_flag,
        "chronic_condition_count": chronic_condition_count,
        "charlson_comorbidity_index": charlson_comorbidity_index,
        "er_visits_12m": er_visits_12m,
        "inpatient_admits_12m": inpatient_admits_12m,
        "inpatient_days_12m": inpatient_days_12m,
        "outpatient_visits_12m": outpatient_visits_12m,
        "pcp_visits_12m": pcp_visits_12m,
        "specialist_visits_12m": specialist_visits_12m,
        "urgent_care_visits_12m": urgent_care_visits_12m,
        "telehealth_visits_12m": telehealth_visits_12m,
        "no_show_rate": no_show_rate,
        "avg_days_between_visits": avg_days_between_visits,
        "readmission_30d_count_prior": readmission_30d_count_prior,
        "admits_prior_180d": admits_prior_180d,
        "total_paid_12m": total_paid_12m,
        "total_paid_prior_year": total_paid_prior_year,
        "avg_claim_amount": avg_claim_amount,
        "max_claim_amount": max_claim_amount,
        "pharmacy_paid_12m": pharmacy_paid_12m,
        "medical_paid_12m": medical_paid_12m,
        "out_of_pocket_12m": out_of_pocket_12m,
        "cost_trend_ratio": cost_trend_ratio,
        "high_cost_claim_count": high_cost_claim_count,
        "denied_claims_count": denied_claims_count,
        "num_active_medications": num_active_medications,
        "opioid_rx_flag": opioid_rx_flag,
        "polypharmacy_flag": polypharmacy_flag,
        "medication_adherence_pdc": medication_adherence_pdc,
        "statin_adherence_pdc": statin_adherence_pdc,
        "insulin_flag": insulin_flag,
        "anticoagulant_flag": anticoagulant_flag,
        "medication_changes_90d": medication_changes_90d,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "glucose_last": glucose_last,
        "creatinine_last": creatinine_last,
        "hba1c_last": hba1c_last,
        "ldl_last": ldl_last,
        "egfr_last": egfr_last,
        "abnormal_lab_count_12m": abnormal_lab_count_12m,
        "missed_labs_flag": missed_labs_flag,
        "days_since_last_pcp_visit": days_since_last_pcp_visit,
        "days_since_last_annual_wellness": days_since_last_annual_wellness,
        "overdue_screenings_count": overdue_screenings_count,
        "care_gap_diabetes_eye_exam": care_gap_diabetes_eye_exam,
        "care_gap_mammogram": care_gap_mammogram,
        "care_gap_colonoscopy": care_gap_colonoscopy,
        "zip_poverty_rate": zip_poverty_rate,
        "sdoh_risk_score": sdoh_risk_score,
        "transportation_barrier_flag": transportation_barrier_flag,
        "food_insecurity_flag": food_insecurity_flag,
        "pcp_assigned_flag": pcp_assigned_flag,
        "num_distinct_providers_12m": num_distinct_providers_12m,
        "out_of_network_claim_ratio": out_of_network_claim_ratio,
        "provider_specialist_count": provider_specialist_count,
        "care_coordination_flag": care_coordination_flag,
        "claim_frequency_zscore": claim_frequency_zscore,
        "billing_code_diversity": billing_code_diversity,
        "duplicate_claim_count": duplicate_claim_count,
        "provider_outlier_score": provider_outlier_score,
        "upcoding_flag_proxy": upcoding_flag_proxy,
    })
    return df
