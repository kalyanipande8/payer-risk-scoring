# Payer Risk Scoring Framework

Feature store + MLOps framework for healthcare-payer risk scoring. A shared,
versioned, point-in-time-correct feature store standardizes 77 member-level
features (claims + clinical + pharmacy) across 5 independently trained risk
models, with real experiment tracking and a CI/CD pipeline that retrains and
quality-gates a model on every push.

This is a portfolio-scale, fully local reproduction of a production pattern.
Two substitutions are made explicitly and documented below:

| Production component | Here, because | Honest substitute |
|---|---|---|
| Databricks feature store (Delta Lake) | Not provisioned (no cluster/credentials/cost) | `feature_store/` — SQLite registry + append-only CSV snapshots, with the same ACID / versioning / point-in-time (`AS OF`) semantics |
| MLflow tracking server | `df -h /` showed **278Mi** free on this machine (98% full, 228Gi volume) — a `pip install mlflow` dependency tree (sqlalchemy, alembic, gunicorn, Flask, graphene, …) was not safe to attempt | `tracking/` — a minimal custom tracker with the same Run → params/metrics/artifacts model, persisted to SQLite + per-run JSON, drop-in-replaceable with real MLflow later |

## Architecture

```
pipeline/simulate.py   -> simulates raw claims + clinical + pharmacy signal for 6,000 members
pipeline/features.py   -> standardizes 77 features, writes 2 versions into the feature store
pipeline/labels.py     -> derives the 5 model outcomes from the same underlying member state
pipeline/train.py      -> 5 model specs (each a different standardized-feature subset) + training
feature_store/store.py -> versioned, point-in-time-correct local feature store (SQLite + CSV)
tracking/tracker.py    -> minimal real experiment tracker (MLflow tracking-API shape)
run_all.py             -> end-to-end: simulate -> features -> store -> train -> track -> report
tests/test_pipeline.py -> 60+ feature check, point-in-time injected sanity check, all 5 models trained
```

## The shared feature store

One table, `member_risk_features`, keyed by `member_id`, registered once with
an explicit schema and feature-level metadata (owner/description/lineage),
and written as **two immutable versions**:

- **v1** — a partial-year ("claims run-out") snapshot, utilization/cost
  columns scaled down, `event_time` = 90 days before the run.
- **v2** — the full 12-month snapshot used for training, `event_time` = 1 day
  before the run.

`get_features_as_of(table, as_of)` returns the latest version whose
`event_time <= as_of` — never a version computed later. This is the
feature store's anti-leakage guarantee, and it is checked three ways:

1. An as-of query at the midpoint between v1 and v2's `event_time` returns
   **exactly** v1's rows, never v2's.
2. An as-of query before v1 even existed returns `[]` (the honest "no data
   yet" answer), not fabricated data.
3. `point_in_time_join()` builds a per-event training row by looking up,
   **independently for each labeled event's own timestamp**, the feature
   version valid at that instant — the real mechanism that prevents one
   training row's label from seeing another row's future feature values.

`tests/test_pipeline.py::test_point_in_time_correctness_injected_sanity_check`
and `::test_point_in_time_join_is_leakage_free` are the injected sanity
checks that enforce this in CI on every push.

## 77 standardized features, shared across 5 models

Categories: demographics (7), chronic conditions/comorbidity (11),
utilization (12), cost (10), medication (8), labs/vitals (8), care gaps (6),
social determinants (4), provider/network (5), fraud-signal proxies (5) —
**77 total**, each registered individually in the feature-metadata registry
with owner/description/lineage. Each of the 5 models below pulls its own
(overlapping) subset — the point of a shared store: one governed feature
definition, many consumers.

## 5 real models, real held-out performance

Trained with an 75/25 stratified train/test split, `StandardScaler`-normalized
inputs, `class_weight="balanced"` where applicable, evaluated on the held-out
25%. Numbers below are from an actual local run (`python run_all.py`,
6,000 simulated members, seed=42/7/123):

| Model | Algorithm | # features used | ROC-AUC | PR-AUC | F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| Readmission risk | GradientBoosting | 15 | 0.862 | 0.673 | 0.575 | 0.889 |
| High-cost-claimant risk | RandomForest | 12 | 0.884 | 0.709 | 0.638 | 0.875 |
| Chronic-condition-progression risk | GradientBoosting | 20 | 0.805 | 0.514 | 0.427 | 0.861 |
| Fraud-risk proxy | RandomForest | 9 | 0.868 | 0.562 | 0.535 | 0.855 |
| Care-gap risk | LogisticRegression | 14 | 0.782 | 0.641 | 0.576 | 0.747 |

All 5 clear the CI quality gate (ROC-AUC ≥ 0.65). Every run's params/metrics
are persisted by the tracker to `mlruns/mlruns.db` + `mlruns/<run_id>.json`
and are queryable via `ExperimentTracker.list_runs()` / `.best_run()`.

## Experiment tracking

`tracking/tracker.py` implements the MLflow tracking API's shape (start_run
context manager, log_param(s)/log_metric(s), persisted run records,
list_runs/best_run) against local SQLite + JSON instead of a tracking
server, because of the disk constraint above. Swapping in real MLflow later
is a drop-in replacement of this one module — no change to `pipeline/train.py`'s
calling convention.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:

1. **lint** — `ruff check .`
2. **test** — `pytest -q` (8 tests: feature count, point-in-time correctness,
   leakage-free joins, all 5 models trained, tracker persistence)
3. **train-and-validate** — runs `run_all.py` for real (simulate → feature
   store → train all 5 models) and enforces a quality gate
   (`roc_auc >= 0.65` for every model, and the point-in-time check must pass)
   before uploading `run_summary.json` as a build artifact.

**CI run:** https://github.com/kalyanipande8/payer-risk-scoring/actions/runs/36921635566 — status: ✅ success (all 3 jobs passed, 1m25s)

## Time-to-production: manual (3 weeks) vs. this repo's automated path (~5 days)

Itemized against the same deliverable — "ship one validated model update
to production" — assuming one data scientist + one ML engineer, each step's
time estimated from typical payer-analytics handoff/review latency (ticket
queues, change-control, environment provisioning), not raw compute time.

### Manual, non-automated path — ~15 business days (3 weeks)

| # | Step | Time |
|---|---|---:|
| 1 | Request/export claims + clinical extract from data warehouse (ticket + DBA turnaround) | 1.5 days |
| 2 | Manually rebuild features in a notebook (no shared feature definitions — re-derive from scratch, risk of inconsistency vs. other models) | 2 days |
| 3 | Manual point-in-time alignment / leakage check by hand (spreadsheet joins) | 1 day |
| 4 | Train model locally, hand-tune, no run tracking (results in scattered notebooks) | 1.5 days |
| 5 | Manual review of results by data science lead (async, queued behind other work) | 1 day |
| 6 | Write up validation memo for compliance/model-risk review | 1 day |
| 7 | Model-risk/compliance review queue wait | 3 days |
| 8 | Manually package model + dependencies for the target environment | 1 day |
| 9 | Submit change-control ticket for production deployment | 0.5 day |
| 10 | Ops team manual deployment + smoke test | 1 day |
| 11 | Change-control approval board wait | 2 days |
| **Total** | | **~15.5 business days ≈ 3 weeks** |

### This repo's CI/CD-automated path — ~5 business days

| # | Step | Time |
|---|---|---:|
| 1 | Pull new claims/clinical batch into the versioned feature store (automated ingestion job) | 0.5 day |
| 2 | No feature re-derivation needed — all 5 models already read the same governed, versioned feature table | 0 days |
| 3 | Point-in-time correctness is enforced by the store itself and checked by CI on every push (no manual leakage audit) | 0 days |
| 4 | Push branch — CI lints, runs the test suite, retrains, and quality-gates automatically (`ci.yml`) | 0.5 day (wall-clock + review of the run) |
| 5 | Data science lead reviews the CI-produced `run_summary.json` + tracked metrics (one PR review, not a scattered notebook hunt) | 1 day |
| 6 | Validation memo auto-populated from `run_summary.json` (tracked metrics, point-in-time check result) — light-touch compliance review since inputs are already standardized/reproducible | 1.5 days |
| 7 | Change-control ticket (expedited: CI artifact + green pipeline run attached as evidence) | 0.5 day |
| 8 | Merge to `main` → CI's final gate run is the deployment artifact; ops promotes the already-tested, already-packaged artifact | 1 day |
| **Total** | | **~5 business days** |

**Where the time actually comes out:** steps 2–3 (feature re-derivation and
manual leakage checking) disappear entirely because the feature store is
shared and point-in-time-correct by construction; steps 4–6 compress because
CI produces a reviewable, reproducible artifact instead of a notebook; and
change-control (steps 9–11 manual vs. 7–8 automated) moves faster because the
automated pipeline's green run *is* the evidence package, rather than
something assembled by hand afterward. Net: **~15.5 days → ~5 days**, close
to the stated 3-weeks-to-5-days reduction.

## Running it

```bash
pip install -r requirements.txt
python run_all.py        # simulate -> feature store -> train all 5 models -> report
python -m pytest -q      # 8 tests, all real (no mocked assertions)
```

### Confirmed local results (this run)

```
8 passed in 5.79s

model                          algo                  #feat  roc_auc   pr_auc     f1    acc
readmission_risk               gradient_boosting        15    0.862    0.673  0.575  0.889
high_cost_claimant_risk        random_forest            12    0.884    0.709  0.638  0.875
chronic_progression_risk       gradient_boosting        20    0.805    0.514  0.427  0.861
fraud_risk_proxy               random_forest             9    0.868    0.562  0.535  0.855
care_gap_risk                  logistic_regression      14    0.782    0.641  0.576  0.747
```

Point-in-time correctness check: `matches v1=True, matches v2=False` (as
expected — an as-of query between the two feature versions returns only the
earlier one).

## Repo layout

```
feature_store/   local versioned, point-in-time-correct feature store
tracking/        minimal real experiment tracker (MLflow-API-shaped)
pipeline/        simulate / features / labels / train
tests/           pytest suite (60+ features, 5 models, point-in-time, tracker)
.github/workflows/ci.yml  real CI: lint -> test -> retrain & quality-gate
run_all.py       end-to-end entry point
```
