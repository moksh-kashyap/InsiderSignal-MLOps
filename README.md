# InsiderSignal-MLOps

![Pipeline Tests](https://github.com/moksh-kashyap/InsiderSignal-MLOps/actions/workflows/test.yml/badge.svg)

Production-grade MLOps layer for InsiderSignal — model registry, drift detection, automated retraining with safe promotion, Airflow orchestration, and CI/CD.

This is not a new project — it's a layer on top of InsiderSignal. It takes the pipeline from "I trained a model" to "I built a system that manages, monitors, and heals its own model over time."

## What's here

| Component | What it does | What it proves |
|---|---|---|
| Model Registry | Every trained model version is stored with metadata, never overwritten | Reproducibility & auditability matter in production |
| Drift Detection | Weekly statistical check comparing incoming data vs. training data (KS-test) | Models decay silently; you monitor for it |
| Automated Retraining | Pipeline retrains itself when drift crosses a threshold, evaluates the new model, and only promotes it if it is actually better | You understand safe deployment, not just training |
| CI/CD | Automated tests run on every push, against a real MySQL container | You write pipelines like software, not notebooks |

## Architecture

Daily ETL Pipeline → New feature rows land in MySQL → Drift Detector (weekly job) writes to `drift_reports` table.

- If `drift_score > threshold`: Trigger Retraining Job
- Else: Do nothing (log check)

Retraining Job: Train candidate model → Evaluate vs. current production model on the same fresh test set.

- If candidate is better: Promote to production, update version registry
- Else: Discard candidate, log reason

## The key design decision

A retrained model is only promoted if it beats the current production model **on the same held-out test set**, by more than a noise margin (+0.01 F1). This prevents a classic failure mode — a "retrain" that is actually just overfitting to a slightly different random split, silently downgrading production.

## Stack

- **Orchestration:** Apache Airflow (WSL2)
- **Database:** MySQL 8.0
- **CI/CD:** GitHub Actions with a real MySQL service container per run
- **Testing:** pytest with session-scoped fixtures, credentials from environment variables

## Running the tests locally

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Requires a local MySQL instance and a `MYSQL_ROOT_PASSWORD` environment variable set. Schema is in `schema.sql`.

## How to talk about this in an interview

"The model itself was not the hard part — training a classifier is standard. The harder problem was: how do I know when it stops being trustworthy? I added a weekly drift check comparing incoming feature distributions against training data using a KS-test, and if drift crosses a threshold, the pipeline automatically retrains — but it only promotes the new model if it beats the current one on a fair, held-out comparison. That last guardrail matters because a retrain is not automatically an improvement."

## Real debugging log

While standing this up, I hit and diagnosed:

- Airflow's webserver and scheduler dying silently when their terminal windows closed — fixed by running them detached with `nohup`
- A WSL2 networking gotcha: `localhost` inside WSL does not route to Windows, so a MySQL server running as a Windows service was unreachable until I found the correct `vEthernet (WSL)` gateway IP
- A GitHub Actions environment variable naming mismatch between the workflow and test fixtures
- `register_model()` assuming a `models/` directory that does not exist on a fresh CI machine — fixed with `os.makedirs(..., exist_ok=True)`

Each of these is a real, reproducible failure with a clear root cause, not a contrived "gotcha," which is exactly the kind of story worth telling in a systems-design interview.
