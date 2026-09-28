# FinGuard — Financial Transaction Anomaly Detection Platform

FinGuard is an explainable financial transaction anomaly detection platform. It combines deterministic business rules, statistical testing, and machine-learning models into one composite risk score that reviewers can investigate in a focused dashboard.

## Problem statement

Financial teams review large transaction populations under time pressure. Simple threshold checks catch obvious issues but miss unusual patterns, while black-box models can be difficult to explain during an audit or investigation. FinGuard addresses both problems by combining transparent rules with statistical and machine-learning signals, then preserving the reasons behind each risk score.

The project uses a reproducible synthetic dataset with hidden anomaly labels. Those labels are reserved for final evaluation and are not used by the detection logic.

## Architecture overview

FinGuard uses a three-layer detection system:

1. **SQL rule detection** — five explainable PostgreSQL rules identify duplicate transactions, round-number amounts, threshold evasion, weekend or holiday posting, and unusual vendor-employee pairings.
2. **Benford's Law statistical testing** — goodness-of-fit tests measure whether leading-digit distributions for accounts and departments deviate from expected Benford patterns.
3. **ML anomaly detection** — standardized transaction features are scored by Isolation Forest and PyOD Local Outlier Factor. Their normalized scores are averaged into one ML signal.

Phase 5 combines the signals into a 0–100 composite risk score:

- SQL rule evidence: 40%
- Benford deviation: 30%
- ML anomaly score: 30%

Scores are grouped into Low, Medium, and High risk levels. The dashboard exposes the component signals and triggered rule names so a reviewer can understand why a transaction ranked highly.

## Tech stack

- Python 3.11
- PostgreSQL
- Streamlit
- pandas and NumPy
- SciPy for Benford goodness-of-fit testing
- scikit-learn for Isolation Forest and evaluation metrics
- PyOD for Local Outlier Factor
- psycopg for PostgreSQL access
- Replit Autoscale deployment

## Key results

Phase 5 evaluated the composite score against the hidden anomaly labels:

- **Dataset:** 30,000 transactions
- **Known anomaly rate:** 4.5% (1,350 transactions)
- **Precision:** 0.8883
- **Recall:** 0.3711
- **ROC-AUC:** 0.9824

The High-risk bucket uses a composite score threshold of 45. Precision measures how often High-risk predictions are true anomalies. Recall measures how many known anomalies reach the High-risk bucket. ROC-AUC measures how well the continuous score ranks anomalous transactions above normal transactions across all thresholds.

## Dashboard

The Streamlit dashboard includes:

- Portfolio overview and value at risk
- Filterable transaction explorer
- Ranked highest-risk transactions with rule reasons
- Daily flagged-transaction trends
- Precision, recall, and ROC-AUC model performance cards

> **Screenshot placeholder:** Add a screenshot of the published FinGuard dashboard here after the deployment review.

## Live demo

> **Live demo:** Add the published Replit URL here after deployment.

## How to run locally

### 1. Set up Python and dependencies

Python 3.11 or newer and PostgreSQL are required.

```bash
git clone https://github.com/eekshitaandhavarapu/finguard.git
cd finguard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Configure the database

Set `DATABASE_URL` to a PostgreSQL connection string in your shell or local `.env` file. Never commit credentials.

```bash
export DATABASE_URL="<your-postgres-connection-string>"
```

### 3. Build the detection pipeline

Run the phases in order:

```bash
python data/generate_synthetic_data.py
python data/load_data.py --replace
python analysis/run_sql_rules.py
python analysis/benford_analysis.py
python analysis/anomaly_model.py
python scoring/risk_scoring.py
```

### 4. Start the dashboard

```bash
streamlit run app.py \
  --server.port 5000 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableWebsocketCompression false \
  --browser.gatherUsageStats false
```

Open `http://localhost:5000` in a browser.

## Repository layout

```text
data/       Synthetic data generation and PostgreSQL loading
sql/        Source SQL rules and transaction schema
analysis/   Rule, Benford, and ML detection phases
scoring/    Composite risk scoring and model evaluation
app.py      Streamlit transaction risk dashboard
```