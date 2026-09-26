# Phase 1 data pipeline

Generate the reproducible transaction population:

```bash
python data/generate_synthetic_data.py
```

Load it into the development PostgreSQL database:

```bash
python data/load_data.py --replace
```

The loader reads `DATABASE_URL` from the environment and never hardcodes a
connection string. The `is_anomaly` and `anomaly_type` columns are ground-truth
labels reserved for evaluation in later phases; detection code must not use
them. Normal rows keep `anomaly_type` as SQL `NULL`.