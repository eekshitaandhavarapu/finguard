# FinGuard

FinGuard is a financial transaction anomaly detection platform that combines SQL rules, statistical analysis, and machine learning into an explainable risk dashboard.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string
- `python data/generate_synthetic_data.py` — generate the reproducible Phase 1 dataset
- `python data/load_data.py --replace` — create and load `public.transactions`

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `data/generate_synthetic_data.py` — synthetic transaction population and labeled anomaly injection
- `data/load_data.py` — PostgreSQL schema creation and bulk loading
- `data/generated/transactions.csv` — generated local dataset output (ignored by Git)
- `sql/transactions_schema.sql` — source-of-truth Phase 1 database schema
- `sql/rule_based_flags.sql` — source-of-truth SQL detection rules
- `analysis/run_sql_rules.py` — executes rules and rebuilds `sql_flags`
- `analysis/benford_analysis.py` — Benford goodness-of-fit analysis and transaction scores
- `analysis/anomaly_model.py` — Isolation Forest and PyOD LOF model scores
- `scoring/risk_scoring.py` — weighted composite risk score and held-out evaluation
- `artifacts/api-server` — shared API service for the later dashboard

## Architecture decisions

- Ground-truth anomaly labels are stored with transactions for final evaluation but are not used by detection logic.
- Phase 1 uses a reproducible NumPy/Faker seed so the dataset can be regenerated consistently.
- The Python loader uses PostgreSQL `COPY` for bulk loading rather than one INSERT per transaction.

## Product

FinGuard will let reviewers explore suspicious transactions, understand why they were flagged, and compare composite risk scores with held-out ground truth.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

_Populate as you build — sharp edges, "always run X before Y" rules._

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
