"""Create the Phase 1 table and bulk-load generated transactions into Postgres."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import psycopg


DEFAULT_INPUT = Path("data/generated/transactions.csv")
SCHEMA_PATH = Path("sql/transactions_schema.sql")
EXPECTED_COLUMNS = [
    "transaction_id",
    "transaction_date",
    "posting_time",
    "account_code",
    "department",
    "amount",
    "vendor_name",
    "employee_id",
    "approval_status",
    "approval_threshold",
    "is_anomaly",
    "anomaly_type",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the development table before loading the generated data.",
    )
    return parser.parse_args()


def _validate_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {', '.join(missing)}")
    if frame["transaction_id"].duplicated().any():
        raise ValueError("transaction_id values must be unique")
    if frame["amount"].le(0).any():
        raise ValueError("amount values must be greater than zero")


def _copy_rows(connection: psycopg.Connection, frame: pd.DataFrame) -> None:
    copy_sql = """
        COPY transactions (
            transaction_id, transaction_date, posting_time, account_code,
            department, amount, vendor_name, employee_id, approval_status,
            approval_threshold, is_anomaly, anomaly_type
        )
        FROM STDIN
    """
    with connection.cursor() as cursor:
        with cursor.copy(copy_sql) as copy:
            for row in frame[EXPECTED_COLUMNS].itertuples(index=False, name=None):
                copy.write_row(tuple(None if pd.isna(value) else value for value in row))


def main() -> None:
    args = _parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required to load transactions")
    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} does not exist. Run data/generate_synthetic_data.py first."
        )

    frame = pd.read_csv(
        args.input,
        parse_dates=["transaction_date"],
    )
    _validate_frame(frame)

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            if args.replace:
                cursor.execute("DROP TABLE IF EXISTS transactions")
            cursor.execute(schema_sql)
        _copy_rows(connection, frame)
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*), COUNT(*) FILTER (WHERE is_anomaly)
                FROM transactions
                """
            )
            total, anomalies = cursor.fetchone()

    print(f"Loaded {total:,} rows into public.transactions")
    print(f"Ground-truth anomaly labels present for evaluation: {anomalies:,}")


if __name__ == "__main__":
    main()