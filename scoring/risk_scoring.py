"""Build composite FinGuard risk scores and evaluate them against ground truth."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import psycopg
from sklearn.metrics import precision_score, recall_score, roc_auc_score


SQL_RULE_WEIGHT = 0.4
BENFORD_WEIGHT = 0.3
ML_WEIGHT = 0.3
# Fixed score bands keep the buckets interpretable on the 0-100 composite
# scale; they are not fitted against the held-out labels.
HIGH_RISK_THRESHOLD = 45.0
MEDIUM_RISK_THRESHOLD = 35.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--high-risk-threshold",
        type=float,
        default=HIGH_RISK_THRESHOLD,
        help="Risk-score threshold used for precision and recall evaluation.",
    )
    parser.add_argument(
        "--medium-risk-threshold",
        type=float,
        default=MEDIUM_RISK_THRESHOLD,
        help="Lower bound for the Medium risk bucket.",
    )
    return parser.parse_args()


def _risk_level(score: float, medium_threshold: float, high_threshold: float) -> str:
    if score >= high_threshold:
        return "High"
    if score >= medium_threshold:
        return "Medium"
    return "Low"


def _copy_scores(connection: psycopg.Connection, scores: pd.DataFrame) -> None:
    with connection.cursor() as cursor:
        with cursor.copy(
            """
            COPY risk_scores (
                transaction_id, transaction_date, posting_time, account_code,
                department, amount, vendor_name, employee_id, sql_flag_count,
                sql_rule_score, benford_deviation_score,
                isolation_forest_score, lof_score, ml_anomaly_score,
                risk_score, risk_level, flag_reasons
            )
            FROM STDIN
            """
        ) as copy:
            for row in scores.itertuples(index=False, name=None):
                copy.write_row(row)


def _build_scores(
    frame: pd.DataFrame,
    medium_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["sql_flag_count"] = frame["sql_flag_count"].astype(int)
    frame["sql_rule_score"] = frame["sql_rule_score"].astype(float)
    frame["benford_deviation_score"] = frame["benford_deviation_score"].astype(float)
    frame["isolation_forest_score"] = frame["isolation_forest_score"].astype(float)
    frame["lof_score"] = frame["lof_score"].astype(float)
    frame["ml_anomaly_score"] = (
        frame["isolation_forest_score"] + frame["lof_score"]
    ) / 2

    frame["risk_score"] = (
        SQL_RULE_WEIGHT * frame["sql_rule_score"]
        + BENFORD_WEIGHT * frame["benford_deviation_score"]
        + ML_WEIGHT * frame["ml_anomaly_score"]
    ) * 100
    frame["risk_score"] = frame["risk_score"].clip(0, 100).round(2)
    frame["risk_level"] = frame["risk_score"].map(
        lambda score: _risk_level(score, medium_threshold, high_threshold)
    )
    frame["flag_reasons"] = frame["flag_reasons"].fillna("")
    return frame


def _print_evaluation(frame: pd.DataFrame, high_risk_threshold: float) -> None:
    actual = frame["is_anomaly"].astype(int)
    predicted_high_risk = (frame["risk_score"] >= high_risk_threshold).astype(int)
    precision = precision_score(actual, predicted_high_risk, zero_division=0)
    recall = recall_score(actual, predicted_high_risk, zero_division=0)
    roc_auc = roc_auc_score(actual, frame["risk_score"])

    print("\nFinGuard composite risk evaluation")
    print("----------------------------------")
    print(f"Transactions evaluated: {len(frame):,}")
    print(f"Actual anomalies: {int(actual.sum()):,}")
    print(
        f"High-risk predictions (score >= {high_risk_threshold:.2f}): "
        f"{int(predicted_high_risk.sum()):,}"
    )
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nRisk distribution:")
    print(frame["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).to_string())

    print(
        "\nPlain-language interpretation: precision is the share of high-risk "
        "transactions that are truly anomalous; recall is the share of known "
        "anomalies that reached the high-risk bucket; ROC-AUC measures how well "
        "the continuous score ranks anomalies above normal transactions."
    )


def main() -> None:
    args = _parse_args()
    if not 0 <= args.high_risk_threshold <= 100:
        raise ValueError("--high-risk-threshold must be between 0 and 100")
    if not 0 <= args.medium_risk_threshold < args.high_risk_threshold:
        raise ValueError(
            "--medium-risk-threshold must be at least 0 and below the high-risk threshold"
        )

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required to calculate risk scores")

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH sql_aggregation AS (
                    SELECT
                        transaction_id,
                        COUNT(*) FILTER (WHERE flagged)::INTEGER AS sql_flag_count,
                        COUNT(*)::DOUBLE PRECISION AS sql_rule_count,
                        STRING_AGG(flag_type, ', ' ORDER BY flag_type)
                            FILTER (WHERE flagged) AS flag_reasons
                    FROM sql_flags
                    GROUP BY transaction_id
                )
                SELECT
                    transactions.transaction_id,
                    transactions.transaction_date,
                    transactions.posting_time,
                    transactions.account_code,
                    transactions.department,
                    transactions.amount,
                    transactions.vendor_name,
                    transactions.employee_id,
                    sql_aggregation.sql_flag_count,
                    sql_aggregation.sql_flag_count
                        / NULLIF(sql_aggregation.sql_rule_count, 0)
                        AS sql_rule_score,
                    benford_scores.benford_deviation_score,
                    ml_scores.isolation_forest_score,
                    ml_scores.lof_score,
                    COALESCE(sql_aggregation.flag_reasons, '') AS flag_reasons,
                    transactions.is_anomaly
                FROM transactions
                INNER JOIN sql_aggregation
                    ON sql_aggregation.transaction_id = transactions.transaction_id
                INNER JOIN benford_scores
                    ON benford_scores.transaction_id = transactions.transaction_id
                INNER JOIN ml_scores
                    ON ml_scores.transaction_id = transactions.transaction_id
                ORDER BY transactions.transaction_id
                """
            )
            rows = cursor.fetchall()

        if not rows:
            raise RuntimeError(
                "No joined detection results found; run Phases 2, 3, and 4 first"
            )

        frame = pd.DataFrame(
            rows,
            columns=[
                "transaction_id",
                "transaction_date",
                "posting_time",
                "account_code",
                "department",
                "amount",
                "vendor_name",
                "employee_id",
                "sql_flag_count",
                "sql_rule_score",
                "benford_deviation_score",
                "isolation_forest_score",
                "lof_score",
                "flag_reasons",
                "is_anomaly",
            ],
        )
        scores = _build_scores(
            frame,
            medium_threshold=args.medium_risk_threshold,
            high_threshold=args.high_risk_threshold,
        )

        output_columns = [
            "transaction_id",
            "transaction_date",
            "posting_time",
            "account_code",
            "department",
            "amount",
            "vendor_name",
            "employee_id",
            "sql_flag_count",
            "sql_rule_score",
            "benford_deviation_score",
            "isolation_forest_score",
            "lof_score",
            "ml_anomaly_score",
            "risk_score",
            "risk_level",
            "flag_reasons",
        ]
        output = scores[output_columns].copy()
        output["transaction_id"] = output["transaction_id"].map(int)
        output["sql_flag_count"] = output["sql_flag_count"].map(int)
        for column in (
            "sql_rule_score",
            "benford_deviation_score",
            "isolation_forest_score",
            "lof_score",
            "ml_anomaly_score",
            "risk_score",
        ):
            output[column] = output[column].map(float)

        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_scores (
                    transaction_id BIGINT PRIMARY KEY
                        REFERENCES transactions(transaction_id) ON DELETE CASCADE,
                    transaction_date DATE NOT NULL,
                    posting_time TIME NOT NULL,
                    account_code VARCHAR(20) NOT NULL,
                    department VARCHAR(20) NOT NULL,
                    amount NUMERIC(14, 2) NOT NULL,
                    vendor_name VARCHAR(160) NOT NULL,
                    employee_id VARCHAR(20) NOT NULL,
                    sql_flag_count SMALLINT NOT NULL CHECK (sql_flag_count >= 0),
                    sql_rule_score DOUBLE PRECISION NOT NULL
                        CHECK (sql_rule_score >= 0 AND sql_rule_score <= 1),
                    benford_deviation_score DOUBLE PRECISION NOT NULL
                        CHECK (benford_deviation_score >= 0 AND benford_deviation_score <= 1),
                    isolation_forest_score DOUBLE PRECISION NOT NULL
                        CHECK (isolation_forest_score >= 0 AND isolation_forest_score <= 1),
                    lof_score DOUBLE PRECISION NOT NULL
                        CHECK (lof_score >= 0 AND lof_score <= 1),
                    ml_anomaly_score DOUBLE PRECISION NOT NULL
                        CHECK (ml_anomaly_score >= 0 AND ml_anomaly_score <= 1),
                    risk_score DOUBLE PRECISION NOT NULL
                        CHECK (risk_score >= 0 AND risk_score <= 100),
                    risk_level VARCHAR(10) NOT NULL
                        CHECK (risk_level IN ('Low', 'Medium', 'High')),
                    flag_reasons TEXT NOT NULL DEFAULT ''
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_risk_scores_level
                    ON risk_scores (risk_level, risk_score DESC)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_risk_scores_date
                    ON risk_scores (transaction_date)
                """
            )
            cursor.execute("DELETE FROM risk_scores")
        _copy_scores(connection, output)
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*),
                    COUNT(*) FILTER (WHERE risk_level = 'Low'),
                    COUNT(*) FILTER (WHERE risk_level = 'Medium'),
                    COUNT(*) FILTER (WHERE risk_level = 'High'),
                    MIN(risk_score),
                    MAX(risk_score)
                FROM risk_scores
                """
            )
            table_summary = cursor.fetchone()

    print(
        f"Stored {table_summary[0]:,} joined transaction scores in public.risk_scores"
    )
    print(
        f"Risk-score range: {table_summary[4]:.2f}–{table_summary[5]:.2f}; "
        f"Low={table_summary[1]:,}, Medium={table_summary[2]:,}, "
        f"High={table_summary[3]:,}"
    )
    _print_evaluation(scores, args.high_risk_threshold)


if __name__ == "__main__":
    main()