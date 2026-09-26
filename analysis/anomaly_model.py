"""Train Isolation Forest and PyOD LOF models for transaction anomalies."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import psycopg
from pyod.models.lof import LOF
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


DEFAULT_RANDOM_STATE = 42
DEFAULT_LOF_NEIGHBORS = 20
FEATURE_COLUMNS = (
    "amount_log",
    "day_of_week",
    "time_of_day",
    "department_average_deviation",
    "vendor_transaction_frequency",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--lof-neighbors", type=int, default=DEFAULT_LOF_NEIGHBORS)
    return parser.parse_args()


def _time_to_minutes(value: object) -> float:
    return float(value.hour * 60 + value.minute + value.second / 60)


def _minmax_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    minimum = values.min()
    maximum = values.max()
    if maximum == minimum:
        return np.zeros_like(values)
    return np.clip((values - minimum) / (maximum - minimum), 0, 1)


def _copy_scores(connection: psycopg.Connection, scores: pd.DataFrame) -> None:
    with connection.cursor() as cursor:
        with cursor.copy(
            """
            COPY ml_scores (
                transaction_id, amount_log, day_of_week, time_of_day,
                department_average_deviation, vendor_transaction_frequency,
                isolation_forest_score, lof_score
            )
            FROM STDIN
            """
        ) as copy:
            for row in scores.itertuples(index=False, name=None):
                copy.write_row(row)


def main() -> None:
    args = _parse_args()
    if args.lof_neighbors < 2:
        raise ValueError("--lof-neighbors must be at least 2")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required to train anomaly models")

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH vendor_frequency AS (
                    SELECT vendor_name, COUNT(*)::DOUBLE PRECISION AS transaction_frequency
                    FROM transactions
                    GROUP BY vendor_name
                ),
                department_deviation AS (
                    SELECT
                        department,
                        AVG(benford_deviation_score)::DOUBLE PRECISION
                            AS average_deviation
                    FROM benford_scores
                    GROUP BY department
                )
                SELECT
                    transactions.transaction_id,
                    transactions.amount,
                    transactions.transaction_date,
                    transactions.posting_time,
                    COALESCE(department_deviation.average_deviation, 0.0)
                        AS department_average_deviation,
                    vendor_frequency.transaction_frequency
                        AS vendor_transaction_frequency
                FROM transactions
                INNER JOIN vendor_frequency
                    ON vendor_frequency.vendor_name = transactions.vendor_name
                LEFT JOIN department_deviation
                    ON department_deviation.department = transactions.department
                ORDER BY transactions.transaction_id
                """
            )
            rows = cursor.fetchall()

        if not rows:
            raise RuntimeError("transactions is empty; run Phase 1 first")

        frame = pd.DataFrame(
            rows,
            columns=[
                "transaction_id",
                "amount",
                "transaction_date",
                "posting_time",
                "department_average_deviation",
                "vendor_transaction_frequency",
            ],
        )
        frame["amount_log"] = np.log1p(frame["amount"].astype(float))
        frame["day_of_week"] = pd.to_datetime(frame["transaction_date"]).dt.dayofweek
        frame["time_of_day"] = frame["posting_time"].map(_time_to_minutes) / (24 * 60)
        frame["department_average_deviation"] = frame[
            "department_average_deviation"
        ].astype(float)
        frame["vendor_transaction_frequency"] = frame[
            "vendor_transaction_frequency"
        ].astype(float)

        raw_features = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        scaled_features = StandardScaler().fit_transform(raw_features)

        isolation_forest = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=args.random_state,
            n_jobs=-1,
        )
        isolation_forest.fit(scaled_features)
        isolation_raw_scores = -isolation_forest.decision_function(scaled_features)

        neighbor_count = min(args.lof_neighbors, len(frame) - 1)
        lof = LOF(
            n_neighbors=neighbor_count,
            contamination=0.1,
            novelty=False,
        )
        lof.fit(scaled_features)
        lof_raw_scores = lof.decision_scores_

        scores = frame[
            [
                "transaction_id",
                "amount_log",
                "day_of_week",
                "time_of_day",
                "department_average_deviation",
                "vendor_transaction_frequency",
            ]
        ].copy()
        scores["isolation_forest_score"] = _minmax_score(isolation_raw_scores)
        scores["lof_score"] = _minmax_score(lof_raw_scores)

        scores["transaction_id"] = scores["transaction_id"].map(int)
        scores["amount_log"] = scores["amount_log"].map(float)
        scores["day_of_week"] = scores["day_of_week"].map(int)
        for column in (
            "time_of_day",
            "department_average_deviation",
            "vendor_transaction_frequency",
            "isolation_forest_score",
            "lof_score",
        ):
            scores[column] = scores[column].map(float)

        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_scores (
                    transaction_id BIGINT PRIMARY KEY
                        REFERENCES transactions(transaction_id) ON DELETE CASCADE,
                    amount_log DOUBLE PRECISION NOT NULL,
                    day_of_week SMALLINT NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
                    time_of_day DOUBLE PRECISION NOT NULL
                        CHECK (time_of_day >= 0 AND time_of_day <= 1),
                    department_average_deviation DOUBLE PRECISION NOT NULL,
                    vendor_transaction_frequency DOUBLE PRECISION NOT NULL
                        CHECK (vendor_transaction_frequency >= 1),
                    isolation_forest_score DOUBLE PRECISION NOT NULL
                        CHECK (isolation_forest_score >= 0 AND isolation_forest_score <= 1),
                    lof_score DOUBLE PRECISION NOT NULL
                        CHECK (lof_score >= 0 AND lof_score <= 1)
                )
                """
            )
            cursor.execute("DELETE FROM ml_scores")
        _copy_scores(connection, scores)
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*),
                    MIN(isolation_forest_score),
                    MAX(isolation_forest_score),
                    MIN(lof_score),
                    MAX(lof_score)
                FROM ml_scores
                """
            )
            summary = cursor.fetchone()

    print(
        f"Stored {summary[0]:,} model scores in public.ml_scores "
        f"(Isolation Forest range {summary[1]:.4f}-{summary[2]:.4f}; "
        f"LOF range {summary[3]:.4f}-{summary[4]:.4f})"
    )


if __name__ == "__main__":
    main()