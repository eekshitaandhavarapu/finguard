"""Run Benford's Law goodness-of-fit tests for accounts and departments."""

from __future__ import annotations

import argparse
import os
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from scipy.stats import chisquare, chi2


DEFAULT_ALPHA = 0.05
DEFAULT_GROUP_STATS_TABLE = "benford_group_stats"
DEFAULT_SCORE_TABLE = "benford_scores"
BENFORD_DIGITS = np.arange(1, 10)
BENFORD_PROBABILITIES = np.log10(1 + (1 / BENFORD_DIGITS))
BENFORD_CRITICAL_VALUE = float(chi2.ppf(0.95, df=8))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    return parser.parse_args()


def _leading_digit(amount: Decimal) -> int:
    normalized = format(abs(amount), "f").replace(".", "").lstrip("0")
    if not normalized:
        raise ValueError("Transaction amounts must be positive to extract a leading digit")
    return int(normalized[0])


def _group_statistics(
    frame: pd.DataFrame,
    group_level: str,
    group_column: str,
    alpha: float,
) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    for group_value, group in frame.groupby(group_column, sort=True):
        observed = np.bincount(
            group["leading_digit"].astype(int),
            minlength=10,
        )[1:]
        expected = len(group) * BENFORD_PROBABILITIES
        test = chisquare(f_obs=observed, f_exp=expected)
        deviation_score = min(1.0, float(test.statistic) / BENFORD_CRITICAL_VALUE)
        results.append(
            {
                "group_level": group_level,
                "group_value": str(group_value),
                "sample_size": int(len(group)),
                "chi_square_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "significant": bool(test.pvalue < alpha),
                "deviation_score": deviation_score,
            }
        )
    return pd.DataFrame(results)


def _copy_group_stats(connection: psycopg.Connection, stats: pd.DataFrame) -> None:
    with connection.cursor() as cursor:
        with cursor.copy(
            """
            COPY benford_group_stats (
                group_level, group_value, sample_size, chi_square_statistic,
                p_value, significant, deviation_score
            )
            FROM STDIN
            """
        ) as copy:
            for row in stats.itertuples(index=False, name=None):
                copy.write_row(row)


def _copy_scores(connection: psycopg.Connection, scores: pd.DataFrame) -> None:
    with connection.cursor() as cursor:
        with cursor.copy(
            """
            COPY benford_scores (
                transaction_id, leading_digit, account_code, department,
                account_chi_square_statistic, account_p_value, account_significant,
                account_deviation_score, department_chi_square_statistic,
                department_p_value, department_significant,
                department_deviation_score, benford_deviation_score
            )
            FROM STDIN
            """
        ) as copy:
            for row in scores.itertuples(index=False, name=None):
                copy.write_row(row)


def main() -> None:
    args = _parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required to run Benford analysis")

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT transaction_id, amount, account_code, department
                FROM transactions
                ORDER BY transaction_id
                """
            )
            rows = cursor.fetchall()

        if not rows:
            raise RuntimeError("transactions is empty; run Phase 1 first")

        frame = pd.DataFrame(
            rows,
            columns=["transaction_id", "amount", "account_code", "department"],
        )
        frame["leading_digit"] = frame["amount"].map(_leading_digit)

        account_stats = _group_statistics(
            frame, "account_code", "account_code", args.alpha
        )
        department_stats = _group_statistics(
            frame, "department", "department", args.alpha
        )
        group_stats = pd.concat([account_stats, department_stats], ignore_index=True)

        account_lookup = account_stats.set_index("group_value")
        department_lookup = department_stats.set_index("group_value")
        scores = frame[
            ["transaction_id", "leading_digit", "account_code", "department"]
        ].copy()
        scores["account_chi_square_statistic"] = scores["account_code"].map(
            account_lookup["chi_square_statistic"]
        )
        scores["account_p_value"] = scores["account_code"].map(
            account_lookup["p_value"]
        )
        scores["account_significant"] = scores["account_code"].map(
            account_lookup["significant"]
        )
        scores["account_deviation_score"] = scores["account_code"].map(
            account_lookup["deviation_score"]
        )
        scores["department_chi_square_statistic"] = scores["department"].map(
            department_lookup["chi_square_statistic"]
        )
        scores["department_p_value"] = scores["department"].map(
            department_lookup["p_value"]
        )
        scores["department_significant"] = scores["department"].map(
            department_lookup["significant"]
        )
        scores["department_deviation_score"] = scores["department"].map(
            department_lookup["deviation_score"]
        )
        scores["benford_deviation_score"] = scores[
            ["account_deviation_score", "department_deviation_score"]
        ].max(axis=1)

        # Convert NumPy/Pandas scalar values before psycopg COPY so the script
        # works consistently across psycopg and pandas versions.
        stats_rows = group_stats[
            [
                "group_level",
                "group_value",
                "sample_size",
                "chi_square_statistic",
                "p_value",
                "significant",
                "deviation_score",
            ]
        ].copy()
        stats_rows["sample_size"] = stats_rows["sample_size"].map(int)
        stats_rows["chi_square_statistic"] = stats_rows[
            "chi_square_statistic"
        ].map(float)
        stats_rows["p_value"] = stats_rows["p_value"].map(float)
        stats_rows["significant"] = stats_rows["significant"].map(bool)
        stats_rows["deviation_score"] = stats_rows["deviation_score"].map(float)

        score_rows = scores[
            [
                "transaction_id",
                "leading_digit",
                "account_code",
                "department",
                "account_chi_square_statistic",
                "account_p_value",
                "account_significant",
                "account_deviation_score",
                "department_chi_square_statistic",
                "department_p_value",
                "department_significant",
                "department_deviation_score",
                "benford_deviation_score",
            ]
        ].copy()
        score_rows["transaction_id"] = score_rows["transaction_id"].map(int)
        score_rows["leading_digit"] = score_rows["leading_digit"].map(int)
        for column in (
            "account_chi_square_statistic",
            "account_p_value",
            "account_deviation_score",
            "department_chi_square_statistic",
            "department_p_value",
            "department_deviation_score",
            "benford_deviation_score",
        ):
            score_rows[column] = score_rows[column].map(float)
        for column in ("account_significant", "department_significant"):
            score_rows[column] = score_rows[column].map(bool)

        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS benford_group_stats (
                    group_level VARCHAR(30) NOT NULL,
                    group_value VARCHAR(80) NOT NULL,
                    sample_size INTEGER NOT NULL CHECK (sample_size > 0),
                    chi_square_statistic DOUBLE PRECISION NOT NULL,
                    p_value DOUBLE PRECISION NOT NULL CHECK (p_value >= 0 AND p_value <= 1),
                    significant BOOLEAN NOT NULL,
                    deviation_score DOUBLE PRECISION NOT NULL
                        CHECK (deviation_score >= 0 AND deviation_score <= 1),
                    PRIMARY KEY (group_level, group_value)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS benford_scores (
                    transaction_id BIGINT PRIMARY KEY
                        REFERENCES transactions(transaction_id) ON DELETE CASCADE,
                    leading_digit SMALLINT NOT NULL CHECK (leading_digit BETWEEN 1 AND 9),
                    account_code VARCHAR(20) NOT NULL,
                    department VARCHAR(20) NOT NULL,
                    account_chi_square_statistic DOUBLE PRECISION NOT NULL,
                    account_p_value DOUBLE PRECISION NOT NULL,
                    account_significant BOOLEAN NOT NULL,
                    account_deviation_score DOUBLE PRECISION NOT NULL,
                    department_chi_square_statistic DOUBLE PRECISION NOT NULL,
                    department_p_value DOUBLE PRECISION NOT NULL,
                    department_significant BOOLEAN NOT NULL,
                    department_deviation_score DOUBLE PRECISION NOT NULL,
                    benford_deviation_score DOUBLE PRECISION NOT NULL
                        CHECK (benford_deviation_score >= 0 AND benford_deviation_score <= 1)
                )
                """
            )
            cursor.execute("DELETE FROM benford_group_stats")
            cursor.execute("DELETE FROM benford_scores")
        _copy_group_stats(connection, stats_rows)
        _copy_scores(connection, score_rows)
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT group_level, COUNT(*), COUNT(*) FILTER (WHERE significant)
                FROM benford_group_stats
                GROUP BY group_level
                ORDER BY group_level
                """
            )
            group_counts = cursor.fetchall()

    print(
        f"Stored {len(score_rows):,} per-transaction Benford scores in "
        "public.benford_scores"
    )
    for group_level, total_groups, significant_groups in group_counts:
        print(
            f"{group_level}: {total_groups} groups tested, "
            f"{significant_groups} significant at p < {args.alpha}"
        )


if __name__ == "__main__":
    main()