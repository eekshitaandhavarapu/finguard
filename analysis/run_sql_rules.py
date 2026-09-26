"""Run FinGuard's Phase 2 SQL rules and materialize their results."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import psycopg


DEFAULT_RULES_PATH = Path("sql/rule_based_flags.sql")
EXPECTED_RULES = (
    "duplicate_transaction",
    "round_number_amount",
    "threshold_evasion",
    "weekend_holiday_posting",
    "unusual_vendor_employee_pairing",
)
RULE_MARKER = re.compile(r"(?m)^--\s*rule:\s*([a-z_]+)\s*$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    return parser.parse_args()


def _load_rule_queries(path: Path) -> dict[str, str]:
    sql = path.read_text(encoding="utf-8")
    markers = list(RULE_MARKER.finditer(sql))
    if not markers:
        raise ValueError(f"No '-- rule: name' markers found in {path}")

    queries: dict[str, str] = {}
    for position, marker in enumerate(markers):
        name = marker.group(1)
        end = markers[position + 1].start() if position + 1 < len(markers) else len(sql)
        query = sql[marker.end() : end].strip()
        if name in queries:
            raise ValueError(f"Duplicate rule name: {name}")
        if not query:
            raise ValueError(f"Rule {name} has no SQL query")
        queries[name] = query

    if tuple(queries) != EXPECTED_RULES:
        raise ValueError(
            "Rule order or names do not match the expected set: "
            f"{', '.join(EXPECTED_RULES)}"
        )
    if re.search(r"\b(is_anomaly|anomaly_type)\b", sql, flags=re.IGNORECASE):
        raise ValueError("Detection rules must not reference ground-truth columns")
    return queries


def _copy_flags(
    connection: psycopg.Connection,
    transaction_ids: list[int],
    flagged_by_rule: dict[str, set[int]],
) -> None:
    copy_sql = """
        COPY sql_flags (transaction_id, flag_type, flagged)
        FROM STDIN
    """
    with connection.cursor() as cursor:
        with cursor.copy(copy_sql) as copy:
            for transaction_id in transaction_ids:
                for rule_name in EXPECTED_RULES:
                    copy.write_row(
                        (
                            transaction_id,
                            rule_name,
                            transaction_id in flagged_by_rule[rule_name],
                        )
                    )


def main() -> None:
    args = _parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required to run SQL rules")
    queries = _load_rule_queries(args.rules)

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sql_flags (
                    transaction_id BIGINT NOT NULL
                        REFERENCES transactions(transaction_id) ON DELETE CASCADE,
                    flag_type VARCHAR(50) NOT NULL,
                    flagged BOOLEAN NOT NULL,
                    PRIMARY KEY (transaction_id, flag_type)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sql_flags_flag_type
                    ON sql_flags (flag_type, flagged)
                """
            )
            cursor.execute(
                "SELECT transaction_id FROM transactions ORDER BY transaction_id"
            )
            transaction_ids = [row[0] for row in cursor.fetchall()]
            if not transaction_ids:
                raise RuntimeError("transactions is empty; run Phase 1 first")

            flagged_by_rule: dict[str, set[int]] = {}
            for rule_name, query in queries.items():
                cursor.execute(query)
                flagged_ids = {row[0] for row in cursor.fetchall()}
                unknown_ids = flagged_ids.difference(transaction_ids)
                if unknown_ids:
                    raise ValueError(
                        f"Rule {rule_name} returned unknown transaction IDs"
                    )
                flagged_by_rule[rule_name] = flagged_ids

            # sql_flags is a derived table. Rebuilding it makes reruns safe and
            # prevents stale flags after the transaction population changes.
            cursor.execute("DELETE FROM sql_flags")
            _copy_flags(connection, transaction_ids, flagged_by_rule)
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT flag_type, COUNT(*) FILTER (WHERE flagged) AS flagged_count
                FROM sql_flags
                GROUP BY flag_type
                ORDER BY flag_type
                """
            )
            rule_counts = cursor.fetchall()
            cursor.execute("SELECT COUNT(*) FROM sql_flags")
            total_flags = cursor.fetchone()[0]

    print(f"Materialized {total_flags:,} transaction-rule results in public.sql_flags")
    for flag_type, flagged_count in rule_counts:
        print(f"{flag_type}: {flagged_count:,} flagged")


if __name__ == "__main__":
    main()