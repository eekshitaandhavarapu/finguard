"""Generate the labeled transaction population used by FinGuard.

The `is_anomaly` and `anomaly_type` columns are ground truth only. Detection
phases must treat them as held-out evaluation fields and must not use them as
features or rules.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker


DEFAULT_ROWS = 30_000
DEFAULT_SEED = 42
DEFAULT_OUTPUT = Path("data/generated/transactions.csv")
START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)

DEPARTMENTS = ("Finance", "Ops", "Sales", "IT", "HR")
DEPARTMENT_ACCOUNT_CODES = {
    "Finance": ("FIN-100", "FIN-210", "FIN-310"),
    "Ops": ("OPS-120", "OPS-220", "OPS-420"),
    "Sales": ("SAL-130", "SAL-230", "SAL-330"),
    "IT": ("IT-140", "IT-240", "IT-340"),
    "HR": ("HR-150", "HR-250", "HR-350"),
}
APPROVAL_THRESHOLDS = {
    "Finance": 25_000.00,
    "Ops": 50_000.00,
    "Sales": 15_000.00,
    "IT": 35_000.00,
    "HR": 20_000.00,
}
ANOMALY_TYPES = (
    "duplicate_transaction",
    "round_number_amount",
    "threshold_evasion",
    "weekend_holiday_posting",
    "unusual_vendor_employee_pairing",
)
ANOMALIES_PER_TYPE = 270

# A small fixed holiday calendar keeps the dataset reproducible and avoids
# adding a holiday-library dependency just for the data-generation phase.
HOLIDAYS = {
    date(2025, 1, 1),
    date(2025, 1, 20),
    date(2025, 2, 17),
    date(2025, 5, 26),
    date(2025, 6, 19),
    date(2025, 7, 4),
    date(2025, 9, 1),
    date(2025, 10, 13),
    date(2025, 11, 11),
    date(2025, 11, 27),
    date(2025, 12, 25),
}


def _date_from_offset(offset: int) -> date:
    return START_DATE + timedelta(days=offset)


def _posting_time(rng: np.random.Generator) -> time:
    minutes_after_midnight = int(rng.integers(8 * 60, 18 * 60 + 1))
    return time(minutes_after_midnight // 60, minutes_after_midnight % 60)


def _valid_business_dates() -> list[date]:
    days = (END_DATE - START_DATE).days + 1
    return [
        _date_from_offset(offset)
        for offset in range(days)
        if _date_from_offset(offset).weekday() < 5
        and _date_from_offset(offset) not in HOLIDAYS
    ]


def _anomaly_dates(rng: np.random.Generator, count: int) -> list[date]:
    weekend_dates = [
        _date_from_offset(offset)
        for offset in range((END_DATE - START_DATE).days + 1)
        if _date_from_offset(offset).weekday() >= 5
    ]
    candidates = weekend_dates + sorted(HOLIDAYS)
    return [candidates[int(index)] for index in rng.integers(0, len(candidates), size=count)]


def _generate_base_rows(
    rows: int,
    rng: np.random.Generator,
    fake: Faker,
) -> pd.DataFrame:
    departments = rng.choice(DEPARTMENTS, size=rows)
    dates = rng.choice(_valid_business_dates(), size=rows)
    department_thresholds = np.array([APPROVAL_THRESHOLDS[department] for department in departments])

    # A log-normal distribution produces many everyday purchases and a smaller
    # long tail of large payments, which is more realistic than uniform values.
    amounts = np.clip(
        rng.lognormal(mean=np.log(2_500), sigma=1.05, size=rows),
        25.00,
        225_000.00,
    ).round(2)
    posting_times = [_posting_time(rng) for _ in range(rows)]
    vendors = [fake.company() for _ in range(rows)]
    employees = [f"EMP-{int(value):05d}" for value in rng.integers(1, 1_201, size=rows)]
    account_codes = [
        str(rng.choice(DEPARTMENT_ACCOUNT_CODES[department]))
        for department in departments
    ]

    statuses = np.where(
        amounts >= department_thresholds,
        rng.choice(["approved", "pending"], size=rows, p=[0.94, 0.06]),
        rng.choice(["approved", "pending", "rejected"], size=rows, p=[0.91, 0.07, 0.02]),
    )

    return pd.DataFrame(
        {
            "transaction_id": np.arange(1, rows + 1, dtype=np.int64),
            "transaction_date": pd.to_datetime(dates).date,
            "posting_time": posting_times,
            "account_code": account_codes,
            "department": departments,
            "amount": amounts,
            "vendor_name": vendors,
            "employee_id": employees,
            "approval_status": statuses,
            "approval_threshold": department_thresholds.round(2),
            "is_anomaly": False,
            "anomaly_type": None,
        }
    )


def _mark_anomaly(
    frame: pd.DataFrame,
    indices: np.ndarray,
    anomaly_type: str,
) -> None:
    frame.loc[indices, "is_anomaly"] = True
    frame.loc[indices, "anomaly_type"] = anomaly_type


def _inject_anomalies(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    fake: Faker,
) -> None:
    anomaly_indices = rng.choice(frame.index.to_numpy(), size=ANOMALIES_PER_TYPE * len(ANOMALY_TYPES), replace=False)
    groups = np.array_split(anomaly_indices, len(ANOMALY_TYPES))

    # 1. Duplicate-like transactions: same vendor and amount as another row,
    # posted within one day of the source transaction.
    duplicate_indices = groups[0]
    source_indices = rng.choice(
        frame.index.difference(anomaly_indices).to_numpy(),
        size=len(duplicate_indices),
        replace=False,
    )
    for target_index, source_index in zip(duplicate_indices, source_indices):
        source_date = frame.at[source_index, "transaction_date"]
        frame.at[target_index, "vendor_name"] = frame.at[source_index, "vendor_name"]
        frame.at[target_index, "amount"] = frame.at[source_index, "amount"]
        frame.at[target_index, "transaction_date"] = source_date + timedelta(
            days=int(rng.choice([-1, 0, 1]))
        )
    _mark_anomaly(frame, duplicate_indices, "duplicate_transaction")

    # 2. Exact round numbers, deliberately independent from approval thresholds.
    round_indices = groups[1]
    frame.loc[round_indices, "amount"] = rng.integers(1, 21, size=len(round_indices)) * 10_000
    _mark_anomaly(frame, round_indices, "round_number_amount")

    # 3. Values just below the configured approval boundary.
    threshold_indices = groups[2]
    frame.loc[threshold_indices, "amount"] = (
        frame.loc[threshold_indices, "approval_threshold"]
        * rng.uniform(0.95, 0.995, size=len(threshold_indices))
    ).round(2)
    frame.loc[threshold_indices, "approval_status"] = "approved"
    _mark_anomaly(frame, threshold_indices, "threshold_evasion")

    # 4. Weekend or holiday posting dates; keep normal business-hour times so
    # the date itself is the suspicious signal.
    calendar_indices = groups[3]
    frame.loc[calendar_indices, "transaction_date"] = _anomaly_dates(
        rng, len(calendar_indices)
    )
    _mark_anomaly(frame, calendar_indices, "weekend_holiday_posting")

    # 5. Unique vendors paired with a high-value employee transaction.
    unusual_indices = groups[4]
    unique_vendor_names = [
        f"{fake.company()} One-Time Services {int(index):04d}"
        for index in range(1, len(unusual_indices) + 1)
    ]
    frame.loc[unusual_indices, "vendor_name"] = unique_vendor_names
    frame.loc[unusual_indices, "amount"] = (
        frame.loc[unusual_indices, "approval_threshold"]
        * rng.uniform(1.25, 3.25, size=len(unusual_indices))
    ).round(2)
    frame.loc[unusual_indices, "approval_status"] = "pending"
    _mark_anomaly(frame, unusual_indices, "unusual_vendor_employee_pairing")


def generate_transactions(rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Return a reproducible transaction population with held-out labels."""
    if rows < ANOMALIES_PER_TYPE * len(ANOMALY_TYPES):
        raise ValueError(
            f"rows must be at least {ANOMALIES_PER_TYPE * len(ANOMALY_TYPES)} "
            "to inject the requested anomaly population"
        )

    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)
    frame = _generate_base_rows(rows, rng, fake)
    _inject_anomalies(frame, rng, fake)
    return frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    transactions = generate_transactions(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(args.output, index=False, date_format="%Y-%m-%d")

    anomaly_counts = transactions.loc[transactions["is_anomaly"], "anomaly_type"].value_counts()
    print(f"Generated {len(transactions):,} transactions at {args.output}")
    print(f"Ground-truth anomalies: {int(transactions['is_anomaly'].sum()):,} ({transactions['is_anomaly'].mean():.1%})")
    print(anomaly_counts.to_string())


if __name__ == "__main__":
    main()