-- FinGuard Phase 2: independent SQL rule-based anomaly detection.
--
-- Every rule returns exactly one column: transaction_id.
-- Ground-truth fields are deliberately absent from every query.
-- The Python runner combines each result with every transaction to materialize
-- both flagged=true and flagged=false rows in sql_flags.

-- rule: duplicate_transaction
WITH ordered_transactions AS (
    SELECT
        transaction_id,
        vendor_name,
        amount,
        transaction_date,
        posting_time,
        ROW_NUMBER() OVER (
            PARTITION BY vendor_name, amount
            ORDER BY transaction_date, posting_time, transaction_id
        ) AS vendor_amount_sequence
    FROM transactions
),
duplicate_pairs AS (
    SELECT
        first_transaction.transaction_id AS first_transaction_id,
        second_transaction.transaction_id AS second_transaction_id
    FROM ordered_transactions AS first_transaction
    INNER JOIN ordered_transactions AS second_transaction
        ON first_transaction.vendor_name = second_transaction.vendor_name
        AND first_transaction.amount = second_transaction.amount
        AND first_transaction.transaction_id < second_transaction.transaction_id
        AND ABS(first_transaction.transaction_date - second_transaction.transaction_date) <= 1
        AND first_transaction.vendor_amount_sequence <> second_transaction.vendor_amount_sequence
),
duplicate_transaction_ids AS (
    SELECT first_transaction_id AS transaction_id
    FROM duplicate_pairs
    UNION
    SELECT second_transaction_id AS transaction_id
    FROM duplicate_pairs
)
SELECT DISTINCT transaction_id
FROM duplicate_transaction_ids;

-- rule: round_number_amount
WITH amount_rules AS (
    SELECT
        transaction_id,
        amount
    FROM transactions
)
SELECT transaction_id
FROM amount_rules
WHERE MOD(amount, 10000) = 0;

-- rule: threshold_evasion
WITH approval_window AS (
    SELECT
        transaction_id,
        amount,
        approval_threshold,
        ROW_NUMBER() OVER (
            PARTITION BY department
            ORDER BY amount DESC, transaction_id
        ) AS department_amount_rank
    FROM transactions
)
SELECT transaction_id
FROM approval_window
WHERE amount < approval_threshold
  AND amount >= approval_threshold * 0.95;

-- rule: weekend_holiday_posting
WITH holiday_dates(transaction_date) AS (
    VALUES
        (DATE '2025-01-01'),
        (DATE '2025-01-20'),
        (DATE '2025-02-17'),
        (DATE '2025-05-26'),
        (DATE '2025-06-19'),
        (DATE '2025-07-04'),
        (DATE '2025-09-01'),
        (DATE '2025-10-13'),
        (DATE '2025-11-11'),
        (DATE '2025-11-27'),
        (DATE '2025-12-25')
),
calendar_exceptions AS (
    SELECT transaction_id, transaction_date
    FROM transactions
    WHERE EXTRACT(ISODOW FROM transaction_date) >= 6

    UNION

    SELECT transactions.transaction_id, transactions.transaction_date
    FROM transactions
    INNER JOIN holiday_dates
        ON holiday_dates.transaction_date = transactions.transaction_date
)
SELECT transaction_id
FROM calendar_exceptions;

-- rule: unusual_vendor_employee_pairing
WITH vendor_statistics AS (
    SELECT
        transaction_id,
        amount,
        approval_threshold,
        COUNT(*) OVER (PARTITION BY vendor_name) AS vendor_transaction_count,
        ROW_NUMBER() OVER (
            PARTITION BY vendor_name
            ORDER BY amount DESC, transaction_id
        ) AS vendor_amount_rank
    FROM transactions
)
SELECT transaction_id
FROM vendor_statistics
WHERE vendor_transaction_count = 1
  AND amount >= approval_threshold;