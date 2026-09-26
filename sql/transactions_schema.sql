CREATE TABLE IF NOT EXISTS transactions (
    transaction_id BIGINT PRIMARY KEY,
    transaction_date DATE NOT NULL,
    posting_time TIME NOT NULL,
    account_code VARCHAR(20) NOT NULL,
    department VARCHAR(20) NOT NULL,
    amount NUMERIC(14, 2) NOT NULL CHECK (amount > 0),
    vendor_name VARCHAR(160) NOT NULL,
    employee_id VARCHAR(20) NOT NULL,
    approval_status VARCHAR(20) NOT NULL,
    approval_threshold NUMERIC(14, 2) NOT NULL CHECK (approval_threshold > 0),
    is_anomaly BOOLEAN NOT NULL DEFAULT FALSE,
    anomaly_type VARCHAR(40)
);

CREATE INDEX IF NOT EXISTS idx_transactions_date
    ON transactions (transaction_date);

CREATE INDEX IF NOT EXISTS idx_transactions_department
    ON transactions (department);

CREATE INDEX IF NOT EXISTS idx_transactions_vendor_amount
    ON transactions (vendor_name, amount);