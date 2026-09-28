"""FinGuard Streamlit dashboard for transaction risk review."""

from __future__ import annotations

import os
import pandas as pd
import psycopg
import streamlit as st
from sklearn.metrics import precision_score, recall_score, roc_auc_score


PAGE_TITLE = "FinGuard | Transaction Risk Dashboard"
RISK_LEVELS = ["Low", "Medium", "High"]
HIGH_RISK_THRESHOLD = 45.0


st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data(ttl=60, show_spinner=False)
def load_risk_scores() -> pd.DataFrame:
    """Load the materialized scores and ground truth for evaluation only."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured for this dashboard")

    query = """
        SELECT
            risk_scores.transaction_id,
            risk_scores.transaction_date,
            risk_scores.posting_time,
            risk_scores.account_code,
            risk_scores.department,
            risk_scores.amount,
            risk_scores.vendor_name,
            risk_scores.employee_id,
            risk_scores.sql_flag_count,
            risk_scores.sql_rule_score,
            risk_scores.benford_deviation_score,
            risk_scores.isolation_forest_score,
            risk_scores.lof_score,
            risk_scores.ml_anomaly_score,
            risk_scores.risk_score,
            risk_scores.risk_level,
            risk_scores.flag_reasons,
            transactions.is_anomaly
        FROM risk_scores
        INNER JOIN transactions
            ON transactions.transaction_id = risk_scores.transaction_id
        ORDER BY risk_scores.transaction_date, risk_scores.transaction_id
    """
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]
        frame = pd.DataFrame(rows, columns=columns)

    if frame.empty:
        raise RuntimeError(
            "risk_scores is empty. Run the Phase 5 scoring pipeline before opening the dashboard."
        )

    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"])
    frame["amount"] = frame["amount"].astype(float)
    frame["flag_reasons"] = frame["flag_reasons"].fillna("")
    return frame


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def display_table(frame: pd.DataFrame, *, height: int = 420) -> None:
    """Render a consistently formatted transaction table."""
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        height=height,
        column_config={
            "transaction_id": st.column_config.NumberColumn("Transaction ID", format="%d"),
            "transaction_date": st.column_config.DateColumn("Date"),
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "risk_score": st.column_config.NumberColumn("Risk score", format="%.2f"),
            "sql_rule_score": st.column_config.NumberColumn("SQL rule score", format="%.2f"),
            "benford_deviation_score": st.column_config.NumberColumn(
                "Benford deviation", format="%.2f"
            ),
            "ml_anomaly_score": st.column_config.NumberColumn(
                "ML anomaly score", format="%.2f"
            ),
        },
    )


def render_overview(frame: pd.DataFrame) -> None:
    st.header("Overview")
    st.caption(
        "Portfolio-wide view of materialized composite risk scores. "
        "Flagged means Medium or High risk."
    )

    flagged = frame[frame["risk_level"].isin(["Medium", "High"])]
    total_value_at_risk = flagged["amount"].sum()
    flagged_percent = len(flagged) / len(frame) * 100

    total_metric, flagged_metric, value_metric = st.columns(3)
    total_metric.metric("Total transactions", f"{len(frame):,}")
    flagged_metric.metric("Risk flagged", f"{flagged_percent:.1f}%")
    value_metric.metric("Value at risk", format_currency(total_value_at_risk))

    distribution = (
        frame["risk_level"]
        .value_counts()
        .reindex(RISK_LEVELS, fill_value=0)
        .rename("Transactions")
        .to_frame()
    )
    distribution["Share"] = distribution["Transactions"] / len(frame) * 100

    chart_column, detail_column = st.columns([1.6, 1])
    with chart_column:
        st.subheader("Risk distribution")
        st.bar_chart(distribution["Transactions"], color="#2563eb")
    with detail_column:
        st.subheader("Risk mix")
        risk_mix = distribution.reset_index(names="Risk level")
        risk_mix["Share"] = risk_mix["Share"].map(format_percent)
        risk_mix["Transactions"] = risk_mix["Transactions"].map(lambda value: f"{value:,}")
        st.dataframe(risk_mix, hide_index=True, width="stretch")


def render_transaction_explorer(frame: pd.DataFrame) -> None:
    st.header("Transaction Explorer")
    st.caption(
        "Search and narrow the scored population by organization, employee, date, "
        "and risk level."
    )

    min_date = frame["transaction_date"].min().date()
    max_date = frame["transaction_date"].max().date()
    filter_column, employee_column, date_column, level_column = st.columns(4)

    departments = filter_column.multiselect(
        "Department",
        options=sorted(frame["department"].dropna().unique()),
        placeholder="All departments",
    )
    employees = employee_column.multiselect(
        "Employee",
        options=sorted(frame["employee_id"].dropna().unique()),
        placeholder="All employees",
    )
    date_range = date_column.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    selected_levels = level_column.multiselect(
        "Risk level",
        options=RISK_LEVELS,
        default=RISK_LEVELS,
    )
    search_text = st.text_input(
        "Search transactions",
        placeholder="Search by transaction ID, vendor, employee, department, or rule",
    ).strip().lower()

    filtered = frame.copy()
    if departments:
        filtered = filtered[filtered["department"].isin(departments)]
    if employees:
        filtered = filtered[filtered["employee_id"].isin(employees)]
    if selected_levels:
        filtered = filtered[filtered["risk_level"].isin(selected_levels)]
    else:
        filtered = filtered.iloc[0:0]

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            filtered["transaction_date"].dt.date.between(start_date, end_date)
        ]

    if search_text:
        searchable = (
            filtered["transaction_id"].astype(str)
            + " "
            + filtered["vendor_name"].astype(str)
            + " "
            + filtered["employee_id"].astype(str)
            + " "
            + filtered["department"].astype(str)
            + " "
            + filtered["flag_reasons"].astype(str)
        ).str.lower()
        filtered = filtered[searchable.str.contains(search_text, regex=False)]

    st.write(f"Showing **{len(filtered):,}** of **{len(frame):,}** transactions")
    explorer_columns = [
        "transaction_id",
        "transaction_date",
        "vendor_name",
        "department",
        "employee_id",
        "amount",
        "risk_score",
        "risk_level",
        "flag_reasons",
    ]
    display_table(
        filtered.sort_values(["risk_score", "transaction_id"], ascending=[False, True])[
            explorer_columns
        ].rename(
            columns={
                "vendor_name": "Vendor",
                "department": "Department",
                "employee_id": "Employee",
                "flag_reasons": "Rule reasons",
            }
        )
    )


def render_top_risk_transactions(frame: pd.DataFrame) -> None:
    st.header("Top Risk Transactions")
    st.caption(
        "Highest composite scores with the rule, Benford, and machine-learning "
        "signals that contributed to the ranking."
    )

    top_count = st.slider("Transactions to show", min_value=5, max_value=50, value=10, step=5)
    top_risks = frame.nlargest(top_count, "risk_score").copy()
    top_risks["Benford deviation"] = top_risks["benford_deviation_score"]
    top_risks["ML anomaly score"] = top_risks["ml_anomaly_score"]
    top_risks["SQL rules triggered"] = top_risks["flag_reasons"].replace("", "None")
    top_columns = [
        "transaction_id",
        "transaction_date",
        "vendor_name",
        "department",
        "employee_id",
        "amount",
        "risk_score",
        "risk_level",
        "SQL rules triggered",
        "Benford deviation",
        "ML anomaly score",
    ]
    display_table(
        top_risks[top_columns].rename(
            columns={
                "vendor_name": "Vendor",
                "department": "Department",
                "employee_id": "Employee",
            }
        ),
        height=460,
    )


def render_trends(frame: pd.DataFrame) -> None:
    st.header("Trends")
    st.caption("Daily count of transactions that reached Medium or High risk.")

    daily_flagged = (
        frame[frame["risk_level"].isin(["Medium", "High"])]
        .groupby("transaction_date")
        .size()
        .rename("Flagged transactions")
        .to_frame()
    )
    daily_flagged = daily_flagged.asfreq("D", fill_value=0)
    st.line_chart(daily_flagged, color="#dc2626")


def render_model_performance(frame: pd.DataFrame) -> None:
    st.header("Model Performance")
    st.caption(
        "Evaluation compares the High-risk bucket with the hidden is_anomaly label. "
        f"High risk is defined as a composite score of at least {HIGH_RISK_THRESHOLD:.0f}."
    )

    actual = frame["is_anomaly"].astype(int)
    predicted_high_risk = (frame["risk_level"] == "High").astype(int)
    precision = precision_score(actual, predicted_high_risk, zero_division=0)
    recall = recall_score(actual, predicted_high_risk, zero_division=0)
    roc_auc = roc_auc_score(actual, frame["risk_score"])

    precision_column, recall_column, auc_column = st.columns(3)
    precision_column.metric("Precision", f"{precision:.4f}")
    recall_column.metric("Recall", f"{recall:.4f}")
    auc_column.metric("ROC-AUC", f"{roc_auc:.4f}")

    comparison = pd.DataFrame(
        {
            "Count": [
                int(actual.sum()),
                int(predicted_high_risk.sum()),
                int((actual & predicted_high_risk.astype(bool)).sum()),
            ]
        },
        index=["Actual anomalies", "High-risk predictions", "Correct high-risk hits"],
    )
    st.bar_chart(comparison, color="#7c3aed")
    st.info(
        "Precision answers “when FinGuard says High risk, how often is it correct?” "
        "Recall answers “how many known anomalies reached High risk?” ROC-AUC evaluates "
        "the ranking quality of the continuous risk score across every threshold."
    )


def main() -> None:
    st.title("FinGuard")
    st.subheader("Transaction Risk Dashboard")
    st.caption(
        "A composite view of rule-based, Benford, and machine-learning signals "
        "for financial transaction review."
    )

    try:
        frame = load_risk_scores()
    except Exception as error:
        st.error(str(error))
        st.stop()

    render_overview(frame)
    st.divider()
    render_transaction_explorer(frame)
    st.divider()
    render_top_risk_transactions(frame)
    st.divider()
    render_trends(frame)
    st.divider()
    render_model_performance(frame)


if __name__ == "__main__":
    main()