
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "customer_churn_sample.csv"

st.title("Customer Churn & Revenue Analytics")
st.caption(
    "EdVyro Data Analytics Internship | Task 3: Business KPI Dashboard"
)

# --------------------------------------------------
# 2. Load and validate the dataset
# --------------------------------------------------
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    required_columns = [
        "CustomerID",
        "Gender",
        "Age",
        "TenureMonths",
        "SubscriptionType",
        "MonthlyCharges",
        "TotalCharges",
        "ContractType",
        "SupportTickets",
        "PaymentMethod",
        "Churn"
    ]

    missing_columns = [
        col for col in required_columns if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data[required_columns].copy()

    # Convert numeric columns safely
    numeric_columns = [
        "Age",
        "TenureMonths",
        "MonthlyCharges",
        "TotalCharges",
        "SupportTickets"
    ]

    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Standardize text values
    text_columns = [
        "CustomerID",
        "Gender",
        "SubscriptionType",
        "ContractType",
        "PaymentMethod",
        "Churn"
    ]

    for col in text_columns:
        data[col] = data[col].astype("string").str.strip()

    data["Churn"] = data["Churn"].str.title()

    # Basic data-quality checks
    data = data.drop_duplicates(subset=["CustomerID"])

    invalid_churn = ~data["Churn"].isin(["Yes", "No"])
    data.loc[invalid_churn, "Churn"] = pd.NA

    return data



# Load and validate the dataset
if not DATA_FILE.exists():
    st.error(
        f"Dataset not found: {DATA_FILE}"
    )
    st.stop()

try:
    df = load_data(DATA_FILE)
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()

# Check the dataset only after loading it
if df.empty:
    st.error("The dataset contains no customer records.")
    st.stop()

# --------------------------------------------------
# 3. Sidebar filters
# --------------------------------------------------
st.sidebar.header("Dashboard Filters")

contract_options = sorted(
    df["ContractType"].dropna().unique().tolist()
)
payment_options = sorted(
    df["PaymentMethod"].dropna().unique().tolist()
)

selected_contracts = st.sidebar.multiselect(
    "Contract Type",
    options=contract_options,
    default=contract_options
)

selected_payments = st.sidebar.multiselect(
    "Payment Method",
    options=payment_options,
    default=payment_options
)

valid_tenure = df["TenureMonths"].dropna()

if valid_tenure.empty:
    st.error("No valid tenure values are available.")
    st.stop()

min_tenure = int(valid_tenure.min())
max_tenure = int(valid_tenure.max())

tenure_range = st.sidebar.slider(
    "Customer Tenure (Months)",
    min_value=min_tenure,
    max_value=max_tenure,
    value=(min_tenure, max_tenure)
)

filtered_df = df[
    df["ContractType"].isin(selected_contracts)
    & df["PaymentMethod"].isin(selected_payments)
    & df["TenureMonths"].between(
        tenure_range[0], tenure_range[1]
    )
    & df["Churn"].isin(["Yes", "No"])
].copy()

st.caption(
    f"Showing {len(filtered_df)} of {len(df)} customer records"
)

if filtered_df.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

# --------------------------------------------------
# 4. Calculate KPIs
# --------------------------------------------------
total_customers = filtered_df["CustomerID"].nunique()

churned = filtered_df[filtered_df["Churn"] == "Yes"]
retained = filtered_df[filtered_df["Churn"] == "No"]

churn_count = churned["CustomerID"].nunique()
retained_count = retained["CustomerID"].nunique()

churn_rate = churn_count / total_customers * 100
retention_rate = retained_count / total_customers * 100

# Monthly charges associated with customers who churned.
# This is not a forecast of future revenue loss.
monthly_revenue_lost = churned["MonthlyCharges"].sum()

average_tenure = filtered_df["TenureMonths"].mean()

# --------------------------------------------------
# 5. KPI cards
# --------------------------------------------------
st.subheader("Business Performance KPIs")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Churn Rate",
    f"{churn_rate:.1f}%",
    help="Churned customers divided by all customers in the filtered data."
)

col3.metric(
    "Retention Rate",
    f"{retention_rate:.1f}%",
    help="Retained customers divided by all customers in the filtered data."
)

col4.metric(
    "Monthly Revenue Lost to Churn",
    f"{monthly_revenue_lost:,.2f}",
    help=(
        "Sum of monthly charges for customers marked as churned. "
        "This is a proxy, not a forecast of future revenue loss."
    )
)

st.metric(
    "Average Customer Tenure",
    f"{average_tenure:.1f} months"
)

st.divider()

# --------------------------------------------------
# 6. Churn by contract type
# --------------------------------------------------
st.subheader("Churn Analysis")

left, right = st.columns(2)

with left:
    contract_summary = (
        filtered_df.groupby(
            ["ContractType", "Churn"]
        )["CustomerID"]
        .nunique()
        .reset_index(name="Customers")
    )

    fig_contract = px.bar(
        contract_summary,
        x="ContractType",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Customers by Contract Type and Churn",
        labels={
            "ContractType": "Contract Type",
            "Customers": "Number of Customers",
            "Churn": "Churn Status"
        },
        color_discrete_map={
            "Yes": "#E45756",
            "No": "#4C78A8"
        }
    )

    st.plotly_chart(fig_contract, use_container_width=True)

with right:
    payment_summary = (
        filtered_df.groupby("PaymentMethod")
        .agg(
            Total_Customers=("CustomerID", "nunique"),
            Churned_Customers=(
                "Churn",
                lambda values: (values == "Yes").sum()
            )
        )
        .reset_index()
    )

    payment_summary["Churn_Rate"] = (
        payment_summary["Churned_Customers"]
        / payment_summary["Total_Customers"]
        * 100
    )

    fig_payment = px.bar(
        payment_summary,
        x="PaymentMethod",
        y="Churn_Rate",
        text=payment_summary["Churn_Rate"].map(
            lambda value: f"{value:.1f}%"
        ),
        title="Churn Rate by Payment Method",
        labels={
            "PaymentMethod": "Payment Method",
            "Churn_Rate": "Churn Rate (%)"
        }
    )

    fig_payment.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig_payment, use_container_width=True)

# --------------------------------------------------
# 7. Tenure and support-ticket analysis
# --------------------------------------------------
left, right = st.columns(2)

with left:
    fig_tenure = px.histogram(
        filtered_df,
        x="TenureMonths",
        color="Churn",
        nbins=10,
        barmode="overlay",
        title="Customer Tenure Distribution",
        labels={
            "TenureMonths": "Tenure (Months)",
            "count": "Customer Records",
            "Churn": "Churn Status"
        },
        color_discrete_map={
            "Yes": "#E45756",
            "No": "#4C78A8"
        }
    )

    st.plotly_chart(fig_tenure, use_container_width=True)

with right:
    fig_tickets = px.box(
        filtered_df,
        x="Churn",
        y="SupportTickets",
        color="Churn",
        title="Support Tickets by Churn Status",
        labels={
            "Churn": "Churn Status",
            "SupportTickets": "Support Tickets"
        },
        color_discrete_map={
            "Yes": "#E45756",
            "No": "#4C78A8"
        }
    )

    st.plotly_chart(fig_tickets, use_container_width=True)

# --------------------------------------------------
# 8. Business observations
# --------------------------------------------------
st.subheader("Business Insights")

contract_rates = (
    filtered_df.groupby("ContractType")
    .agg(
        Customers=("CustomerID", "nunique"),
        Churned=("Churn", lambda values: (values == "Yes").sum())
    )
)

contract_rates = contract_rates[
    contract_rates["Customers"] > 0
]

contract_rates["ChurnRate"] = (
    contract_rates["Churned"]
    / contract_rates["Customers"]
    * 100
)

if not contract_rates.empty:
    highest_contract = contract_rates["ChurnRate"].idxmax()
    highest_rate = contract_rates.loc[
        highest_contract, "ChurnRate"
    ]

    st.write(
        f"- **Highest observed contract churn:** "
        f"{highest_contract} ({highest_rate:.1f}%). "
        "Review the number of customers in this group before drawing conclusions."
    )

avg_tickets = filtered_df.groupby("Churn")["SupportTickets"].mean()

if "Yes" in avg_tickets.index and "No" in avg_tickets.index:
    st.write(
        f"- **Average support tickets:** "
        f"{avg_tickets['Yes']:.1f} for churned customers and "
        f"{avg_tickets['No']:.1f} for retained customers."
    )

st.write(
    "- **Recommended action:** Investigate customer feedback, "
    "support resolution times, and contract preferences before "
    "designing retention offers. These data show associations, "
    "not proof of why customers churn."
)

st.info(
    "This dataset contains a small sample. Treat group-level "
    "percentages as descriptive observations, not reliable "
    "forecasts of overall customer behavior."
)

# --------------------------------------------------
# 9. Customer details and download
# --------------------------------------------------
with st.expander("View filtered customer records"):
    st.dataframe(filtered_df, use_container_width=True)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Customer Data",
    data=csv_data,
    file_name="filtered_customers.csv",
    mime="text/csv"
)

st.caption(
    "Dashboard developed with Python, Pandas, Streamlit, "
    "and Plotly."
)
