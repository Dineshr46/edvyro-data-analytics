import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR/"data"/"customer_churn_sample.csv"
OUTPUT_DIR = BASE_DIR/"outputs"

OUTPUT_DIR.mkdir(exist_ok = True)

# Load the dataset
df = pd.read_csv(DATA_FILE)

print("Dataset Loaded Successfully")
print("Number of Rows and Columns", df.shape)

print("\nFirst five rows: ")
print(df.head())

print("\nColumn Names: ")
print(df.columns.tolist())

print("\nDataset Information: ")
print(df.info())

print("\nStatistical Summary: ")
print(df.describe(include="all").T)


# Missing-value Analysis
missing_report = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (
        df.isnull().mean() * 100
    ).round(2)
})

missing_report = missing_report.sort_values(
    "Missing_Percentage",
    ascending=False
)

print("\nMissing-value report: ")
print(missing_report)

missing_report.to_csv(
    OUTPUT_DIR/"missing_values_report.csv"
)


# Duplicate Analysis
duplicat_count = df.duplicated().sum()
print("\nDuplicate rows found: ", duplicat_count)

# Remove confirmed duplicate rows
df = df.drop_duplicates().copy()

print("\nShape after duplicate removed: ", df.shape)


# Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", "_", regex=True)
)

# Remove unnecessary spaces from text values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].map(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )

print("\nColumn names after cleaning:")
print(df.columns.tolist())


# Outlier analysis
numeric_columns = df.select_dtypes(include="number").columns

outlier_summary = []

for col in numeric_columns:
    values = df[col].dropna()

    if values.empty:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    count = (
        (values < lower_bound) |
        (values > upper_bound)
    ).sum()

    outlier_summary.append({
        "Column": col,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Potential_Outliers": int(count)
    })

outlier_report = pd.DataFrame(outlier_summary)

print("\nPotential outliers:")
print(outlier_report)

outlier_report.to_csv(
    OUTPUT_DIR / "outlier_report.csv",
    index=False
)



# Chart 1: Missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(
    ascending=False
)

if not missing.empty:
    missing.plot(kind="bar")
    plt.title("Missing Values by Column")
    plt.xlabel("Column")
    plt.ylabel("Missing Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_values.png")
    plt.close()

# Chart 2: Numeric distributions
for col in numeric_columns:
    if df[col].notna().any():
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col].dropna(), bins=20)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()

        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in col
        )

        plt.savefig(
            OUTPUT_DIR / f"{safe_name}_distribution.png"
        )
        plt.close()

# Chart 3: Correlation heatmap
numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] >= 2:
    plt.figure(figsize=(9, 6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Between Numeric Variables")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()

print("\nCharts saved in the outputs folder.")


# Export cleaned dataset
df.to_csv(
    OUTPUT_DIR / "cleaned_dataset.csv",
    index=False
)

# Save final dataset summary
summary = pd.DataFrame({
    "Metric": [
        "Rows after cleaning",
        "Columns after cleaning",
        "Duplicate rows remaining",
        "Missing cells remaining"
    ],
    "Value": [
        len(df),
        len(df.columns),
        int(df.duplicated().sum()),
        int(df.isnull().sum().sum())
    ]
})

summary.to_csv(
    OUTPUT_DIR / "final_data_quality_summary.csv",
    index=False
)

print("\nTask 2 analysis completed.")
print("Check the outputs folder for your results.")