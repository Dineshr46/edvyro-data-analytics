import pandas as pd
from pathlib import Path

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


