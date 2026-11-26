
from pathlib import Path
import pandas as pd

# Define the path to the Excel dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = PROJECT_ROOT / "data" / "weekly_sales_price_data_shared.xls"

# Check whether the file exists
if not FILE_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {FILE_PATH}")

# Open the Excel workbook
workbook = pd.ExcelFile(FILE_PATH, engine="xlrd")

print("=" * 70)
print("DEMAND PLANNING PROJECT - INITIAL DATA INSPECTION")
print("=" * 70)

# Display available sheets
print("\nAvailable Excel sheets:")
print(workbook.sheet_names)

# Inspect each sheet
for sheet in workbook.sheet_names:

    df = pd.read_excel(workbook, sheet_name=sheet, header=None)

    print("\n" + "=" * 70)
    print(f"SHEET: {sheet}")
    print("=" * 70)

    print(f"\nNumber of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False, header=False))

    print("\nLast 15 rows:")
    print(df.tail(15).to_string(index=False, header=False))

print("\nInspection completed successfully!")
