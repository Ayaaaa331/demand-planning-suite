
from pathlib import Path
import pandas as pd

# ============================================================
# 1. DEFINE FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = PROJECT_ROOT / "data" / "weekly_sales_price_data_shared.xls"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD THE ORIGINAL DATA
# ============================================================

sales = pd.read_excel(
    FILE_PATH,
    sheet_name="Sales in unit",
    engine="xlrd"
)

prices = pd.read_excel(
    FILE_PATH,
    sheet_name="Prices in Euro",
    engine="xlrd"
)

# Clean column names
sales.columns = sales.columns.str.strip()
prices.columns = prices.columns.str.strip()

# Convert dates
sales["Week"] = pd.to_datetime(sales["Week"])
prices["Week"] = pd.to_datetime(prices["Week"])

# ============================================================
# 3. CONVERT WIDE FORMAT TO LONG FORMAT
# ============================================================

sales_long = sales.melt(
    id_vars=["Week"],
    var_name="Series",
    value_name="Sales"
)

prices_long = prices.melt(
    id_vars=["Week"],
    var_name="Series",
    value_name="Price"
)

# ============================================================
# 4. MERGE SALES AND PRICES
# ============================================================

df = pd.merge(
    sales_long,
    prices_long,
    on=["Week", "Series"],
    how="outer",
    validate="one_to_one",
    indicator=True
)

# Verify that all sales and price records match
if not df["_merge"].eq("both").all():
    raise ValueError("Some sales and price records do not match.")

df = df.drop(columns="_merge")

# ============================================================
# 5. EXTRACT REGION AND PRODUCT GROUP
# ============================================================

df[["Region", "Product_Group"]] = df["Series"].str.rsplit(
    " ", n=1, expand=True
)

df["Product_Group"] = df["Product_Group"].astype(int)

df["Sales"] = pd.to_numeric(df["Sales"], errors="raise")
df["Price"] = pd.to_numeric(df["Price"], errors="raise")

# ============================================================
# 6. IDENTIFY HISTORICAL AND FORECAST PERIODS
# ============================================================

last_historical_week = df.loc[df["Sales"].notna(), "Week"].max()

df["Period"] = "Historical"

df.loc[
    df["Week"] > last_historical_week,
    "Period"
] = "Forecast"

# Reorder columns
df = df[
    [
        "Week",
        "Region",
        "Product_Group",
        "Series",
        "Sales",
        "Price",
        "Period"
    ]
]

# Sort observations
df = df.sort_values(
    ["Region", "Product_Group", "Week"]
).reset_index(drop=True)

# ============================================================
# 7. SEPARATE HISTORICAL AND FUTURE DATA
# ============================================================

historical = df[df["Period"] == "Historical"].copy()
future = df[df["Period"] == "Forecast"].copy()

# ============================================================
# 8. INITIAL DATA QUALITY CHECKS
# ============================================================

print("=" * 70)
print("DEMAND PLANNING - DATA PREPARATION")
print("=" * 70)

print(f"\nTotal observations: {len(df)}")
print(f"Historical observations: {len(historical)}")
print(f"Future observations: {len(future)}")
print(f"Number of series: {df['Series'].nunique()}")
print(f"Historical weeks: {historical['Week'].nunique()}")
print(f"Forecast weeks: {future['Week'].nunique()}")

print("\nHistorical period:")
print(f"Start: {historical['Week'].min().date()}")
print(f"End: {historical['Week'].max().date()}")

print("\nForecast period:")
print(f"Start: {future['Week'].min().date()}")
print(f"End: {future['Week'].max().date()}")

print("\nDATA QUALITY CHECKS")

print(f"Missing historical sales: {historical['Sales'].isna().sum()}")
print(f"Missing historical prices: {historical['Price'].isna().sum()}")
print(f"Zero historical sales: {historical['Sales'].eq(0).sum()}")
print(f"Negative sales: {historical['Sales'].lt(0).sum()}")
print(f"Non-positive prices: {historical['Price'].le(0).sum()}")

print(
    f"Duplicate observations: "
    f"{df.duplicated(['Week', 'Series']).sum()}"
)

# ============================================================
# 9. SUMMARY BY SERIES
# ============================================================

summary = historical.groupby("Series").agg(
    Weeks=("Week", "nunique"),
    Total_Sales=("Sales", "sum"),
    Average_Sales=("Sales", "mean"),
    Missing_Sales=("Sales", lambda x: x.isna().sum()),
    Zero_Sales=("Sales", lambda x: x.eq(0).sum()),
    Missing_Prices=("Price", lambda x: x.isna().sum()),
    Average_Price=("Price", "mean")
).round(2)

print("\nSUMMARY BY SERIES")
print(summary.to_string())

print("\nFIRST 10 PREPARED OBSERVATIONS")
print(df.head(10).to_string(index=False))

# ============================================================
# 10. EXPORT PREPARED DATASETS
# ============================================================

df.to_csv(
    OUTPUT_DIR / "prepared_data.csv",
    index=False
)

historical.to_csv(
    OUTPUT_DIR / "historical_data.csv",
    index=False
)

future.to_csv(
    OUTPUT_DIR / "future_data.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "initial_data_quality_summary.csv"
)

print("\nFiles successfully exported to:")
print(OUTPUT_DIR)

print("\nData preparation completed successfully!")
