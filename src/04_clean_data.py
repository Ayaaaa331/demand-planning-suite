
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD FLAGGED HISTORICAL DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

df = pd.read_csv(
    OUTPUT_DIR / "historical_data_flagged.csv",
    parse_dates=["Week"]
)

df = df.sort_values(
    ["Series", "Week"]
).reset_index(drop=True)

print("=" * 70)
print("DEMAND PLANNING - DATA TREATMENT")
print("=" * 70)

# ============================================================
# 2. PRESERVE ORIGINAL OBSERVATIONS
# ============================================================

df["Sales_Original"] = df["Sales"]
df["Price_Original"] = df["Price"]

# ============================================================
# 3. IDENTIFY ACTIVE DEMAND PERIODS
# ============================================================

df["Pre_Launch"] = (
    df["Lifecycle_Status"] == "Before_First_Sale"
)

df["Active_Zero_Sales"] = (
    (df["Lifecycle_Status"] == "Active_Window") &
    (df["Sales"] == 0)
)

df["Potential_Availability_Issue"] = (
    df["Active_Zero_Sales"] &
    df["Price"].isna()
)

# ============================================================
# 4. DEFINE ANALYTICAL ELIGIBILITY
# ============================================================

# Exclude pre-launch observations from active-demand analysis

df["Use_For_Demand_Analysis"] = (
    (df["Lifecycle_Status"] == "Active_Window") &
    df["Sales"].notna()
)

# Price analysis requires an observed positive price
# and positive recorded sales

df["Use_For_Price_Analysis"] = (
    df["Use_For_Demand_Analysis"] &
    df["Price"].notna() &
    (df["Price"] > 0) &
    (df["Sales"] > 0)
)

# Potential outliers remain eligible for analysis

# ============================================================
# 5. CALCULATE OBSERVED REVENUE
# ============================================================

df["Revenue"] = df["Sales"] * df["Price"]

# Zero units sold means zero sales revenue,
# even when the price is missing

df.loc[df["Sales"] == 0, "Revenue"] = 0

# Revenue remains unknown when sales are positive
# but the corresponding price is missing

df["Revenue_Known"] = df["Revenue"].notna()

# ============================================================
# 6. ADD CALENDAR VARIABLES
# ============================================================

df["Year"] = df["Week"].dt.year
df["Month"] = df["Week"].dt.month
df["Quarter"] = df["Week"].dt.quarter

df["Week_of_Year"] = (
    df["Week"].dt.isocalendar().week.astype(int)
)

df["Time_Index"] = (
    (df["Week"] - df["Week"].min()).dt.days // 7
)

df["Weeks_Since_First_Sale"] = np.nan

for series, group in df.groupby("Series"):

    positive_sales = group[group["Sales"] > 0]

    if positive_sales.empty:
        continue

    first_sale = positive_sales["Week"].min()

    active = group[group["Week"] >= first_sale]

    df.loc[
        active.index,
        "Weeks_Since_First_Sale"
    ] = (
        (active["Week"] - first_sale).dt.days // 7
    )

# ============================================================
# 7. CREATE ANALYTICAL SUBSETS
# ============================================================

demand_data = df[
    df["Use_For_Demand_Analysis"]
].copy()

price_data = df[
    df["Use_For_Price_Analysis"]
].copy()

# ============================================================
# 8. DOCUMENT DATA TREATMENT DECISIONS
# ============================================================

treatment_summary = pd.DataFrame([
    {
        "Issue": "Pre-launch observations",
        "Count": int(df["Pre_Launch"].sum()),
        "Treatment": (
            "Retained in master dataset; excluded "
            "from active-demand analysis"
        )
    },
    {
        "Issue": "Active zero sales",
        "Count": int(df["Active_Zero_Sales"].sum()),
        "Treatment": (
            "Retained and flagged as potential "
            "availability issues"
        )
    },
    {
        "Issue": "Missing prices with positive sales",
        "Count": int(
            (df["Price"].isna() & (df["Sales"] > 0)).sum()
        ),
        "Treatment": (
            "Sales retained; excluded from "
            "observed-price analysis"
        )
    },
    {
        "Issue": "Potential sales outliers",
        "Count": int(df["Sales_Outlier"].sum()),
        "Treatment": (
            "Retained for investigation of "
            "seasonality and promotions"
        )
    },
    {
        "Issue": "Missing revenue",
        "Count": int(df["Revenue"].isna().sum()),
        "Treatment": (
            "Left unknown because observed "
            "prices are unavailable"
        )
    }
])

# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\nDATASET SUMMARY")

print(f"Original historical observations: {len(df)}")
print(f"Pre-launch observations: {df['Pre_Launch'].sum()}")

print(
    f"Active demand observations: {len(demand_data)}"
)

print(
    f"Observed-price analysis observations: "
    f"{len(price_data)}"
)

print(
    f"Potential availability issues: "
    f"{df['Potential_Availability_Issue'].sum()}"
)

print(
    f"Observations with unknown revenue: "
    f"{df['Revenue'].isna().sum()}"
)

print("\nANALYTICAL OBSERVATIONS BY SERIES")

summary = df.groupby("Series").agg(
    Total_Weeks=("Week", "count"),
    Demand_Analysis_Weeks=("Use_For_Demand_Analysis", "sum"),
    Price_Analysis_Weeks=("Use_For_Price_Analysis", "sum"),
    Pre_Launch_Weeks=("Pre_Launch", "sum"),
    Active_Zero_Sales=("Active_Zero_Sales", "sum"),
    Potential_Outliers=("Sales_Outlier", "sum"),
    Known_Revenue_Weeks=("Revenue_Known", "sum")
)

print(summary.to_string())

print("\nDATA TREATMENT DECISIONS")
print(treatment_summary.to_string(index=False))

# ============================================================
# 10. VALIDATE DATA PRESERVATION
# ============================================================

assert df["Sales"].equals(df["Sales_Original"])
assert df["Price"].equals(df["Price_Original"])

assert not df.duplicated(["Week", "Series"]).any()

assert not price_data["Price"].isna().any()

print("\nValidation passed: original observations preserved.")

# ============================================================
# 11. EXPORT ANALYTICAL DATASETS
# ============================================================

df.to_csv(
    OUTPUT_DIR / "historical_data_clean.csv",
    index=False
)

demand_data.to_csv(
    OUTPUT_DIR / "demand_analysis_data.csv",
    index=False
)

price_data.to_csv(
    OUTPUT_DIR / "price_analysis_data.csv",
    index=False
)

treatment_summary.to_csv(
    OUTPUT_DIR / "data_treatment_summary.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "analytical_dataset_summary.csv"
)

print("\n" + "=" * 70)
print("Data treatment completed successfully!")
print("All results exported to:", OUTPUT_DIR)
print("=" * 70)
