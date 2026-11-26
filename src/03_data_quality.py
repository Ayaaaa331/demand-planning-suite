
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 1. DEFINE FILE PATHS AND LOAD DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

df = pd.read_csv(
    OUTPUT_DIR / "historical_data.csv",
    parse_dates=["Week"]
)

df = df.sort_values(
    ["Series", "Week"]
).reset_index(drop=True)

print("=" * 70)
print("DEMAND PLANNING - DETAILED DATA QUALITY ASSESSMENT")
print("=" * 70)

# ============================================================
# 2. CHECK WEEKLY DATA COMPLETENESS
# ============================================================

print("\n1. WEEKLY DATA COMPLETENESS")

expected_weeks = pd.date_range(
    start=df["Week"].min(),
    end=df["Week"].max(),
    freq="W-MON"
)

for series, group in df.groupby("Series"):

    actual_weeks = pd.DatetimeIndex(group["Week"])
    missing_weeks = expected_weeks.difference(actual_weeks)

    print(
        f"{series}: "
        f"{len(actual_weeks)} observations, "
        f"{len(missing_weeks)} missing weeks"
    )

# ============================================================
# 3. INVESTIGATE PRODUCT LIFECYCLE
# ============================================================

print("\n2. PRODUCT LIFECYCLE ANALYSIS")

lifecycle_records = []

df["Lifecycle_Status"] = "Active_Window"

for series, group in df.groupby("Series"):

    positive_sales = group[group["Sales"] > 0]

    if positive_sales.empty:
        df.loc[group.index, "Lifecycle_Status"] = "Never_Observed_Selling"
        continue

    first_sale = positive_sales["Week"].min()
    last_sale = positive_sales["Week"].max()

    pre_launch = (
        (df["Series"] == series) &
        (df["Week"] < first_sale)
    )

    post_last_sale = (
        (df["Series"] == series) &
        (df["Week"] > last_sale)
    )

    df.loc[pre_launch, "Lifecycle_Status"] = "Before_First_Sale"

    df.loc[
        post_last_sale,
        "Lifecycle_Status"
    ] = "After_Last_Sale"

    active_zero_sales = group[
        (group["Week"] >= first_sale) &
        (group["Week"] <= last_sale) &
        (group["Sales"] == 0)
    ]

    lifecycle_records.append({
        "Series": series,
        "First_Positive_Sale": first_sale,
        "Last_Positive_Sale": last_sale,
        "Weeks_Before_First_Sale": int(pre_launch.sum()),
        "Weeks_After_Last_Sale": int(post_last_sale.sum()),
        "Zero_Sales_Within_Active_Window": len(active_zero_sales),
        "Positive_Sales_Weeks": len(positive_sales)
    })

lifecycle = pd.DataFrame(lifecycle_records)

print(lifecycle.to_string(index=False))

# ============================================================
# 4. CLASSIFY MISSING PRICES
# ============================================================

print("\n3. MISSING PRICE ANALYSIS")

df["Price_Status"] = "Available"

df.loc[
    df["Price"].isna(),
    "Price_Status"
] = "Missing_Unclassified"

df.loc[
    df["Price"].isna() &
    (df["Lifecycle_Status"] == "Before_First_Sale") &
    (df["Sales"] == 0),
    "Price_Status"
] = "Missing_Before_First_Sale"

df.loc[
    df["Price"].isna() &
    (df["Sales"] > 0),
    "Price_Status"
] = "Missing_With_Positive_Sales"

df.loc[
    df["Price"].isna() &
    (df["Sales"] == 0) &
    (df["Lifecycle_Status"] == "Active_Window"),
    "Price_Status"
] = "Missing_With_Active_Zero_Sales"

df.loc[
    df["Price"].isna() &
    (df["Sales"] == 0) &
    (df["Lifecycle_Status"] == "After_Last_Sale"),
    "Price_Status"
] = "Missing_After_Last_Sale"

missing_prices = df[df["Price"].isna()].copy()

print("\nMissing price categories:")
print(
    missing_prices["Price_Status"]
    .value_counts()
    .to_string()
)

print("\nAll missing price observations:")

print(
    missing_prices[
        ["Week", "Series", "Sales", "Price_Status"]
    ].to_string(index=False)
)

# ============================================================
# 5. INVESTIGATE ZERO SALES
# ============================================================

print("\n4. ZERO SALES ANALYSIS")

zero_sales = df[df["Sales"] == 0].copy()

print("\nAll zero-sales observations:")

print(
    zero_sales[
        ["Week", "Series", "Price", "Lifecycle_Status"]
    ].to_string(index=False)
)

# ============================================================
# 6. DETECT POTENTIAL SALES OUTLIERS
# ============================================================

print("\n5. POTENTIAL SALES OUTLIERS")

df["Sales_Outlier"] = False
df["Sales_Outlier_Type"] = "Not_Flagged"

outlier_records = []

for series, group in df.groupby("Series"):

    active = group[
        group["Lifecycle_Status"] == "Active_Window"
    ]

    if active.empty:
        continue

    q1 = active["Sales"].quantile(0.25)
    q3 = active["Sales"].quantile(0.75)

    iqr = q3 - q1

    lower_bound = max(0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr

    high_outliers = active[
        active["Sales"] > upper_bound
    ]

    low_outliers = active[
        active["Sales"] < lower_bound
    ]

    df.loc[
        high_outliers.index,
        "Sales_Outlier"
    ] = True

    df.loc[
        high_outliers.index,
        "Sales_Outlier_Type"
    ] = "Unusually_High"

    df.loc[
        low_outliers.index,
        "Sales_Outlier"
    ] = True

    df.loc[
        low_outliers.index,
        "Sales_Outlier_Type"
    ] = "Unusually_Low"

    outlier_records.append({
        "Series": series,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "High_Outliers": len(high_outliers),
        "Low_Outliers": len(low_outliers)
    })

outlier_summary = pd.DataFrame(outlier_records)

print(outlier_summary.round(2).to_string(index=False))

outliers = df[df["Sales_Outlier"]].copy()

print("\nTotal potential sales outliers:", len(outliers))

print("\n10 highest flagged sales observations:")

print(
    outliers.nlargest(10, "Sales")[
        [
            "Week",
            "Series",
            "Sales",
            "Price",
            "Sales_Outlier_Type"
        ]
    ].to_string(index=False)
)

# ============================================================
# 7. CREATE DATA QUALITY SUMMARY
# ============================================================

quality_summary = df.groupby("Series").agg(
    Observations=("Week", "count"),
    Missing_Sales=("Sales", lambda x: x.isna().sum()),
    Missing_Prices=("Price", lambda x: x.isna().sum()),
    Zero_Sales=("Sales", lambda x: x.eq(0).sum()),
    Potential_Outliers=("Sales_Outlier", "sum")
).reset_index()

print("\n6. FINAL DATA QUALITY SUMMARY")
print(quality_summary.to_string(index=False))

# ============================================================
# 8. EXPORT RESULTS
# ============================================================

df.to_csv(
    OUTPUT_DIR / "historical_data_flagged.csv",
    index=False
)

lifecycle.to_csv(
    OUTPUT_DIR / "product_lifecycle.csv",
    index=False
)

missing_prices.to_csv(
    OUTPUT_DIR / "missing_price_details.csv",
    index=False
)

zero_sales.to_csv(
    OUTPUT_DIR / "zero_sales_details.csv",
    index=False
)

outliers.to_csv(
    OUTPUT_DIR / "sales_outliers.csv",
    index=False
)

outlier_summary.to_csv(
    OUTPUT_DIR / "outlier_summary.csv",
    index=False
)

quality_summary.to_csv(
    OUTPUT_DIR / "detailed_data_quality_summary.csv",
    index=False
)

print("\n" + "=" * 70)
print("Data quality assessment completed successfully!")
print("All results exported to:", OUTPUT_DIR)
print("=" * 70)
