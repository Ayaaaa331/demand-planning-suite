
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px

# ============================================================
# 1. DEFINE PATHS AND LOAD DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    OUTPUT_DIR / "historical_data_clean.csv",
    parse_dates=["Week"]
)

active = df[df["Use_For_Demand_Analysis"]].copy()

print("=" * 70)
print("DEMAND PLANNING - HISTORICAL SALES ANALYSIS")
print("=" * 70)

# ============================================================
# 2. OVERALL BUSINESS PERFORMANCE
# ============================================================

total_sales = df["Sales"].sum()
observed_revenue = df["Revenue"].sum()

unknown_revenue_units = df.loc[
    df["Revenue"].isna(), "Sales"
].sum()

revenue_coverage = (
    1 - unknown_revenue_units / total_sales
) * 100

print("\n1. OVERALL BUSINESS PERFORMANCE")

print(f"Total units sold: {total_sales:,.0f}")

print(
    f"Total observed revenue: "
    f"EUR {observed_revenue:,.2f}"
)

print(
    f"Units sold without recorded prices: "
    f"{unknown_revenue_units:,.0f}"
)

print(
    f"Revenue coverage by sales volume: "
    f"{revenue_coverage:.2f}%"
)

# ============================================================
# 3. PRODUCT GROUP PERFORMANCE
# ============================================================

product_summary = df.groupby(
    "Product_Group"
).agg(
    Total_Sales=("Sales", "sum"),
    Observed_Revenue=("Revenue", "sum"),
    Missing_Revenue_Observations=(
        "Revenue", lambda x: x.isna().sum()
    )
).reset_index()

product_summary["Sales_Share_Pct"] = (
    product_summary["Total_Sales"] /
    total_sales * 100
)

product_summary["Observed_Revenue_Share_Pct"] = (
    product_summary["Observed_Revenue"] /
    observed_revenue * 100
)

product_summary = product_summary.sort_values(
    "Total_Sales", ascending=False
)

print("\n2. PRODUCT GROUP PERFORMANCE")

print(
    product_summary.round(2).to_string(index=False)
)

# ============================================================
# 4. REGIONAL PERFORMANCE
# ============================================================

region_summary = df.groupby("Region").agg(
    Total_Sales=("Sales", "sum"),
    Observed_Revenue=("Revenue", "sum"),
    Missing_Revenue_Observations=(
        "Revenue", lambda x: x.isna().sum()
    )
).reset_index()

region_summary["Sales_Share_Pct"] = (
    region_summary["Total_Sales"] /
    total_sales * 100
)

region_summary["Observed_Revenue_Share_Pct"] = (
    region_summary["Observed_Revenue"] /
    observed_revenue * 100
)

print("\n3. REGIONAL PERFORMANCE")

print(
    region_summary.round(2).to_string(index=False)
)

# ============================================================
# 5. REGIONAL COMPARISON BY PRODUCT GROUP
# ============================================================

regional_comparison = df.pivot_table(
    index="Product_Group",
    columns="Region",
    values="Sales",
    aggfunc="sum"
).reset_index()

regional_comparison["South_vs_North_Pct"] = (
    (
        regional_comparison["South"] -
        regional_comparison["North"]
    ) /
    regional_comparison["North"] * 100
)

print("\n4. REGIONAL COMPARISON BY PRODUCT")

print(
    regional_comparison.round(2).to_string(index=False)
)

# ============================================================
# 6. DEMAND VARIABILITY BY SERIES
# ============================================================

# Active-period observations are used so that
# probable pre-launch weeks do not distort
# average demand and demand variability.

series_summary = active.groupby("Series").agg(
    Active_Weeks=("Week", "count"),
    Total_Sales=("Sales", "sum"),
    Average_Weekly_Sales=("Sales", "mean"),
    Median_Weekly_Sales=("Sales", "median"),
    Standard_Deviation=("Sales", "std"),
    Minimum_Sales=("Sales", "min"),
    Maximum_Sales=("Sales", "max"),
    Zero_Sales_Weeks=("Sales", lambda x: x.eq(0).sum())
).reset_index()

# Coefficient of variation:
# Standard deviation divided by mean demand.

series_summary["CV_Pct"] = (
    series_summary["Standard_Deviation"] /
    series_summary["Average_Weekly_Sales"] * 100
)

series_summary = series_summary.sort_values("CV_Pct")

print("\n5. DEMAND VARIABILITY BY SERIES")

print(
    series_summary.round(2).to_string(index=False)
)

most_stable = series_summary.iloc[0]
most_variable = series_summary.iloc[-1]

print(
    f"\nLowest demand variability: "
    f"{most_stable['Series']} "
    f"(CV: {most_stable['CV_Pct']:.2f}%)"
)

print(
    f"Highest demand variability: "
    f"{most_variable['Series']} "
    f"(CV: {most_variable['CV_Pct']:.2f}%)"
)

# ============================================================
# 7. CHART SETTINGS
# ============================================================

REGION_COLORS = {
    "North": "#365C8D",
    "South": "#46A094"
}

# ============================================================
# 8. CHART 1 - SALES BY PRODUCT AND REGION
# ============================================================

sales_by_product_region = df.groupby(
    ["Product_Group", "Region"]
)["Sales"].sum().reset_index()

sales_by_product_region["Product"] = (
    "Group " +
    sales_by_product_region["Product_Group"].astype(str)
)

fig1 = px.bar(
    sales_by_product_region,
    x="Product",
    y="Sales",
    color="Region",
    barmode="group",
    title="Historical Sales Volume by Product and Region",
    labels={
        "Sales": "Units Sold",
        "Product": "Product Group"
    },
    color_discrete_map=REGION_COLORS,
    template="plotly_white"
)

fig1.write_html(
    CHART_DIR / "01_sales_by_product_region.html"
)

# ============================================================
# 9. CHART 2 - PRODUCT SHARE OF TOTAL SALES
# ============================================================

product_summary["Product"] = (
    "Group " +
    product_summary["Product_Group"].astype(str)
)

fig2 = px.bar(
    product_summary.sort_values("Sales_Share_Pct"),
    x="Sales_Share_Pct",
    y="Product",
    orientation="h",
    text="Sales_Share_Pct",
    title="Product Contribution to Total Sales",
    labels={
        "Sales_Share_Pct": "Share of Sales (%)",
        "Product": "Product Group"
    },
    template="plotly_white"
)

fig2.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
    marker_color="#365C8D"
)

fig2.update_xaxes(
    range=[
        0,
        product_summary["Sales_Share_Pct"].max() * 1.2
    ]
)

fig2.write_html(
    CHART_DIR / "02_product_sales_share.html"
)

# ============================================================
# 10. CHART 3 - HISTORICAL WEEKLY SALES
# ============================================================

history = df.copy()

history["Product"] = (
    "Group " + history["Product_Group"].astype(str)
)

history = history.sort_values(
    ["Product_Group", "Region", "Week"]
)

product_order = [
    f"Group {product}"
    for product in sorted(df["Product_Group"].unique())
]

fig3 = px.line(
    history,
    x="Week",
    y="Sales",
    color="Region",
    facet_row="Product",
    title="Weekly Sales History by Product Group",
    color_discrete_map=REGION_COLORS,
    category_orders={
        "Product": product_order
    },
    labels={
        "Sales": "Units Sold",
        "Week": "Week"
    },
    template="plotly_white",
    height=1100
)

# Use independent vertical scales so that
# lower-volume products remain readable.

fig3.update_yaxes(matches=None)

fig3.for_each_annotation(
    lambda a: a.update(
        text=a.text.split("=")[-1]
    )
)

fig3.write_html(
    CHART_DIR / "03_weekly_sales_history.html"
)

# ============================================================
# 11. CHART 4 - DEMAND VARIABILITY
# ============================================================

fig4 = px.bar(
    series_summary,
    x="Series",
    y="CV_Pct",
    text="CV_Pct",
    title="Demand Variability by Series",
    labels={
        "CV_Pct": "Coefficient of Variation (%)",
        "Series": "Demand Series"
    },
    template="plotly_white",
    color="CV_Pct",
    color_continuous_scale="Blues"
)

fig4.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

fig4.update_layout(
    coloraxis_showscale=False
)

fig4.update_yaxes(
    range=[
        0,
        series_summary["CV_Pct"].max() * 1.15
    ]
)

fig4.write_html(
    CHART_DIR / "04_demand_variability.html"
)

# ============================================================
# 12. VALIDATE ANALYTICAL RESULTS
# ============================================================

assert np.isclose(
    product_summary["Total_Sales"].sum(),
    total_sales
), "Product sales totals do not match."

assert np.isclose(
    region_summary["Total_Sales"].sum(),
    total_sales
), "Regional sales totals do not match."

assert np.isclose(
    product_summary["Sales_Share_Pct"].sum(),
    100
), "Product sales shares do not sum to 100%."

assert len(series_summary) == df["Series"].nunique(), (
    "Some demand series are missing."
)

print("\n6. VALIDATION")

print("PASS - Product sales totals match.")
print("PASS - Regional sales totals match.")
print("PASS - Sales shares sum to 100%.")
print("PASS - All demand series included.")

# ============================================================
# 13. EXPORT ANALYTICAL RESULTS
# ============================================================

product_summary.to_csv(
    OUTPUT_DIR / "product_sales_summary.csv",
    index=False
)

region_summary.to_csv(
    OUTPUT_DIR / "regional_sales_summary.csv",
    index=False
)

regional_comparison.to_csv(
    OUTPUT_DIR / "regional_product_comparison.csv",
    index=False
)

series_summary.to_csv(
    OUTPUT_DIR / "series_demand_statistics.csv",
    index=False
)

print("\n" + "=" * 70)
print("Historical sales analysis completed successfully!")
print("Analytical tables saved to:", OUTPUT_DIR)
print("Interactive charts saved to:", CHART_DIR)
print("=" * 70)
