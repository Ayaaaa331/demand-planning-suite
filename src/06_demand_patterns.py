
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

active = df[
    df["Use_For_Demand_Analysis"]
].copy()

active = active.sort_values(
    ["Series", "Week"]
).reset_index(drop=True)

print("=" * 70)
print("DEMAND PLANNING - TRENDS AND SEASONALITY")
print("=" * 70)

# ============================================================
# 2. CALCULATE MOVING AVERAGES
# ============================================================

# Trailing 13-week moving average.
# Uses only current and earlier observations.

active["MA_13"] = active.groupby(
    "Series"
)["Sales"].transform(
    lambda x: x.rolling(
        window=13,
        min_periods=13
    ).mean()
)

# ============================================================
# 3. ANALYZE DEMAND TRENDS
# ============================================================

trend_records = []

for series, group in active.groupby("Series"):

    group = group.sort_values("Week")

    first_13 = group.head(13)["Sales"].mean()
    last_13 = group.tail(13)["Sales"].mean()

    change_pct = (
        (last_13 / first_13 - 1) * 100
        if first_13 > 0 else np.nan
    )

    trend_records.append({
        "Series": series,
        "First_13_Weeks_Avg": first_13,
        "Last_13_Weeks_Avg": last_13,
        "Change_Pct": change_pct
    })

trend_summary = pd.DataFrame(trend_records)

print("\n1. LONGER-TERM DEMAND DEVELOPMENT")

print(
    trend_summary.round(2).to_string(index=False)
)

print(
    "\nNote: First-versus-last period comparisons "
    "may also reflect seasonal differences."
)

# ============================================================
# 4. CREATE MATCHED 52-WEEK COMPARISONS
# ============================================================

# Compare observations exactly 364 days apart.
# This provides a consistent weekly annual lag.

previous = active[
    ["Series", "Week", "Sales"]
].copy()

previous["Week"] = (
    previous["Week"] + pd.Timedelta(weeks=52)
)

previous = previous.rename(
    columns={"Sales": "Sales_Previous_Year"}
)

yoy = active.merge(
    previous,
    on=["Series", "Week"],
    how="inner",
    validate="one_to_one"
)

yoy["YoY_Change_Units"] = (
    yoy["Sales"] -
    yoy["Sales_Previous_Year"]
)

yoy_records = []

for series, group in yoy.groupby("Series"):

    previous_sales = group[
        "Sales_Previous_Year"
    ].sum()

    current_sales = group["Sales"].sum()

    growth_pct = (
        (current_sales / previous_sales - 1) * 100
        if previous_sales > 0 else np.nan
    )

    correlation = group["Sales"].corr(
        group["Sales_Previous_Year"]
    )

    yoy_records.append({
        "Series": series,
        "Matched_Weeks": len(group),
        "Previous_Period_Sales": previous_sales,
        "Current_Period_Sales": current_sales,
        "YoY_Growth_Pct": growth_pct,
        "Lag_52_Correlation": correlation
    })

yoy_summary = pd.DataFrame(yoy_records)

print("\n2. MATCHED YEAR-OVER-YEAR COMPARISON")

print(
    yoy_summary.round(3).to_string(index=False)
)

# ============================================================
# 5. MONTHLY DEMAND PROFILES
# ============================================================

active["Year"] = active["Week"].dt.year
active["Month"] = active["Week"].dt.month

monthly = active.groupby(
    ["Series", "Year", "Month"]
).agg(
    Weeks_Observed=("Week", "count"),
    Total_Sales=("Sales", "sum"),
    Average_Weekly_Sales=("Sales", "mean")
).reset_index()

monthly["Month_Date"] = pd.to_datetime(
    {
        "year": monthly["Year"],
        "month": monthly["Month"],
        "day": 1
    }
)

# Calculate a normalized demand index.
# 100 represents the series' historical active-period average.

series_average = active.groupby(
    "Series"
)["Sales"].mean().rename(
    "Historical_Weekly_Average"
)

monthly = monthly.merge(
    series_average,
    on="Series",
    how="left"
)

monthly["Demand_Index"] = (
    monthly["Average_Weekly_Sales"] /
    monthly["Historical_Weekly_Average"] * 100
)

# Flag months with incomplete weekly coverage.

monthly["Expected_Weeks"] = monthly[
    "Month_Date"
].apply(
    lambda date: len(
        pd.date_range(
            start=date,
            end=date + pd.offsets.MonthEnd(0),
            freq="W-MON"
        )
    )
)

monthly["Complete_Month"] = (
    monthly["Weeks_Observed"] ==
    monthly["Expected_Weeks"]
)

complete_months = monthly[
    monthly["Complete_Month"]
].copy()

seasonal_profile = complete_months.groupby(
    ["Series", "Month"]
).agg(
    Years_Observed=("Year", "nunique"),
    Average_Weekly_Sales=(
        "Average_Weekly_Sales", "mean"
    ),
    Average_Demand_Index=(
        "Demand_Index", "mean"
    )
).reset_index()

print("\n3. HIGHEST MONTHLY DEMAND INDICES")

peak_months = seasonal_profile.sort_values(
    ["Series", "Average_Demand_Index"],
    ascending=[True, False]
).groupby("Series").head(2)

print(
    peak_months.round(2).to_string(index=False)
)

print(
    "\nSeasonal indices are descriptive. "
    "Some months are observed in only one year."
)

# ============================================================
# 6. INVESTIGATE HIGHEST SALES WEEKS
# ============================================================

weekly_total = df.groupby(
    "Week"
).agg(
    Total_Sales=("Sales", "sum"),
    Observed_Revenue=("Revenue", "sum"),
    Flagged_Series=("Sales_Outlier", "sum")
).reset_index()

top_weeks = weekly_total.nlargest(
    10, "Total_Sales"
)

print("\n4. TEN HIGHEST-SELLING WEEKS")

print(
    top_weeks.round(2).to_string(index=False)
)

# ============================================================
# 7. REGIONAL DEMAND CORRELATION
# ============================================================

regional_records = []

for product, group in active.groupby(
    "Product_Group"
):

    pivot = group.pivot(
        index="Week",
        columns="Region",
        values="Sales"
    )

    if not {"North", "South"}.issubset(
        pivot.columns
    ):
        continue

    paired = pivot[
        ["North", "South"]
    ].dropna()

    correlation = paired["North"].corr(
        paired["South"]
    )

    regional_records.append({
        "Product_Group": product,
        "Matched_Weeks": len(paired),
        "North_South_Correlation": correlation
    })

regional_correlation = pd.DataFrame(
    regional_records
)

print("\n5. CROSS-REGIONAL DEMAND CORRELATION")

print(
    regional_correlation.round(3)
    .to_string(index=False)
)

# ============================================================
# 8. CHART 1 - WEEKLY SALES AND MOVING AVERAGES
# ============================================================

trend_chart_data = active.melt(
    id_vars=["Week", "Series"],
    value_vars=["Sales", "MA_13"],
    var_name="Measure",
    value_name="Units"
)

trend_chart_data["Measure"] = (
    trend_chart_data["Measure"].replace({
        "Sales": "Actual sales",
        "MA_13": "13-week moving average"
    })
)

trend_chart_data = trend_chart_data.sort_values(
    ["Series", "Measure", "Week"]
)

fig1 = px.line(
    trend_chart_data,
    x="Week",
    y="Units",
    color="Measure",
    facet_col="Series",
    facet_col_wrap=2,
    title="Weekly Demand and 13-Week Moving Average",
    color_discrete_map={
        "Actual sales": "#8FA9C2",
        "13-week moving average": "#C77C35"
    },
    template="plotly_white",
    height=1500
)

fig1.update_yaxes(matches=None)

fig1.for_each_annotation(
    lambda a: a.update(
        text=a.text.split("=")[-1]
    )
)

fig1.write_html(
    CHART_DIR / "05_demand_trends.html"
)

# ============================================================
# 9. CHART 2 - MONTHLY DEMAND INDEX
# ============================================================

heatmap_data = seasonal_profile.pivot(
    index="Series",
    columns="Month",
    values="Average_Demand_Index"
).reindex(
    columns=range(1, 13)
)

heatmap_data.columns = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]

fig2 = px.imshow(
    heatmap_data,
    text_auto=".0f",
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=100,
    aspect="auto",
    title="Monthly Demand Index by Series",
    labels={
        "x": "Month",
        "y": "Demand Series",
        "color": "Demand Index"
    },
    template="plotly_white",
    height=550
)

fig2.write_html(
    CHART_DIR / "06_seasonality_heatmap.html"
)

# ============================================================
# 10. CHART 3 - MATCHED YEAR-OVER-YEAR GROWTH
# ============================================================

fig3 = px.bar(
    yoy_summary.sort_values("YoY_Growth_Pct"),
    x="YoY_Growth_Pct",
    y="Series",
    orientation="h",
    text="YoY_Growth_Pct",
    title="Sales Growth Across Matched 52-Week Periods",
    labels={
        "YoY_Growth_Pct": "Sales Growth (%)"
    },
    template="plotly_white"
)

fig3.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
    marker_color="#365C8D",
    cliponaxis=False
)

fig3.write_html(
    CHART_DIR / "07_year_over_year_growth.html"
)

# ============================================================
# 11. CHART 4 - TOTAL WEEKLY SALES
# ============================================================

fig4 = px.line(
    weekly_total.sort_values("Week"),
    x="Week",
    y="Total_Sales",
    title="Total Weekly Sales Across All Products and Regions",
    labels={
        "Total_Sales": "Units Sold"
    },
    template="plotly_white"
)

fig4.add_scatter(
    x=top_weeks["Week"],
    y=top_weeks["Total_Sales"],
    mode="markers",
    name="Top 10 sales weeks",
    marker={
        "color": "#C77C35",
        "size": 9
    }
)

fig4.write_html(
    CHART_DIR / "08_peak_sales_weeks.html"
)

# ============================================================
# 12. VALIDATION
# ============================================================

assert not active.duplicated(
    ["Series", "Week"]
).any()

assert len(trend_summary) == df[
    "Series"
].nunique()

assert len(yoy_summary) == df[
    "Series"
].nunique()

assert yoy_summary[
    "Lag_52_Correlation"
].dropna().between(-1, 1).all()

assert np.isclose(
    weekly_total["Total_Sales"].sum(),
    df["Sales"].sum()
)

print("\n6. VALIDATION")

print("PASS - No duplicate observations.")
print("PASS - All demand series included.")
print("PASS - Annual lag correlations valid.")
print("PASS - Weekly sales totals reconciled.")

# ============================================================
# 13. EXPORT RESULTS
# ============================================================

active.to_csv(
    OUTPUT_DIR / "demand_with_moving_averages.csv",
    index=False
)

trend_summary.to_csv(
    OUTPUT_DIR / "demand_trend_summary.csv",
    index=False
)

yoy.to_csv(
    OUTPUT_DIR / "year_over_year_details.csv",
    index=False
)

yoy_summary.to_csv(
    OUTPUT_DIR / "year_over_year_summary.csv",
    index=False
)

monthly.to_csv(
    OUTPUT_DIR / "monthly_demand_summary.csv",
    index=False
)

seasonal_profile.to_csv(
    OUTPUT_DIR / "seasonality_profile.csv",
    index=False
)

top_weeks.to_csv(
    OUTPUT_DIR / "peak_sales_weeks.csv",
    index=False
)

regional_correlation.to_csv(
    OUTPUT_DIR / "regional_demand_correlation.csv",
    index=False
)

print("\n" + "=" * 70)
print("Demand pattern analysis completed successfully!")
print("Analytical tables saved to:", OUTPUT_DIR)
print("Interactive charts saved to:", CHART_DIR)
print("=" * 70)
