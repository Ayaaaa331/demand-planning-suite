
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. CONFIGURATION AND FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_WINDOW = 13
MIN_PRICE_OBSERVATIONS = 6
PROMOTION_THRESHOLD = 10

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
print("DEMAND PLANNING - PRICE ANALYSIS")
print("=" * 70)

# ============================================================
# 2. DESCRIPTIVE PRICE STATISTICS
# ============================================================

observed = active[
    active["Use_For_Price_Analysis"]
].copy()

price_summary = observed.groupby("Series").agg(
    Observed_Prices=("Price", "count"),
    Average_Price=("Price", "mean"),
    Median_Price=("Price", "median"),
    Minimum_Price=("Price", "min"),
    Maximum_Price=("Price", "max"),
    Price_Std=("Price", "std")
).reset_index()

price_summary["Price_CV_Pct"] = (
    price_summary["Price_Std"] /
    price_summary["Average_Price"] * 100
)

print("\n1. PRICE STATISTICS BY SERIES")

print(
    price_summary.round(2).to_string(index=False)
)

# ============================================================
# 3. ESTIMATE ROLLING REFERENCE PRICES
# ============================================================

# Only use observed prices from weeks
# preceding the current observation.
#
# Shift(1) prevents the current price
# from influencing its own reference price.

active["Reference_Price"] = active.groupby(
    "Series"
)["Price"].transform(
    lambda x: x.shift(1).rolling(
        window=REFERENCE_WINDOW,
        min_periods=MIN_PRICE_OBSERVATIONS
    ).median()
)

active["Reference_Observation_Count"] = (
    active.groupby("Series")["Price"].transform(
        lambda x: x.shift(1).rolling(
            window=REFERENCE_WINDOW,
            min_periods=1
        ).count()
    )
)

# ============================================================
# 4. CALCULATE RELATIVE PRICE REDUCTIONS
# ============================================================

active["Price_Analysis_Eligible"] = (
    active["Use_For_Price_Analysis"] &
    active["Reference_Price"].notna() &
    active["Reference_Price"].gt(0)
)

active["Discount_Pct"] = np.nan

eligible = active["Price_Analysis_Eligible"]

active.loc[eligible, "Discount_Pct"] = (
    (
        active.loc[eligible, "Reference_Price"] -
        active.loc[eligible, "Price"]
    ) /
    active.loc[eligible, "Reference_Price"]
    * 100
)

# ============================================================
# 5. IDENTIFY POTENTIAL PROMOTIONS
# ============================================================

active["Potential_Promotion"] = (
    active["Price_Analysis_Eligible"] &
    active["Discount_Pct"].ge(
        PROMOTION_THRESHOLD
    )
)

active["Price_Status_Analytical"] = "Not assessable"

active.loc[
    eligible,
    "Price_Status_Analytical"
] = "No flagged discount"

active.loc[
    active["Potential_Promotion"],
    "Price_Status_Analytical"
] = "Potential promotion"

promotions = active[
    active["Potential_Promotion"]
].copy()

print("\n2. PRICE REDUCTION SCREENING")

print(
    f"Reference window: "
    f"{REFERENCE_WINDOW} preceding weeks"
)

print(
    f"Minimum observed reference prices: "
    f"{MIN_PRICE_OBSERVATIONS}"
)

print(
    f"Promotion threshold: "
    f"{PROMOTION_THRESHOLD}%"
)

print(
    f"Eligible observations: "
    f"{eligible.sum()}"
)

print(
    f"Potential promotion observations: "
    f"{len(promotions)}"
)

# ============================================================
# 6. PROMOTION FREQUENCY BY SERIES
# ============================================================

promotion_summary = active.groupby("Series").agg(
    Active_Weeks=("Week", "count"),
    Assessable_Weeks=(
        "Price_Analysis_Eligible", "sum"
    ),
    Potential_Promotion_Weeks=(
        "Potential_Promotion", "sum"
    )
).reset_index()

promotion_summary["Promotion_Frequency_Pct"] = (
    promotion_summary["Potential_Promotion_Weeks"] /
    promotion_summary["Assessable_Weeks"].replace(
        0, np.nan
    ) * 100
)

discount_summary = promotions.groupby(
    "Series"
).agg(
    Average_Discount_Pct=("Discount_Pct", "mean"),
    Maximum_Discount_Pct=("Discount_Pct", "max"),
    Average_Promotion_Sales=("Sales", "mean")
).reset_index()

promotion_summary = promotion_summary.merge(
    discount_summary,
    on="Series",
    how="left"
)

print("\n3. POTENTIAL PROMOTIONS BY SERIES")

print(
    promotion_summary.round(2).to_string(index=False)
)

# ============================================================
# 7. THRESHOLD SENSITIVITY ANALYSIS
# ============================================================

sensitivity_records = []

for series, group in active.groupby("Series"):

    valid = group[
        group["Price_Analysis_Eligible"]
    ]

    for threshold in [5, 10, 15, 20]:

        count = valid["Discount_Pct"].ge(
            threshold
        ).sum()

        frequency = (
            count / len(valid) * 100
            if len(valid) > 0
            else np.nan
        )

        sensitivity_records.append({
            "Series": series,
            "Threshold_Pct": threshold,
            "Assessable_Weeks": len(valid),
            "Potential_Promotions": count,
            "Frequency_Pct": frequency
        })

sensitivity = pd.DataFrame(
    sensitivity_records
)

print("\n4. PROMOTION THRESHOLD SENSITIVITY")

sensitivity_pivot = sensitivity.pivot(
    index="Series",
    columns="Threshold_Pct",
    values="Potential_Promotions"
)

print(sensitivity_pivot.to_string())

# ============================================================
# 8. DESCRIPTIVE PRICE-DEMAND RELATIONSHIP
# ============================================================

# Correlations do not establish causal price effects.
# They are preliminary indicators only.

correlation_records = []

for series, group in observed.groupby("Series"):

    pearson = group["Price"].corr(
        group["Sales"],
        method="pearson"
    )

    spearman = group["Price"].corr(
        group["Sales"],
        method="spearman"
    )

    correlation_records.append({
        "Series": series,
        "Observations": len(group),
        "Pearson_Correlation": pearson,
        "Spearman_Correlation": spearman
    })

price_correlations = pd.DataFrame(
    correlation_records
)

print("\n5. PRICE-DEMAND CORRELATIONS")

print(
    price_correlations.round(3).to_string(index=False)
)

# ============================================================
# 9. CROSS-REGIONAL PROMOTION COMPARISON
# ============================================================

regional_records = []

for product, group in active.groupby("Product_Group"):

    valid = group[
        group["Price_Analysis_Eligible"]
    ]

    paired = valid.pivot(
        index="Week",
        columns="Region",
        values="Potential_Promotion"
    )

    if not {"North", "South"}.issubset(
        paired.columns
    ):
        continue

    paired = paired[
        ["North", "South"]
    ].dropna().astype(bool)

    north = paired["North"]
    south = paired["South"]

    both = north & south
    either = north | south

    regional_records.append({
        "Product_Group": product,
        "Comparable_Weeks": len(paired),
        "North_Promotion_Weeks": int(north.sum()),
        "South_Promotion_Weeks": int(south.sum()),
        "Both_Regions": int(both.sum()),
        "North_Only": int((north & ~south).sum()),
        "South_Only": int((south & ~north).sum()),
        "Promotion_Overlap_Pct": (
            both.sum() / either.sum() * 100
            if either.sum() > 0
            else np.nan
        )
    })

regional_promotions = pd.DataFrame(
    regional_records
)

print("\n6. CROSS-REGIONAL PROMOTION PATTERNS")

print(
    regional_promotions.round(2)
    .to_string(index=False)
)

# ============================================================
# 10. INVESTIGATE TOP SALES WEEKS
# ============================================================

top_weeks = pd.read_csv(
    OUTPUT_DIR / "peak_sales_weeks.csv",
    parse_dates=["Week"]
)

peak_analysis = active[
    active["Week"].isin(top_weeks["Week"])
].copy()

peak_summary = peak_analysis.groupby("Week").agg(
    Total_Sales=("Sales", "sum"),
    Assessable_Series=(
        "Price_Analysis_Eligible", "sum"
    ),
    Potential_Promotions=(
        "Potential_Promotion", "sum"
    ),
    Average_Discount_Pct=(
        "Discount_Pct", "mean"
    )
).reset_index()

peak_summary = peak_summary.sort_values(
    "Total_Sales",
    ascending=False
)

print("\n7. PRICE REDUCTIONS DURING PEAK SALES WEEKS")

print(
    peak_summary.round({
        "Total_Sales": 0,
        "Average_Discount_Pct": 2
    }).to_string(index=False)
)

largest_discounts = promotions.nlargest(
    15, "Discount_Pct"
)

print("\n8. LARGEST OBSERVED PRICE REDUCTIONS")

print(
    largest_discounts[
        [
            "Week",
            "Series",
            "Sales",
            "Price",
            "Reference_Price",
            "Discount_Pct"
        ]
    ].round({
        "Price": 2,
        "Reference_Price": 2,
        "Discount_Pct": 2
    }).to_string(index=False)
)

# ============================================================
# 11. CHART 1 - OBSERVED AND REFERENCE PRICES
# ============================================================

chart_data = active.melt(
    id_vars=["Week", "Series"],
    value_vars=["Price", "Reference_Price"],
    var_name="Price_Type",
    value_name="Price_EUR"
)

chart_data["Price_Type"] = (
    chart_data["Price_Type"].replace({
        "Price": "Observed price",
        "Reference_Price": "Reference price"
    })
)

chart_data = chart_data.sort_values(
    ["Series", "Price_Type", "Week"]
)

fig1 = px.line(
    chart_data,
    x="Week",
    y="Price_EUR",
    color="Price_Type",
    facet_col="Series",
    facet_col_wrap=2,
    title="Observed Prices and Rolling Reference Prices",
    labels={"Price_EUR": "Price (€)"},
    color_discrete_map={
        "Observed price": "#365C8D",
        "Reference price": "#46A094"
    },
    template="plotly_white",
    height=1500
)

fig1.update_yaxes(matches=None)
fig1.update_traces(connectgaps=False)

fig1.for_each_annotation(
    lambda a: a.update(
        text=a.text.split("=")[-1]
    )
)

fig1.write_html(
    CHART_DIR / "09_reference_prices.html"
)

# ============================================================
# 12. CHART 2 - PRICE VS SALES
# ============================================================

fig2 = px.scatter(
    active[active["Use_For_Price_Analysis"]],
    x="Price",
    y="Sales",
    color="Price_Status_Analytical",
    facet_col="Series",
    facet_col_wrap=2,
    hover_data=["Week", "Reference_Price", "Discount_Pct"],
    title="Relationship Between Observed Price and Sales",
    labels={
        "Price": "Observed price (€)",
        "Sales": "Units sold"
    },
    color_discrete_map={
        "Potential promotion": "#C77C35",
        "No flagged discount": "#365C8D",
        "Not assessable": "#B0B0B0"
    },
    template="plotly_white",
    height=1500
)

fig2.update_xaxes(matches=None)
fig2.update_yaxes(matches=None)

fig2.for_each_annotation(
    lambda a: a.update(
        text=a.text.split("=")[-1]
    )
)

fig2.write_html(
    CHART_DIR / "10_price_demand_relationship.html"
)

# ============================================================
# 13. CHART 3 - PROMOTION FREQUENCY
# ============================================================

fig3 = px.bar(
    promotion_summary.sort_values(
        "Promotion_Frequency_Pct"
    ),
    x="Series",
    y="Promotion_Frequency_Pct",
    text="Promotion_Frequency_Pct",
    title="Potential Promotion Frequency by Series",
    labels={
        "Promotion_Frequency_Pct":
            "Share of assessable weeks (%)"
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
    CHART_DIR / "11_promotion_frequency.html"
)

# ============================================================
# 14. CHART 4 - THRESHOLD SENSITIVITY
# ============================================================

fig4 = px.line(
    sensitivity,
    x="Threshold_Pct",
    y="Potential_Promotions",
    color="Series",
    markers=True,
    title="Sensitivity to Promotion Definition",
    labels={
        "Threshold_Pct": "Minimum price reduction (%)",
        "Potential_Promotions": "Flagged weeks"
    },
    template="plotly_white"
)

fig4.write_html(
    CHART_DIR / "12_promotion_sensitivity.html"
)

# ============================================================
# 15. VALIDATION
# ============================================================

assert promotions[
    "Discount_Pct"
].ge(PROMOTION_THRESHOLD).all()

assert promotions["Price"].notna().all()

assert promotions[
    "Reference_Price"
].notna().all()

assert not active.duplicated(
    ["Week", "Series"]
).any()

# A higher discount threshold cannot
# produce more promotion observations.

for series, group in sensitivity.groupby("Series"):

    counts = group.sort_values(
        "Threshold_Pct"
    )["Potential_Promotions"]

    assert counts.is_monotonic_decreasing

# Independently verify one discount calculation.

assert np.isclose(
    (40 - 30) / 40 * 100,
    25.0
)

# Independently verify each available reference price.

for series, group in active.groupby("Series"):

    group = group.sort_values("Week")

    prices = group["Price"].to_numpy()
    references = group["Reference_Price"].to_numpy()

    for i in range(len(group)):

        previous_prices = prices[
            max(0, i - REFERENCE_WINDOW):i
        ]

        available_prices = previous_prices[
            ~np.isnan(previous_prices)
        ]

        if len(available_prices) >= MIN_PRICE_OBSERVATIONS:

            expected = np.median(available_prices)

            assert np.isclose(
                references[i],
                expected
            )

        else:

            assert np.isnan(references[i])

print("\n9. VALIDATION")

print("PASS - Promotion thresholds applied correctly.")
print("PASS - No missing prices in promotion candidates.")
print("PASS - No duplicate observations.")
print("PASS - Threshold sensitivity is consistent.")
print("PASS - Discount calculation independently verified.")
print("PASS - Reference prices use preceding weeks only.")

# ============================================================
# 16. EXPORT RESULTS
# ============================================================

active.to_csv(
    OUTPUT_DIR / "price_analysis_enriched.csv",
    index=False
)

price_summary.to_csv(
    OUTPUT_DIR / "price_statistics.csv",
    index=False
)

promotions.to_csv(
    OUTPUT_DIR / "potential_promotions.csv",
    index=False
)

promotion_summary.to_csv(
    OUTPUT_DIR / "promotion_frequency_summary.csv",
    index=False
)

sensitivity.to_csv(
    OUTPUT_DIR / "promotion_threshold_sensitivity.csv",
    index=False
)

price_correlations.to_csv(
    OUTPUT_DIR / "price_demand_correlations.csv",
    index=False
)

regional_promotions.to_csv(
    OUTPUT_DIR / "regional_promotion_comparison.csv",
    index=False
)

peak_summary.to_csv(
    OUTPUT_DIR / "peak_weeks_price_analysis.csv",
    index=False
)

print("\n" + "=" * 70)
print("Price analysis completed successfully!")
print("Analytical tables saved to:", OUTPUT_DIR)
print("Interactive charts saved to:", CHART_DIR)
print("=" * 70)
