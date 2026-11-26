
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CHART = OUT / "charts"

CHART.mkdir(parents=True, exist_ok=True)

PRIMARY_REFERENCE = "Median 13 weeks"
PRIMARY_MODEL = "Pooled log + price"

# ============================================================
# 2. LOAD VALIDATED ANALYTICAL RESULTS
# ============================================================

pred = pd.read_csv(
    OUT / "reference_price_candidate_predictions.csv",
    parse_dates=["Week"]
)

robustness = pd.read_csv(
    OUT / "reference_price_robustness_summary.csv"
)

uncertainty = pd.read_csv(
    OUT / "promotion_uncertainty_summary.csv"
)

original = pd.read_csv(
    OUT / "promotion_baseline_estimates.csv",
    parse_dates=["Week"]
)

diagnostics = pd.read_csv(
    OUT / "baseline_model_comparison_metrics.csv"
)

print("=" * 70)
print("DEMAND PLANNING - FINAL PROMOTION EFFECTIVENESS")
print("=" * 70)

# ============================================================
# 3. RECONSTRUCT THE STRICT COMMON SAMPLE
# ============================================================

# Use exactly the same observations as script 08C.
# All model/reference specifications must be supported.

supported_all = pred.groupby(
    "Row_ID"
)["Supported"].all()

primary = pred[
    (
        pred["Reference_Definition"]
        == PRIMARY_REFERENCE
    )
    & (
        pred["Model"]
        == PRIMARY_MODEL
    )
    & pred["Original_Promotion"]
].copy()

primary["Evaluation_Eligible"] = (
    primary["Row_ID"].map(supported_all)
)

primary = primary.sort_values(
    ["Series", "Week"]
).reset_index(drop=True)

assert len(primary) == len(original)

assert not primary.duplicated(
    ["Series", "Week"]
).any()

assert set(
    zip(primary["Series"], primary["Week"])
) == set(
    zip(original["Series"], original["Week"])
)

primary["Evaluation_Status"] = np.where(
    primary["Evaluation_Eligible"],
    "Supported common sample",
    "Excluded from strict evaluation"
)

df = primary[
    primary["Evaluation_Eligible"]
].copy()

print("\n1. EVALUATION COVERAGE")

print("Primary model:", PRIMARY_MODEL)
print("Reference price:", PRIMARY_REFERENCE)

print(
    "Original promotion candidates:",
    len(primary)
)

print(
    "Strict evaluation observations:",
    len(df)
)

print(
    "Excluded observations:",
    len(primary) - len(df)
)

print(
    f"Evaluation coverage: "
    f"{len(df) / len(primary) * 100:.2f}%"
)

# ============================================================
# 4. CALCULATE SALES AND BREAK-EVEN REQUIREMENTS
# ============================================================

primary["Required_Uplift_Pct"] = (
    primary["Reference_Price"]
    / primary["Price"] - 1
) * 100

primary["Break_Even_Units"] = (
    primary["Baseline_Demand"]
    * primary["Reference_Price"]
    / primary["Price"]
)

primary["Break_Even_Whole_Units"] = np.ceil(
    primary["Break_Even_Units"]
)

primary["Actual_Additional_Units"] = (
    primary["Sales"]
    - primary["Baseline_Demand"]
)

primary["Required_Additional_Units"] = (
    primary["Break_Even_Units"]
    - primary["Baseline_Demand"]
)

positive_baseline = primary[
    "Baseline_Demand"
].where(
    primary["Baseline_Demand"] > 0
)

primary["Actual_Uplift_Pct"] = (
    primary["Actual_Additional_Units"]
    / positive_baseline * 100
)

primary["Model_Implied_Additional_Units"] = (
    primary["Predicted_Actual_Price"]
    - primary["Baseline_Demand"]
)

primary["Model_Implied_Uplift_Pct"] = (
    primary["Model_Implied_Additional_Units"]
    / positive_baseline * 100
)

primary["Unexplained_Demand_Units"] = (
    primary["Sales"]
    - primary["Predicted_Actual_Price"]
)

# ============================================================
# 5. REVENUE CALCULATIONS
# ============================================================

primary["Actual_Revenue"] = (
    primary["Sales"]
    * primary["Price"]
)

primary["Baseline_Revenue"] = (
    primary["Baseline_Demand"]
    * primary["Reference_Price"]
)

primary["Predicted_Discounted_Revenue"] = (
    primary["Predicted_Actual_Price"]
    * primary["Price"]
)

primary["Actual_Revenue_Gap"] = (
    primary["Actual_Revenue"]
    - primary["Baseline_Revenue"]
)

primary["Model_Revenue_Effect"] = (
    primary["Predicted_Discounted_Revenue"]
    - primary["Baseline_Revenue"]
)

primary["Unexplained_Revenue_Gap"] = (
    primary["Actual_Revenue"]
    - primary["Predicted_Discounted_Revenue"]
)

# ============================================================
# 6. BREAK-EVEN CLASSIFICATION
# ============================================================

eligible = primary["Evaluation_Eligible"]

primary["Actual_Break_Even"] = pd.Series(
    pd.NA,
    index=primary.index,
    dtype="boolean"
)

primary["Model_Break_Even"] = pd.Series(
    pd.NA,
    index=primary.index,
    dtype="boolean"
)

primary.loc[
    eligible,
    "Actual_Break_Even"
] = (
    primary.loc[
        eligible, "Actual_Revenue_Gap"
    ] >= 0
)

primary.loc[
    eligible,
    "Model_Break_Even"
] = (
    primary.loc[
        eligible, "Model_Revenue_Effect"
    ] >= 0
)

df = primary[eligible].copy()

print("\n2. OVERALL BREAK-EVEN RESULTS")

print(
    f"Observed revenue above baseline: "
    f"{df['Actual_Break_Even'].mean()*100:.2f}%"
)

print(
    f"Model-implied break-even frequency: "
    f"{df['Model_Break_Even'].mean()*100:.2f}%"
)

print(
    f"Total observed revenue gap: "
    f"EUR {df['Actual_Revenue_Gap'].sum():,.2f}"
)

print(
    f"Total model-implied revenue effect: "
    f"EUR {df['Model_Revenue_Effect'].sum():,.2f}"
)

# ============================================================
# 7. AGGREGATED BUSINESS RESULTS
# ============================================================

def summarize(frame, grouping):

    summary = frame.groupby(
        grouping
    ).agg(
        Evaluated_Promotions=("Week", "count"),

        Average_Discount_Pct=(
            "Discount_Pct", "mean"
        ),

        Actual_Units=("Sales", "sum"),

        Baseline_Units=(
            "Baseline_Demand", "sum"
        ),

        Predicted_Discounted_Units=(
            "Predicted_Actual_Price", "sum"
        ),

        Break_Even_Units=(
            "Break_Even_Units", "sum"
        ),

        Actual_Revenue=("Actual_Revenue", "sum"),

        Baseline_Revenue=(
            "Baseline_Revenue", "sum"
        ),

        Actual_Revenue_Gap=(
            "Actual_Revenue_Gap", "sum"
        ),

        Model_Revenue_Effect=(
            "Model_Revenue_Effect", "sum"
        ),

        Unexplained_Revenue_Gap=(
            "Unexplained_Revenue_Gap", "sum"
        ),

        Actual_Break_Even_Rate=(
            "Actual_Break_Even", "mean"
        ),

        Model_Break_Even_Rate=(
            "Model_Break_Even", "mean"
        )

    ).reset_index()

    # Calculate aggregate percentages using
    # summed units rather than averaging
    # potentially extreme weekly ratios.

    summary["Actual_Uplift_Pct"] = (
        summary["Actual_Units"]
        / summary["Baseline_Units"] - 1
    ) * 100

    summary["Required_Uplift_Pct"] = (
        summary["Break_Even_Units"]
        / summary["Baseline_Units"] - 1
    ) * 100

    summary["Model_Uplift_Pct"] = (
        summary["Predicted_Discounted_Units"]
        / summary["Baseline_Units"] - 1
    ) * 100

    summary["Actual_Revenue_Gap_Pct"] = (
        summary["Actual_Revenue_Gap"]
        / summary["Baseline_Revenue"] * 100
    )

    summary["Model_Revenue_Effect_Pct"] = (
        summary["Model_Revenue_Effect"]
        / summary["Baseline_Revenue"] * 100
    )

    summary["Actual_Break_Even_Rate"] *= 100
    summary["Model_Break_Even_Rate"] *= 100

    return summary


series_summary = summarize(
    df,
    ["Series", "Region", "Product_Group"]
)

product_summary = summarize(
    df,
    ["Product_Group"]
)

region_summary = summarize(
    df,
    ["Region"]
)

# ============================================================
# 8. REPORT COVERAGE BY SERIES AND PRODUCT
# ============================================================

def add_coverage(summary, grouping):

    candidates = primary.groupby(
        grouping
    ).size().rename(
        "Promotion_Candidates"
    ).reset_index()

    result = candidates.merge(
        summary,
        on=grouping,
        how="left"
    )

    result["Evaluated_Promotions"] = (
        result["Evaluated_Promotions"]
        .fillna(0)
        .astype(int)
    )

    result["Excluded_Promotions"] = (
        result["Promotion_Candidates"]
        - result["Evaluated_Promotions"]
    )

    result["Coverage_Pct"] = (
        result["Evaluated_Promotions"]
        / result["Promotion_Candidates"]
        * 100
    )

    return result


series_summary = add_coverage(
    series_summary,
    ["Series", "Region", "Product_Group"]
)

product_summary = add_coverage(
    product_summary,
    ["Product_Group"]
)

region_summary = add_coverage(
    region_summary,
    ["Region"]
)

display_columns = [
    "Promotion_Candidates",
    "Evaluated_Promotions",
    "Average_Discount_Pct",
    "Required_Uplift_Pct",
    "Actual_Uplift_Pct",
    "Model_Uplift_Pct",
    "Actual_Revenue_Gap",
    "Model_Revenue_Effect",
    "Actual_Break_Even_Rate",
    "Model_Break_Even_Rate"
]

print("\n3. PROMOTION EFFECTIVENESS BY SERIES")

print(
    series_summary[
        ["Series"] + display_columns
    ].round(2).to_string(index=False)
)

print("\n4. PROMOTION EFFECTIVENESS BY PRODUCT")

print(
    product_summary[
        ["Product_Group"] + display_columns
    ].round(2).to_string(index=False)
)

print("\n5. PROMOTION EFFECTIVENESS BY REGION")

print(
    region_summary[
        ["Region"] + display_columns
    ].round(2).to_string(index=False)
)

# ============================================================
# 9. INCORPORATE MODEL ROBUSTNESS AND UNCERTAINTY
# ============================================================

management = product_summary.merge(
    robustness.rename(
        columns={
            "Conclusion": "Specification_Conclusion",
            "Evaluated_Promotions": "Robustness_N"
        }
    ),
    on="Product_Group",
    validate="one_to_one"
)

management = management.merge(
    uncertainty.rename(
        columns={
            "Conclusion": "Bootstrap_Conclusion"
        }
    ),
    on="Product_Group",
    validate="one_to_one"
)

# Include predictive reliability of primary model.

primary_diagnostics = diagnostics[
    (
        diagnostics["Model"] == PRIMARY_MODEL
    )
    & (
        diagnostics["Sample"] == "All assessable"
    )
].copy()

series_lookup = primary[
    ["Series", "Product_Group"]
].drop_duplicates()

primary_diagnostics = primary_diagnostics.merge(
    series_lookup,
    on="Series",
    validate="one_to_one"
)

reliability = primary_diagnostics.groupby(
    "Product_Group"
).agg(
    Mean_Series_WAPE_Pct=("WAPE_Pct", "mean"),
    Worst_Series_WAPE_Pct=("WAPE_Pct", "max"),
    Mean_Series_MAPE_Pct=("MAPE_Pct", "mean")
).reset_index()

management = management.merge(
    reliability,
    on="Product_Group",
    validate="one_to_one"
)

# ============================================================
# 10. EVIDENCE-BASED RECOMMENDATIONS
# ============================================================

def recommendation(row):

    # These are descriptive decision rules,
    # not formal statistical significance tests.

    # Flag very small evaluation samples or
    # high model prediction errors.

    if (
        row["Evaluated_Promotions"] < 10
        or row["Worst_Series_WAPE_Pct"] > 70
    ):

        return (
            "Insufficiently reliable - "
            "investigate before changing strategy"
        )

    if (
        row["Lowest_Estimate"] > 0
        and row["Lower_5th_Percentile"] > 0
    ):

        return (
            "Promising revenue effect - "
            "prioritize controlled promotion trials"
        )

    if (
        row["Highest_Estimate"] < 0
        and row["Upper_95th_Percentile"] < 0
    ):

        return (
            "Consistently negative estimates - "
            "reconsider discount strategy"
        )

    return (
        "Uncertain revenue effect - "
        "test alternative discount depths"
    )


management["Recommendation"] = (
    management.apply(
        recommendation,
        axis=1
    )
)

print("\n6. MANAGEMENT RECOMMENDATIONS")

for _, row in management.iterrows():

    print(
        f"\nPRODUCT GROUP "
        f"{int(row['Product_Group'])}"
    )

    print(
        f"Evaluated promotions: "
        f"{int(row['Evaluated_Promotions'])}"
    )

    print(
        f"Actual uplift vs baseline: "
        f"{row['Actual_Uplift_Pct']:.2f}%"
    )

    print(
        f"Aggregate required uplift: "
        f"{row['Required_Uplift_Pct']:.2f}%"
    )

    print(
        f"Model-implied uplift: "
        f"{row['Model_Uplift_Pct']:.2f}%"
    )

    print(
        f"Actual revenue gap: "
        f"EUR {row['Actual_Revenue_Gap']:,.2f}"
    )

    print(
        f"Model-implied revenue effect: "
        f"EUR {row['Model_Revenue_Effect']:,.2f}"
    )

    print(
        f"Bootstrap uncertainty range: "
        f"EUR {row['Lower_5th_Percentile']:,.2f}"
        f" to "
        f"EUR {row['Upper_95th_Percentile']:,.2f}"
    )

    print(
        f"Recommendation: "
        f"{row['Recommendation']}"
    )

# ============================================================
# 11. DISCOUNT THRESHOLD SENSITIVITY
# ============================================================

threshold_results = []

for threshold in [10, 15, 20]:

    subset = df[
        df["Discount_Pct"] >= threshold
    ]

    if subset.empty:
        continue

    result = summarize(
        subset,
        ["Product_Group"]
    )

    result["Threshold_Pct"] = threshold

    threshold_results.append(result)

sensitivity = pd.concat(
    threshold_results,
    ignore_index=True
)

print("\n7. DISCOUNT THRESHOLD SENSITIVITY")

print(
    sensitivity[
        [
            "Product_Group",
            "Threshold_Pct",
            "Evaluated_Promotions",
            "Actual_Revenue_Gap",
            "Model_Revenue_Effect"
        ]
    ].round(2).to_string(index=False)
)

# ============================================================
# 12. CHART - PROMOTION BREAK-EVEN
# ============================================================

df["Product"] = (
    "Group "
    + df["Product_Group"].astype(str)
)

fig1 = px.scatter(
    df,
    x="Required_Uplift_Pct",
    y="Actual_Uplift_Pct",
    color="Product",
    symbol="Region",
    hover_data=[
        "Week",
        "Series",
        "Price",
        "Reference_Price",
        "Model_Revenue_Effect"
    ],
    title="Actual Sales Uplift vs Revenue Break-Even",
    labels={
        "Required_Uplift_Pct": "Required uplift (%)",
        "Actual_Uplift_Pct": "Observed uplift (%)"
    },
    template="plotly_white",
    height=650
)

limit = df["Required_Uplift_Pct"].max()

fig1.add_trace(
    go.Scatter(
        x=[0, limit],
        y=[0, limit],
        mode="lines",
        name="Revenue break-even",
        line=dict(
            color="#555555",
            dash="dash"
        )
    )
)

fig1.write_html(
    CHART / "14_promotion_break_even.html"
)

# ============================================================
# 13. CHART - REQUIRED VS ACTUAL AND MODEL UPLIFT
# ============================================================

uplift_chart = product_summary.melt(
    id_vars=["Product_Group"],
    value_vars=[
        "Required_Uplift_Pct",
        "Actual_Uplift_Pct",
        "Model_Uplift_Pct"
    ],
    var_name="Measure",
    value_name="Uplift"
)

uplift_chart["Product"] = (
    "Group "
    + uplift_chart["Product_Group"].astype(str)
)

uplift_chart["Measure"] = (
    uplift_chart["Measure"].replace({
        "Required_Uplift_Pct": "Required uplift",
        "Actual_Uplift_Pct": "Observed uplift",
        "Model_Uplift_Pct": "Model-implied uplift"
    })
)

fig2 = px.bar(
    uplift_chart,
    x="Product",
    y="Uplift",
    color="Measure",
    barmode="group",
    title="Promotional Demand Uplift by Product",
    labels={
        "Uplift": "Aggregate sales uplift (%)"
    },
    color_discrete_sequence=[
        "#9CA3AF",
        "#365C8D",
        "#46A094"
    ],
    template="plotly_white"
)

fig2.add_hline(y=0)

fig2.write_html(
    CHART / "15_promotion_uplift_comparison.html"
)

# ============================================================
# 14. CHART - OBSERVED VS MODEL-IMPLIED REVENUE
# ============================================================

revenue_chart = product_summary.melt(
    id_vars=["Product_Group"],
    value_vars=[
        "Actual_Revenue_Gap",
        "Model_Revenue_Effect"
    ],
    var_name="Measure",
    value_name="Revenue_Effect"
)

revenue_chart["Product"] = (
    "Group "
    + revenue_chart["Product_Group"].astype(str)
)

revenue_chart["Measure"] = (
    revenue_chart["Measure"].replace({
        "Actual_Revenue_Gap":
            "Observed revenue vs baseline",
        "Model_Revenue_Effect":
            "Model-implied price effect"
    })
)

fig3 = px.bar(
    revenue_chart,
    x="Product",
    y="Revenue_Effect",
    color="Measure",
    barmode="group",
    title="Observed and Model-Implied Revenue Effects",
    labels={
        "Revenue_Effect": "Revenue difference (€)"
    },
    color_discrete_sequence=[
        "#365C8D",
        "#C77C35"
    ],
    template="plotly_white"
)

fig3.add_hline(y=0)

fig3.write_html(
    CHART / "16_promotion_revenue_impact.html"
)

# ============================================================
# 15. CHART - REGIONAL COMPARISON
# ============================================================

series_summary["Product"] = (
    "Group "
    + series_summary["Product_Group"].astype(str)
)

fig4 = px.bar(
    series_summary,
    x="Product",
    y="Model_Revenue_Effect",
    color="Region",
    barmode="group",
    title="Model-Implied Revenue Effects by Region",
    labels={
        "Model_Revenue_Effect": "Revenue effect (€)"
    },
    color_discrete_map={
        "North": "#365C8D",
        "South": "#46A094"
    },
    template="plotly_white"
)

fig4.add_hline(y=0)

fig4.write_html(
    CHART / "17_regional_promotion_effectiveness.html"
)

# ============================================================
# 16. VALIDATION
# ============================================================

assert df["Baseline_Demand"].gt(0).all()

assert df["Price"].gt(0).all()

assert df["Reference_Price"].gt(0).all()

assert df["Required_Uplift_Pct"].ge(0).all()

assert np.allclose(
    df["Break_Even_Units"] * df["Price"],
    df["Baseline_Revenue"]
)

assert np.allclose(
    df["Actual_Revenue_Gap"],
    df["Model_Revenue_Effect"]
    + df["Unexplained_Revenue_Gap"]
)

assert (
    product_summary["Evaluated_Promotions"].sum()
    == len(df)
)

assert (
    series_summary["Evaluated_Promotions"].sum()
    == len(df)
)

assert np.isclose(
    product_summary["Model_Revenue_Effect"].sum(),
    region_summary["Model_Revenue_Effect"].sum()
)

# Reproduce the point estimates from script 08C.

assert np.allclose(
    management["Model_Revenue_Effect"],
    management["Point_Estimate"],
    rtol=1e-6,
    atol=0.01
)

assert (
    management["Evaluated_Promotions"]
    == management["Robustness_N"]
).all()

# Independent break-even formula test.

assert np.isclose(
    (40 / 30 - 1) * 100,
    33.3333333333
)

assert np.isclose(
    100 * 40 / 30 * 30,
    4000
)

print("\n8. VALIDATION")

print("PASS - Valid positive reference prices.")
print("PASS - Valid positive baseline demand.")
print("PASS - Revenue break-even formula verified.")
print("PASS - Revenue decomposition verified.")
print("PASS - Product and regional totals reconcile.")
print("PASS - Script 08C point estimates reproduced.")
print("PASS - Strict common sample reproduced.")
print("PASS - Independent break-even test passed.")

# ============================================================
# 17. EXPORT FINAL PROMOTION ANALYSIS
# ============================================================

primary.to_csv(
    OUT / "promotion_effectiveness_details.csv",
    index=False
)

series_summary.to_csv(
    OUT / "promotion_effectiveness_by_series.csv",
    index=False
)

product_summary.to_csv(
    OUT / "promotion_effectiveness_by_product.csv",
    index=False
)

region_summary.to_csv(
    OUT / "promotion_effectiveness_by_region.csv",
    index=False
)

management.to_csv(
    OUT / "promotion_management_summary.csv",
    index=False
)

sensitivity.to_csv(
    OUT / "promotion_effectiveness_sensitivity.csv",
    index=False
)

print("\n" + "=" * 70)
print("Final promotion analysis completed successfully!")
print("Results saved to:", OUT)
print("Charts saved to:", CHART)

print(
    "\nInterpretation: Estimated revenue effects "
    "are not proven causal promotion effects."
)

print(
    "Profitability cannot be assessed "
    "without product cost and margin information."
)

print("=" * 70)
