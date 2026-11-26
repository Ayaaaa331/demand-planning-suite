
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
import plotly.express as px

# Reuse existing forecasting engine.
engine = import_module("12_final_model_evaluation")

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CHART = OUT / "charts"

CHART.mkdir(parents=True, exist_ok=True)

SUBMISSION_HORIZON = 13
EXTENDED_HORIZON = 52

print("=" * 70)
print("DEMAND PLANNING - FINAL FUTURE FORECASTS")
print("=" * 70)

# ============================================================
# 2. LOAD HISTORICAL DATA AND MODEL SELECTION
# ============================================================

history = pd.read_csv(
    OUT / "historical_data_clean.csv",
    parse_dates=["Week"]
)

history = history[
    [
        "Week",
        "Series",
        "Region",
        "Product_Group",
        "Sales",
        "Price"
    ]
].copy()

history = history.sort_values(
    ["Series", "Week"]
)

future_input = pd.read_csv(
    OUT / "future_data.csv",
    parse_dates=["Week"]
)

selection = pd.read_csv(
    OUT / "final_model_selection.csv"
)

development_metrics = pd.read_csv(
    OUT / "forecast_combined_series_metrics.csv"
)

test_metrics = pd.read_csv(
    OUT / "forecast_final_test_selected_metrics.csv"
)

model_names = sorted(
    development_metrics["Model"].unique()
)

series_names = sorted(
    history["Series"].unique()
)

n_series = len(series_names)

assert not history.duplicated(
    ["Series", "Week"]
).any()

assert history["Sales"].notna().all()
assert history["Sales"].ge(0).all()

assert selection["Series"].is_unique

assert set(selection["Series"]) == set(
    series_names
)

# ============================================================
# 3. DEFINE FUTURE FORECAST DATES
# ============================================================

forecast_origin = history["Week"].max()

future_dates = pd.date_range(
    forecast_origin + pd.Timedelta(weeks=1),
    periods=EXTENDED_HORIZON,
    freq="W-MON"
)

submission_dates = future_dates[
    :SUBMISSION_HORIZON
]

provided_dates = pd.DatetimeIndex(
    sorted(future_input["Week"].unique())
)

assert submission_dates.equals(
    provided_dates
), "Forecast dates do not match input workbook."

assert not future_input.duplicated(
    ["Series", "Week"]
).any()

assert len(future_input) == (
    n_series * SUBMISSION_HORIZON
)

assert set(
    future_input["Series"]
) == set(series_names)

assert future_input["Sales"].isna().all()

# This dataset has no supplied future prices.
# The existing engine estimates future prices
# using historical information only.

assert future_input["Price"].isna().all(), (
    "Future prices are supplied. Review price assumptions."
)

print("\n1. FINAL FORECAST DESIGN")

print(
    "Historical weeks:",
    history["Week"].nunique()
)

print(
    "Historical observations:",
    len(history)
)

print(
    "Forecast origin:",
    forecast_origin.date()
)

print(
    "Submission forecast:",
    submission_dates.min().date(),
    "to",
    submission_dates.max().date()
)

print(
    "Extended forecast:",
    future_dates.min().date(),
    "to",
    future_dates.max().date()
)

print(
    "Candidate models:",
    len(model_names)
)

# ============================================================
# 4. REFIT ALL FORECASTING MODELS
# ============================================================

# All historical observations are now available
# for final future forecasting.
#
# Model selection remains unchanged.

print("\n2. FINAL MODEL ESTIMATION")

predictions, fit_audit, prices = (
    engine.generate_forecasts(
        history,
        future_dates,
        model_names
    )
)

print("\n3. MODEL FIT AUDIT")

fit_summary = fit_audit.groupby(
    "Model"
).agg(
    Attempted_Fits=("Success", "count"),
    Successful_Fits=("Success", "sum")
).reset_index()

fit_summary["Failed_Fits"] = (
    fit_summary["Attempted_Fits"]
    - fit_summary["Successful_Fits"]
)

print(
    fit_summary.to_string(index=False)
)

failures = fit_audit[
    ~fit_audit["Success"]
]

if not failures.empty:

    print("\nUnresolved fit failures:")

    print(
        failures.to_string(index=False)
    )

fit_audit.to_csv(
    OUT / "final_future_fit_audit.csv",
    index=False
)

# ============================================================
# 5. CHECK MODEL COVERAGE
# ============================================================

expected_per_model = (
    n_series * EXTENDED_HORIZON
)

coverage = predictions.groupby(
    "Model"
).size().rename(
    "Available_Forecasts"
).reset_index()

coverage["Expected_Forecasts"] = (
    expected_per_model
)

coverage["Complete"] = (
    coverage["Available_Forecasts"]
    == expected_per_model
)

complete_models = coverage.loc[
    coverage["Complete"],
    "Model"
].tolist()

incomplete_models = sorted(
    set(model_names)
    - set(complete_models)
)

print("\n4. MODEL FORECAST COVERAGE")

print(
    coverage.to_string(index=False)
)

print(
    "Incomplete models:",
    incomplete_models
    if incomplete_models
    else "None"
)

# ============================================================
# 6. EXTRACT THE PRESELECTED MODELS
# ============================================================

joined = predictions.merge(
    selection[
        [
            "Series",
            "Selected_Model",
            "Best_WAPE_Model",
            "Development_MAPE",
            "Development_WAPE",
            "Development_Bias_Pct",
            "Planning_Warning"
        ]
    ],
    on="Series",
    validate="many_to_one"
)

selected = joined[
    joined["Model"]
    == joined["Selected_Model"]
].copy()

selected = selected.sort_values(
    ["Series", "Week"]
).reset_index(drop=True)

# No silent fallback to an unselected model.

assert len(selected) == (
    n_series * EXTENDED_HORIZON
), "Selected forecasts are incomplete."

assert selected.groupby(
    "Series"
)["Week"].nunique().eq(
    EXTENDED_HORIZON
).all()

# ============================================================
# 7. ATTACH HISTORICAL TEST ACCURACY
# ============================================================

accuracy = test_metrics[
    [
        "Series",
        "MAPE_Pct",
        "WAPE_Pct",
        "MAD",
        "Bias_Pct"
    ]
].rename(
    columns={
        "MAPE_Pct": "Historical_Test_MAPE",
        "WAPE_Pct": "Historical_Test_WAPE",
        "MAD": "Historical_Test_MAD",
        "Bias_Pct": "Historical_Test_Bias_Pct"
    }
)

selected = selected.merge(
    accuracy,
    on="Series",
    validate="many_to_one"
)

# These metrics describe historical performance,
# not measured accuracy of future predictions.

selected["Historical_Performance_Warning"] = (
    np.select(
        [
            selected["Historical_Test_WAPE"] > 50,
            selected["Historical_Test_WAPE"] > 35
        ],
        [
            "High historical forecast error",
            "Material historical forecast error"
        ],
        default=""
    )
)

# ============================================================
# 8. DOCUMENT PRICE ASSUMPTIONS
# ============================================================

price_assumptions = prices.rename(
    columns={
        "Assumed_Price":
            "Default_Assumed_Price_EUR"
    }
)

selected = selected.merge(
    price_assumptions[
        [
            "Series",
            "Default_Assumed_Price_EUR",
            "Price_Assumption",
            "Training_Observations"
        ]
    ],
    on="Series",
    validate="many_to_one"
)

selected["Uses_Price"] = (
    selected["Model"].isin(
        engine.PRICE_MODELS
    )
)

selected["Price_Used_By_Model_EUR"] = (
    selected["Assumed_Price"]
)

selected["Price_Assumption_Used"] = (
    np.where(
        selected["Uses_Price"],
        selected["Price_Assumption"],
        "Selected model does not use price"
    )
)

print("\n5. FINAL PRICE ASSUMPTIONS")

print(
    price_assumptions[
        [
            "Series",
            "Default_Assumed_Price_EUR",
            "Price_Assumption",
            "Training_Observations"
        ]
    ].round(2).to_string(index=False)
)

# ============================================================
# 9. PREPARE OPERATIONAL FORECAST QUANTITIES
# ============================================================

# Preserve original decimal predictions
# and generate whole-unit planning quantities.

selected["Forecast_Units"] = np.rint(
    selected["Forecast"]
).astype(int)

selected["Forecast_Period"] = np.where(
    selected["Horizon_Week"] <= SUBMISSION_HORIZON,
    "Submission forecast",
    "Extended forecast"
)

selected_13 = selected[
    selected["Horizon_Week"]
    <= SUBMISSION_HORIZON
].copy()

selected_52 = selected.copy()

# ============================================================
# 10. PREPARE ALTERNATIVE WAPE-SELECTED FORECASTS
# ============================================================

alternative = joined[
    joined["Model"]
    == joined["Best_WAPE_Model"]
].copy()

assert len(alternative) == (
    n_series * EXTENDED_HORIZON
)

alternative["Forecast_Units"] = np.rint(
    alternative["Forecast"]
).astype(int)

alternative_13 = alternative[
    alternative["Horizon_Week"]
    <= SUBMISSION_HORIZON
].copy()

# ============================================================
# 11. GENERATE THE SUBMISSION CSV
# ============================================================

# Match the structure of the original sales sheet:
# one row per week, one column per demand series.

submission = selected_13.pivot(
    index="Week",
    columns="Series",
    values="Forecast_Units"
)

submission = submission.reindex(
    index=submission_dates,
    columns=series_names
)

assert submission.notna().all().all()

submission = submission.astype(int)

submission.index.name = "Week"

submission = submission.reset_index()

submission.columns.name = None

# ============================================================
# 12. GENERATE EXTENDED WIDE FORECAST
# ============================================================

wide_52 = selected_52.pivot(
    index="Week",
    columns="Series",
    values="Forecast_Units"
)

wide_52 = wide_52.reindex(
    index=future_dates,
    columns=series_names
)

assert wide_52.notna().all().all()

wide_52 = wide_52.astype(int)

wide_52.index.name = "Week"

wide_52 = wide_52.reset_index()

wide_52.columns.name = None

# ============================================================
# 13. SUMMARIZE FORECAST DEMAND
# ============================================================

def forecast_summary(frame, grouping):

    return frame.groupby(
        grouping
    ).agg(
        Forecast_Weeks=("Week", "nunique"),
        Total_Forecast_Units=("Forecast_Units", "sum"),
        Average_Weekly_Units=("Forecast_Units", "mean"),
        Minimum_Weekly_Units=("Forecast_Units", "min"),
        Maximum_Weekly_Units=("Forecast_Units", "max")
    ).reset_index()


summary_13 = forecast_summary(
    selected_13,
    ["Series", "Region", "Product_Group", "Model"]
)

summary_52 = forecast_summary(
    selected_52,
    ["Series", "Region", "Product_Group", "Model"]
)

product_summary = forecast_summary(
    selected_13,
    ["Product_Group"]
)

region_summary = forecast_summary(
    selected_13,
    ["Region"]
)

# ============================================================
# 14. COMPARE AGAINST RECENT HISTORICAL DEMAND
# ============================================================

recent_dates = pd.DatetimeIndex(
    sorted(history["Week"].unique())
)[-SUBMISSION_HORIZON:]

recent_history = history[
    history["Week"].isin(recent_dates)
]

recent_totals = recent_history.groupby(
    "Series"
)["Sales"].sum().rename(
    "Previous_13_Weeks_Sales"
).reset_index()

summary_13 = summary_13.merge(
    recent_totals,
    on="Series",
    validate="one_to_one"
)

summary_13["Change_From_Previous_13_Weeks_Pct"] = (
    (
        summary_13["Total_Forecast_Units"]
        / summary_13[
            "Previous_13_Weeks_Sales"
        ].replace(0, np.nan)
    ) - 1
) * 100

summary_13 = summary_13.merge(
    accuracy,
    on="Series",
    validate="one_to_one"
)

print("\n6. FINAL 13-WEEK FORECAST BY SERIES")

print(
    summary_13.round(2).to_string(index=False)
)

print("\n7. FINAL FORECAST BY PRODUCT")

print(
    product_summary.round(2).to_string(index=False)
)

print("\n8. FINAL FORECAST BY REGION")

print(
    region_summary.round(2).to_string(index=False)
)

print("\n9. SUBMISSION CSV PREVIEW")

print(
    submission.to_string(index=False)
)

print(
    "\nTotal 13-week forecast units:",
    f"{selected_13['Forecast_Units'].sum():,}"
)

print(
    "Total 52-week forecast units:",
    f"{selected_52['Forecast_Units'].sum():,}"
)

# ============================================================
# 15. COMPARE ALTERNATIVE PLANNING STRATEGIES
# ============================================================

alternative_totals = alternative_13.groupby(
    "Series"
)["Forecast_Units"].sum().rename(
    "WAPE_Selected_Forecast_Units"
).reset_index()

strategy_comparison = summary_13[
    [
        "Series",
        "Model",
        "Total_Forecast_Units"
    ]
].rename(
    columns={
        "Model": "MAPE_Selected_Model",
        "Total_Forecast_Units":
            "MAPE_Selected_Forecast_Units"
    }
)

strategy_comparison = strategy_comparison.merge(
    alternative_totals,
    on="Series",
    validate="one_to_one"
)

strategy_comparison = strategy_comparison.merge(
    selection[
        ["Series", "Best_WAPE_Model"]
    ],
    on="Series",
    validate="one_to_one"
)

strategy_comparison["Difference_Units"] = (
    strategy_comparison[
        "WAPE_Selected_Forecast_Units"
    ]
    - strategy_comparison[
        "MAPE_Selected_Forecast_Units"
    ]
)

print("\n10. ALTERNATIVE PLANNING STRATEGIES")

print(
    strategy_comparison.to_string(index=False)
)

# ============================================================
# 16. FORECAST ASSUMPTIONS TABLE
# ============================================================

assumptions = selection.merge(
    price_assumptions,
    on="Series",
    validate="one_to_one"
)

assumptions = assumptions.merge(
    accuracy,
    on="Series",
    validate="one_to_one"
)

assumptions["Uses_Price"] = (
    assumptions["Selected_Model"].isin(
        engine.PRICE_MODELS
    )
)

assumptions["Submission_Horizon_Weeks"] = (
    SUBMISSION_HORIZON
)

assumptions["Extended_Horizon_Weeks"] = (
    EXTENDED_HORIZON
)

assumptions["Forecast_Interpretation"] = (
    "Expected weekly sales, not inventory order quantities"
)

assumptions["Future_Price_Policy"] = (
    "Constant assumed price; no confirmed future promotions"
)

assumptions["Extended_Horizon_Warning"] = (
    "52-week horizon exceeds validated 13-week horizon"
)

# ============================================================
# 17. CHART - HISTORICAL AND FUTURE DEMAND
# ============================================================

historical_chart = history[
    ["Week", "Series", "Sales"]
].rename(
    columns={"Sales": "Units"}
)

historical_chart["Measure"] = (
    "Historical actual"
)

forecast_chart = selected_52[
    [
        "Week",
        "Series",
        "Forecast",
        "Forecast_Period"
    ]
].rename(
    columns={
        "Forecast": "Units",
        "Forecast_Period": "Measure"
    }
)

chart_data = pd.concat(
    [
        historical_chart,
        forecast_chart
    ],
    ignore_index=True
)

fig1 = px.line(
    chart_data.sort_values("Week"),
    x="Week",
    y="Units",
    color="Measure",
    facet_col="Series",
    facet_col_wrap=2,
    title="Historical Demand and 52-Week Forecast",
    labels={"Units": "Weekly units"},
    color_discrete_map={
        "Historical actual": "#365C8D",
        "Submission forecast": "#46A094",
        "Extended forecast": "#C77C35"
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
    CHART / "28_final_52_week_forecasts.html"
)

# ============================================================
# 18. CHART - SUBMISSION FORECAST BY PRODUCT
# ============================================================

fig2 = px.bar(
    summary_13,
    x="Product_Group",
    y="Total_Forecast_Units",
    color="Region",
    barmode="group",
    title="13-Week Forecast by Product and Region",
    labels={
        "Product_Group": "Product group",
        "Total_Forecast_Units": "Forecast units"
    },
    color_discrete_map={
        "North": "#365C8D",
        "South": "#46A094"
    },
    template="plotly_white"
)

fig2.write_html(
    CHART / "29_submission_forecast_by_product.html"
)

# ============================================================
# 19. CHART - RECENT SALES VS NEXT 13 WEEKS
# ============================================================

comparison_chart = summary_13.melt(
    id_vars=["Series"],
    value_vars=[
        "Previous_13_Weeks_Sales",
        "Total_Forecast_Units"
    ],
    var_name="Period",
    value_name="Units"
)

comparison_chart["Period"] = (
    comparison_chart["Period"].replace({
        "Previous_13_Weeks_Sales":
            "Previous 13 weeks actual",
        "Total_Forecast_Units":
            "Next 13 weeks forecast"
    })
)

fig3 = px.bar(
    comparison_chart,
    x="Series",
    y="Units",
    color="Period",
    barmode="group",
    title="Recent Historical Demand vs Future Forecast",
    template="plotly_white"
)

fig3.write_html(
    CHART / "30_recent_sales_vs_forecast.html"
)

# ============================================================
# 20. VALIDATION
# ============================================================

assert len(selected_13) == (
    n_series * SUBMISSION_HORIZON
)

assert len(selected_52) == (
    n_series * EXTENDED_HORIZON
)

assert not selected_52.duplicated(
    ["Series", "Week"]
).any()

assert selected_52["Model"].eq(
    selected_52["Selected_Model"]
).all()

assert selected_52["Forecast_Units"].ge(0).all()

assert np.isfinite(
    selected_52["Forecast"]
).all()

assert selected_52["Week"].min() > (
    forecast_origin
)

assert submission.shape == (
    SUBMISSION_HORIZON,
    n_series + 1
)

assert wide_52.shape == (
    EXTENDED_HORIZON,
    n_series + 1
)

assert (
    submission[series_names].to_numpy().sum()
    == selected_13["Forecast_Units"].sum()
)

assert (
    wide_52[series_names].to_numpy().sum()
    == selected_52["Forecast_Units"].sum()
)

assert (
    product_summary["Total_Forecast_Units"].sum()
    == region_summary["Total_Forecast_Units"].sum()
)

assert price_assumptions[
    "Default_Assumed_Price_EUR"
].gt(0).all()

# Check price assumptions independently.

for series, group in history.groupby("Series"):

    train = engine.engine.prepare_training(
        group,
        forecast_origin
    )

    observed = train["Price"].where(
        train["Price"] > 0
    )

    recent = observed.tail(13).dropna()

    expected_price = (
        recent.median()
        if not recent.empty
        else observed.dropna().median()
    )

    actual_price = price_assumptions.loc[
        price_assumptions["Series"] == series,
        "Default_Assumed_Price_EUR"
    ].iloc[0]

    assert np.isclose(
        expected_price,
        actual_price
    )

print("\n11. VALIDATION")

print("PASS - All selected models preserved.")
print("PASS - All submission forecasts generated.")
print("PASS - All extended forecasts generated.")
print("PASS - Dates match the original workbook.")
print("PASS - No duplicate forecasts.")
print("PASS - Forecasts finite and nonnegative.")
print("PASS - Wide and long forecast totals reconcile.")
print("PASS - Product and regional totals reconcile.")
print("PASS - Future prices independently verified.")

# ============================================================
# 21. EXPORT FINAL FORECAST FILES
# ============================================================

submission.to_csv(
    OUT / "final_forecast_13_weeks.csv",
    index=False,
    date_format="%Y-%m-%d"
)

wide_52.to_csv(
    OUT / "final_forecast_52_weeks.csv",
    index=False,
    date_format="%Y-%m-%d"
)

selected_13.to_csv(
    OUT / "final_forecast_13_weeks_detailed.csv",
    index=False
)

selected_52.to_csv(
    OUT / "final_forecast_52_weeks_detailed.csv",
    index=False
)

predictions.to_csv(
    OUT / "final_future_all_model_predictions.csv",
    index=False
)

alternative.to_csv(
    OUT / "final_future_wape_alternative.csv",
    index=False
)

summary_13.to_csv(
    OUT / "final_forecast_13_weeks_summary.csv",
    index=False
)

summary_52.to_csv(
    OUT / "final_forecast_52_weeks_summary.csv",
    index=False
)

product_summary.to_csv(
    OUT / "final_forecast_by_product.csv",
    index=False
)

region_summary.to_csv(
    OUT / "final_forecast_by_region.csv",
    index=False
)

assumptions.to_csv(
    OUT / "final_forecast_assumptions.csv",
    index=False
)

strategy_comparison.to_csv(
    OUT / "final_forecast_strategy_comparison.csv",
    index=False
)

coverage.to_csv(
    OUT / "final_future_model_coverage.csv",
    index=False
)

print("\n" + "=" * 70)

print(
    "Final future forecasting completed successfully!"
)

print(
    "13-week selected forecasts:",
    len(selected_13)
)

print(
    "52-week selected forecasts:",
    len(selected_52)
)

print(
    "All candidate model forecasts:",
    len(predictions)
)

print(
    "Submission file:",
    OUT / "final_forecast_13_weeks.csv"
)

print("Results saved to:", OUT)
print("Charts saved to:", CHART)

print(
    "\nThe 13-week forecast uses the frozen "
    "development-based model selection."
)

print(
    "The 52-week forecast is an extended "
    "planning scenario, not a validated "
    "52-week prediction."
)

print("=" * 70)
