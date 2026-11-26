
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from dateutil.easter import easter

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)

RIDGE_ALPHA = 5.0
BLOCK_WEEKS = 4
MIN_TRAIN_OBSERVATIONS = 30

df = pd.read_csv(
    OUTPUT_DIR / "price_analysis_enriched.csv",
    parse_dates=["Week"]
)

df = df.sort_values(
    ["Series", "Week"]
).reset_index(drop=True)

data = df[
    df["Price_Analysis_Eligible"]
].copy().reset_index(drop=True)

ORIGIN = df["Week"].min()

print("=" * 70)
print("DEMAND PLANNING - BASELINE DEMAND ESTIMATION")
print("=" * 70)

# ============================================================
# 2. GENERATE CALENDAR VARIABLES
# ============================================================

def shopping_event_dates(year):

    november = pd.Timestamp(
        year=year,
        month=11,
        day=1
    )

    thursdays = pd.date_range(
        november,
        november + pd.offsets.MonthEnd(0),
        freq="W-THU"
    )

    thanksgiving = thursdays[3]

    black_friday_week = (
        thanksgiving -
        pd.Timedelta(days=3)
    )

    cyber_monday = (
        thanksgiving +
        pd.Timedelta(days=4)
    )

    return black_friday_week, cyber_monday


def christmas_dates(year):

    christmas = pd.Timestamp(
        year=year,
        month=12,
        day=25
    )

    christmas_week = (
        christmas -
        pd.Timedelta(days=christmas.weekday())
    )

    previous_week = (
        christmas_week -
        pd.Timedelta(weeks=1)
    )

    return previous_week, christmas_week


def easter_dates(year):

    easter_sunday = pd.Timestamp(
        easter(year)
    )

    easter_monday = (
        easter_sunday +
        pd.Timedelta(days=1)
    )

    previous_week = (
        easter_monday -
        pd.Timedelta(weeks=1)
    )

    return previous_week, easter_monday


def build_features(frame):

    features = pd.DataFrame(
        index=frame.index
    )

    elapsed_weeks = (
        (frame["Week"] - ORIGIN).dt.days / 7
    )

    features["Trend"] = elapsed_weeks

    # Smooth annual seasonal components.

    angle = (
        2 * np.pi *
        elapsed_weeks / 52.1775
    )

    features["Annual_Sin"] = np.sin(angle)
    features["Annual_Cos"] = np.cos(angle)

    # Relative price:
    # Zero means the observed price equals the reference.

    features["Log_Relative_Price"] = np.log(
        frame["Price"] /
        frame["Reference_Price"]
    )

    features["Shopping_Event"] = [
        int(week in shopping_event_dates(week.year))
        for week in frame["Week"]
    ]

    features["Christmas"] = [
        int(week in christmas_dates(week.year))
        for week in frame["Week"]
    ]

    features["Easter"] = [
        int(week in easter_dates(week.year))
        for week in frame["Week"]
    ]

    return features.astype(float)


# ============================================================
# 3. DEFINE PRIMARY AND ALTERNATIVE MODELS
# ============================================================

PRICE_FEATURES = [
    "Trend",
    "Annual_Sin",
    "Annual_Cos",
    "Log_Relative_Price",
    "Shopping_Event",
    "Christmas",
    "Easter"
]

CALENDAR_FEATURES = [
    "Trend",
    "Annual_Sin",
    "Annual_Cos",
    "Shopping_Event",
    "Christmas",
    "Easter"
]


def make_model():

    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA)
    )


# ============================================================
# 4. CREATE HISTORICAL VALIDATION BLOCKS
# ============================================================

# Each four-week block is excluded from training
# when estimating demand for that block.
#
# This is retrospective counterfactual estimation,
# NOT chronological future-forecast validation.

data["Validation_Block"] = (
    (data["Week"] - ORIGIN).dt.days //
    (BLOCK_WEEKS * 7)
)

results = []

print("\n1. MODEL ESTIMATION")

for series, group in data.groupby("Series"):

    group = group.sort_values("Week")

    print(
        f"Estimating baseline demand for {series} "
        f"({len(group)} observations)"
    )

    for block in sorted(
        group["Validation_Block"].unique()
    ):

        train = group[
            group["Validation_Block"] != block
        ].copy()

        test = group[
            group["Validation_Block"] == block
        ].copy()

        if len(train) < MIN_TRAIN_OBSERVATIONS:

            raise ValueError(
                f"Insufficient training data for "
                f"{series}, block {block}"
            )

        assert set(train["Week"]).isdisjoint(
            set(test["Week"])
        )

        X_train = build_features(train)
        X_test = build_features(test)

        # ----------------------------------------------------
        # PRIMARY MODEL: PRICE + TREND + SEASONALITY
        # ----------------------------------------------------

        model = make_model()

        model.fit(
            X_train[PRICE_FEATURES],
            train["Sales"]
        )

        predicted_actual_price = model.predict(
            X_test[PRICE_FEATURES]
        )

        # Counterfactual scenario:
        # Assume that observed price equals reference price.
        #
        # log(reference / reference) = 0

        X_reference = X_test.copy()

        X_reference["Log_Relative_Price"] = 0.0

        predicted_reference_price = model.predict(
            X_reference[PRICE_FEATURES]
        )

        test["Actual_Price_Prediction_Raw"] = (
            predicted_actual_price
        )

        test["Baseline_Demand_Raw"] = (
            predicted_reference_price
        )

        # Demand cannot be negative.

        test["Predicted_At_Actual_Price"] = np.maximum(
            0,
            predicted_actual_price
        )

        test["Baseline_Demand"] = np.maximum(
            0,
            predicted_reference_price
        )

        # Record whether the model is extrapolating
        # beyond the relative price range in training.

        minimum_ratio = X_train[
            "Log_Relative_Price"
        ].min()

        maximum_ratio = X_train[
            "Log_Relative_Price"
        ].max()

        test["Reference_Within_Training_Range"] = (
            minimum_ratio <= 0 <= maximum_ratio
        )

        test["Actual_Price_Within_Training_Range"] = (
            X_test["Log_Relative_Price"].between(
                minimum_ratio,
                maximum_ratio
            )
        )

        # ----------------------------------------------------
        # ALTERNATIVE MODEL:
        # TRAIN ONLY ON WEEKS WITHOUT FLAGGED DISCOUNTS
        # ----------------------------------------------------

        regular_train = train[
            ~train["Potential_Promotion"]
        ].copy()

        if len(regular_train) >= MIN_TRAIN_OBSERVATIONS:

            alternative_model = make_model()

            regular_features = build_features(
                regular_train
            )

            alternative_model.fit(
                regular_features[CALENDAR_FEATURES],
                regular_train["Sales"]
            )

            alternative_prediction = (
                alternative_model.predict(
                    X_test[CALENDAR_FEATURES]
                )
            )

            test["Alternative_Baseline"] = np.maximum(
                0,
                alternative_prediction
            )

        else:

            test["Alternative_Baseline"] = np.nan

        test["Training_Observations"] = len(train)

        test["Alternative_Training_Observations"] = (
            len(regular_train)
        )

        results.append(test)

# ============================================================
# 5. COMBINE ALL OUT-OF-BLOCK ESTIMATES
# ============================================================

estimates = pd.concat(
    results,
    ignore_index=True
)

estimates = estimates.sort_values(
    ["Series", "Week"]
).reset_index(drop=True)

# Difference between actual recorded sales
# and estimated regular-price demand.

estimates["Actual_Above_Baseline_Units"] = (
    estimates["Sales"] -
    estimates["Baseline_Demand"]
)

# Difference attributable to changing price
# within the statistical model.

estimates["Model_Implied_Price_Effect_Units"] = (
    estimates["Predicted_At_Actual_Price"] -
    estimates["Baseline_Demand"]
)

# Difference between actual sales and the model's
# prediction under the actual observed price.

estimates["Unexplained_Demand_Units"] = (
    estimates["Sales"] -
    estimates["Predicted_At_Actual_Price"]
)

# Avoid dividing by a zero baseline.

positive_baseline = estimates[
    "Baseline_Demand"
].where(
    estimates["Baseline_Demand"] > 0
)

estimates["Actual_Above_Baseline_Pct"] = (
    estimates["Actual_Above_Baseline_Units"] /
    positive_baseline * 100
)

estimates["Model_Implied_Price_Effect_Pct"] = (
    estimates["Model_Implied_Price_Effect_Units"] /
    positive_baseline * 100
)

estimates["Baseline_Model_Difference_Units"] = (
    estimates["Baseline_Demand"] -
    estimates["Alternative_Baseline"]
).abs()

# ============================================================
# 6. EVALUATE HISTORICAL MODEL PERFORMANCE
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = (
        np.isfinite(actual) &
        np.isfinite(predicted)
    )

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:

        return {
            "Observations": 0,
            "MAD": np.nan,
            "MSE": np.nan,
            "MAPE_Pct": np.nan,
            "Bias_Units": np.nan
        }

    # Course convention:
    # Positive errors mean overforecasting.

    errors = predicted - actual

    nonzero = actual > 0

    mape = (
        np.mean(
            np.abs(
                errors[nonzero] /
                actual[nonzero]
            )
        ) * 100
        if nonzero.any()
        else np.nan
    )

    return {
        "Observations": len(actual),
        "MAD": np.mean(np.abs(errors)),
        "MSE": np.mean(errors ** 2),
        "MAPE_Pct": mape,
        "Bias_Units": np.mean(errors)
    }


diagnostic_records = []

for series, group in estimates.groupby("Series"):

    metrics = calculate_metrics(
        group["Sales"],
        group["Predicted_At_Actual_Price"]
    )

    diagnostic_records.append({
        "Series": series,
        "Model": "Price-adjusted regression",
        "Evaluation_Sample": "All assessable weeks",
        **metrics
    })

    regular = group[
        ~group["Potential_Promotion"]
    ]

    # Compare both models on identical non-flagged weeks.

    comparable = regular[
        regular["Alternative_Baseline"].notna()
    ]

    primary_metrics = calculate_metrics(
        comparable["Sales"],
        comparable["Predicted_At_Actual_Price"]
    )

    alternative_metrics = calculate_metrics(
        comparable["Sales"],
        comparable["Alternative_Baseline"]
    )

    diagnostic_records.append({
        "Series": series,
        "Model": "Price-adjusted regression",
        "Evaluation_Sample": "Comparable non-flagged weeks",
        **primary_metrics
    })

    diagnostic_records.append({
        "Series": series,
        "Model": "Non-promotion calendar regression",
        "Evaluation_Sample": "Comparable non-flagged weeks",
        **alternative_metrics
    })

diagnostics = pd.DataFrame(
    diagnostic_records
)

print("\n2. HISTORICAL MODEL DIAGNOSTICS")

print(
    diagnostics.round(2).to_string(index=False)
)

print(
    "\nNote: These are retrospective out-of-block "
    "diagnostics, not future forecasting accuracy."
)

# ============================================================
# 7. SUMMARIZE POTENTIAL PROMOTIONS
# ============================================================

promotions = estimates[
    estimates["Potential_Promotion"]
].copy()

summary = promotions.groupby("Series").agg(
    Promotion_Weeks=("Week", "count"),
    Average_Discount_Pct=("Discount_Pct", "mean"),
    Actual_Sales=("Sales", "mean"),
    Estimated_Baseline=("Baseline_Demand", "mean"),
    Alternative_Baseline=("Alternative_Baseline", "mean"),
    Average_Actual_Uplift_Units=(
        "Actual_Above_Baseline_Units",
        "mean"
    ),
    Average_Model_Price_Effect=(
        "Model_Implied_Price_Effect_Units",
        "mean"
    ),
    Average_Baseline_Disagreement=(
        "Baseline_Model_Difference_Units",
        "mean"
    )
).reset_index()

print("\n3. BASELINE DEMAND BY SERIES")

print(
    summary.round(2).to_string(index=False)
)

# ============================================================
# 8. INVESTIGATE HIGHEST SALES WEEK
# ============================================================

peak_week = (
    df.groupby("Week")["Sales"]
    .sum()
    .idxmax()
)

peak_details = estimates[
    estimates["Week"] == peak_week
].copy()

peak_details = peak_details[
    [
        "Series",
        "Sales",
        "Price",
        "Reference_Price",
        "Discount_Pct",
        "Potential_Promotion",
        "Baseline_Demand",
        "Alternative_Baseline",
        "Predicted_At_Actual_Price",
        "Actual_Above_Baseline_Units",
        "Model_Implied_Price_Effect_Units",
        "Unexplained_Demand_Units"
    ]
]

print(
    f"\n4. HIGHEST SALES WEEK: "
    f"{peak_week.date()}"
)

print(
    peak_details.round(2).to_string(index=False)
)

# ============================================================
# 9. MODEL LIMITATIONS AND QUALITY FLAGS
# ============================================================

print("\n5. MODEL QUALITY FLAGS")

print(
    "Negative raw baseline predictions:",
    estimates["Baseline_Demand_Raw"].lt(0).sum()
)

print(
    "Zero baseline estimates:",
    estimates["Baseline_Demand"].eq(0).sum()
)

print(
    "Reference prices outside training range:",
    (~estimates[
        "Reference_Within_Training_Range"
    ]).sum()
)

print(
    "Observed relative prices outside training range:",
    (~estimates[
        "Actual_Price_Within_Training_Range"
    ]).sum()
)

print(
    "Unavailable alternative baselines:",
    estimates["Alternative_Baseline"].isna().sum()
)

# ============================================================
# 10. INTERACTIVE BASELINE DEMAND CHART
# ============================================================

fig = go.Figure()

series_names = sorted(
    estimates["Series"].unique()
)

for i, series in enumerate(series_names):

    group = estimates[
        estimates["Series"] == series
    ].sort_values("Week")

    promo = group[
        group["Potential_Promotion"]
    ]

    visible = i == 0

    fig.add_trace(
        go.Scatter(
            x=group["Week"],
            y=group["Sales"],
            name="Actual sales",
            mode="lines",
            line=dict(color="#365C8D"),
            visible=visible
        )
    )

    fig.add_trace(
        go.Scatter(
            x=group["Week"],
            y=group["Baseline_Demand"],
            name="Estimated regular-price demand",
            mode="lines",
            line=dict(
                color="#46A094",
                dash="dash"
            ),
            visible=visible
        )
    )

    fig.add_trace(
        go.Scatter(
            x=group["Week"],
            y=group["Alternative_Baseline"],
            name="Alternative baseline",
            mode="lines",
            line=dict(
                color="#9CA3AF",
                dash="dot"
            ),
            visible=visible
        )
    )

    fig.add_trace(
        go.Scatter(
            x=promo["Week"],
            y=promo["Sales"],
            name="Potential promotion",
            mode="markers",
            marker=dict(
                color="#C77C35",
                size=8
            ),
            visible=visible
        )
    )

buttons = []

for i, series in enumerate(series_names):

    visibility = [
        j // 4 == i
        for j in range(len(fig.data))
    ]

    buttons.append({
        "label": series,
        "method": "update",
        "args": [
            {"visible": visibility},
            {
                "title":
                    f"Historical Baseline Demand - {series}"
            }
        ]
    })

fig.update_layout(
    title=(
        f"Historical Baseline Demand - "
        f"{series_names[0]}"
    ),
    xaxis_title="Week",
    yaxis_title="Units sold",
    template="plotly_white",
    updatemenus=[
        {
            "buttons": buttons,
            "direction": "down",
            "x": 0,
            "y": 1.17
        }
    ],
    height=650
)

fig.write_html(
    CHART_DIR / "13_baseline_demand.html"
)

# ============================================================
# 11. VALIDATION
# ============================================================

assert len(estimates) == len(data)

assert not estimates.duplicated(
    ["Week", "Series"]
).any()

assert estimates["Baseline_Demand"].notna().all()

assert estimates["Baseline_Demand"].ge(0).all()

assert estimates[
    "Predicted_At_Actual_Price"
].ge(0).all()

assert len(promotions) == int(
    data["Potential_Promotion"].sum()
)

# Verify the demand decomposition.

assert np.allclose(
    estimates["Actual_Above_Baseline_Units"],
    estimates["Model_Implied_Price_Effect_Units"] +
    estimates["Unexplained_Demand_Units"]
)

# Independent test using the professor's
# forecast error convention.

test_metrics = calculate_metrics(
    [100],
    [120]
)

assert np.isclose(
    test_metrics["MAPE_Pct"], 20
)

assert np.isclose(
    test_metrics["MAD"], 20
)

assert np.isclose(
    test_metrics["MSE"], 400
)

assert np.isclose(
    test_metrics["Bias_Units"], 20
)

print("\n6. VALIDATION")

print("PASS - All assessable observations included.")
print("PASS - No duplicate observations.")
print("PASS - No missing primary baseline predictions.")
print("PASS - No negative demand estimates.")
print("PASS - All promotion candidates preserved.")
print("PASS - Demand decomposition verified.")
print("PASS - Forecast error metrics verified.")

# ============================================================
# 12. EXPORT RESULTS
# ============================================================

estimates.to_csv(
    OUTPUT_DIR / "baseline_demand_estimates.csv",
    index=False
)

diagnostics.to_csv(
    OUTPUT_DIR / "baseline_model_diagnostics.csv",
    index=False
)

promotions.to_csv(
    OUTPUT_DIR / "promotion_baseline_estimates.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "promotion_baseline_summary.csv",
    index=False
)

peak_details.to_csv(
    OUTPUT_DIR / "peak_week_baseline_analysis.csv",
    index=False
)

print("\n" + "=" * 70)
print("Baseline demand estimation completed successfully!")
print("Analytical tables saved to:", OUTPUT_DIR)
print("Interactive charts saved to:", CHART_DIR)
print("=" * 70)
