
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from dateutil.easter import easter

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CHART = OUT / "charts"

CHART.mkdir(parents=True, exist_ok=True)

HORIZON = 13
FINAL_TEST_WEEKS = 13
SEASONAL_PERIOD = 52

MODELS = [
    "Naive",
    "Seasonal naive",
    "Moving average 4",
    "Moving average 13",
    "SES",
    "Holt linear",
    "Holt damped",
    "Linear regression",
    "Seasonal calendar regression"
]

print("=" * 70)
print("DEMAND PLANNING - FORECASTING MODEL DEVELOPMENT")
print("=" * 70)

# ============================================================
# 2. LOAD HISTORICAL DATA
# ============================================================

# Use original sales observations.
# Do not use full-history lifecycle classifications,
# fitted baselines or outlier-based adjustments.

df = pd.read_csv(
    OUT / "historical_data_clean.csv",
    parse_dates=["Week"]
)

df = df[
    [
        "Week",
        "Series",
        "Region",
        "Product_Group",
        "Sales",
        "Price"
    ]
].copy()

df = df.sort_values(
    ["Series", "Week"]
).reset_index(drop=True)

assert not df.duplicated(
    ["Series", "Week"]
).any()

assert df["Sales"].notna().all()

assert df["Sales"].ge(0).all()

weeks = pd.DatetimeIndex(
    sorted(df["Week"].unique())
)

expected_weeks = pd.date_range(
    weeks.min(),
    weeks.max(),
    freq="W-MON"
)

assert weeks.equals(expected_weeks), (
    "Historical calendar is incomplete."
)

for series, group in df.groupby("Series"):

    series_weeks = pd.DatetimeIndex(
        group["Week"].sort_values()
    )

    assert series_weeks.equals(weeks), (
        f"Incomplete weekly history for {series}"
    )

# ============================================================
# 3. CHRONOLOGICAL DATA SPLIT
# ============================================================

development_end = (
    len(weeks) - FINAL_TEST_WEEKS
)

development_weeks = weeks[
    :development_end
]

final_test_weeks = weeks[
    development_end:
]

development = df[
    df["Week"].isin(development_weeks)
].copy()

# Three rolling forecast origins.
#
# For the current dataset:
# 60, 63 and 66 training weeks.

origins = [
    development_end - HORIZON - 6,
    development_end - HORIZON - 3,
    development_end - HORIZON
]

assert min(origins) >= SEASONAL_PERIOD

assert development["Week"].max() < (
    final_test_weeks.min()
)

print("\n1. DATA SPLIT")

print(
    "Historical weeks:",
    len(weeks)
)

print(
    "Development weeks:",
    len(development_weeks)
)

print(
    "Reserved final test weeks:",
    len(final_test_weeks)
)

print(
    "Development period:",
    development_weeks.min().date(),
    "to",
    development_weeks.max().date()
)

print(
    "Reserved test period:",
    final_test_weeks.min().date(),
    "to",
    final_test_weeks.max().date()
)

# ============================================================
# 4. DEFINE VALIDATION PERIODS
# ============================================================

fold_records = []

for fold, origin in enumerate(
    origins,
    start=1
):

    train_dates = weeks[:origin]

    validation_dates = weeks[
        origin:origin + HORIZON
    ]

    assert len(validation_dates) == HORIZON

    assert train_dates.max() < (
        validation_dates.min()
    )

    assert validation_dates.max() < (
        final_test_weeks.min()
    )

    fold_records.append({
        "Fold": fold,
        "Training_Weeks": len(train_dates),
        "Training_Start": train_dates.min(),
        "Training_End": train_dates.max(),
        "Validation_Start": validation_dates.min(),
        "Validation_End": validation_dates.max(),
        "Validation_Weeks": len(validation_dates)
    })

fold_summary = pd.DataFrame(
    fold_records
)

print("\n2. ROLLING VALIDATION PERIODS")

print(
    fold_summary.to_string(index=False)
)

# ============================================================
# 5. CALENDAR FEATURES
# ============================================================

def monday(date):

    return date - pd.Timedelta(
        days=date.weekday()
    )


def event_flags(week):

    year = week.year

    thursdays = pd.date_range(
        f"{year}-11-01",
        f"{year}-11-30",
        freq="W-THU"
    )

    thanksgiving = thursdays[3]

    black_friday_week = monday(
        thanksgiving
    )

    cyber_monday = (
        thanksgiving
        + pd.Timedelta(days=4)
    )

    christmas_week = monday(
        pd.Timestamp(year, 12, 25)
    )

    easter_monday = (
        pd.Timestamp(easter(year))
        + pd.Timedelta(days=1)
    )

    return [
        int(
            week in [
                black_friday_week,
                cyber_monday
            ]
        ),
        int(
            week in [
                christmas_week
                - pd.Timedelta(weeks=1),
                christmas_week
            ]
        ),
        int(
            week in [
                easter_monday
                - pd.Timedelta(weeks=1),
                easter_monday
            ]
        )
    ]


def calendar_features(dates, origin_date):

    dates = pd.DatetimeIndex(dates)

    t = np.asarray(
        (dates - origin_date).days / 7,
        dtype=float
    )

    angle = (
        2 * np.pi * t / 52.1775
    )

    events = np.array([
        event_flags(w)
        for w in dates
    ])

    return np.column_stack([
        t,
        np.sin(angle),
        np.cos(angle),
        events
    ])


# ============================================================
# 6. PREPARE TRAINING HISTORY
# ============================================================

def active_training_history(series_history):

    series_history = (
        series_history.sort_index()
    )

    positive = series_history[
        series_history > 0
    ]

    if positive.empty:

        raise ValueError(
            "No positive historical sales."
        )

    # Determine the first recorded sale using
    # only information available at this origin.

    first_sale = positive.index.min()

    active = series_history.loc[
        first_sale:
    ]

    # Retain active-period zero sales.
    # They may represent availability issues,
    # but this cannot be confirmed.

    assert active.notna().all()

    return active


# ============================================================
# 7. FORECASTING MODEL FUNCTIONS
# ============================================================

def forecast_model(
    model_name,
    train,
    forecast_dates
):

    train = train.sort_index()

    forecast_dates = pd.DatetimeIndex(
        forecast_dates
    )

    y = train.to_numpy(dtype=float)

    h = len(forecast_dates)

    assert train.index.max() < (
        forecast_dates.min()
    )

    # --------------------------------------------------------
    # MODEL 1: NAIVE
    # --------------------------------------------------------

    if model_name == "Naive":

        prediction = np.repeat(
            y[-1],
            h
        )

    # --------------------------------------------------------
    # MODEL 2: SEASONAL NAIVE
    # --------------------------------------------------------

    elif model_name == "Seasonal naive":

        lagged_dates = (
            forecast_dates
            - pd.Timedelta(
                weeks=SEASONAL_PERIOD
            )
        )

        lagged = train.reindex(
            lagged_dates
        )

        if lagged.isna().any():

            raise ValueError(
                "Insufficient active history "
                "for seasonal naive."
            )

        prediction = lagged.to_numpy(
            dtype=float
        )

    # --------------------------------------------------------
    # MODELS 3 AND 4: MOVING AVERAGES
    # --------------------------------------------------------

    elif model_name in [
        "Moving average 4",
        "Moving average 13"
    ]:

        window = (
            4 if model_name == "Moving average 4"
            else 13
        )

        if len(y) < window:

            raise ValueError(
                "Insufficient training history."
            )

        forecast_level = np.mean(
            y[-window:]
        )

        prediction = np.repeat(
            forecast_level,
            h
        )

    # --------------------------------------------------------
    # MODEL 5: SIMPLE EXPONENTIAL SMOOTHING
    # --------------------------------------------------------

    elif model_name == "SES":

        fitted = SimpleExpSmoothing(
            y,
            initialization_method="estimated"
        ).fit(
            optimized=True
        )

        prediction = fitted.forecast(h)

    # --------------------------------------------------------
    # MODEL 6: HOLT LINEAR
    # --------------------------------------------------------

    elif model_name == "Holt linear":

        fitted = Holt(
            y,
            damped_trend=False,
            initialization_method="estimated"
        ).fit(
            optimized=True
        )

        prediction = fitted.forecast(h)

    # --------------------------------------------------------
    # MODEL 7: DAMPED HOLT
    # --------------------------------------------------------

    elif model_name == "Holt damped":

        fitted = Holt(
            y,
            damped_trend=True,
            initialization_method="estimated"
        ).fit(
            optimized=True
        )

        prediction = fitted.forecast(h)

    # --------------------------------------------------------
    # MODEL 8: LINEAR TREND REGRESSION
    # --------------------------------------------------------

    elif model_name == "Linear regression":

        first_date = train.index.min()

        X_train = np.asarray(
            (train.index - first_date).days / 7
        ).reshape(-1, 1)

        X_future = np.asarray(
            (
                forecast_dates - first_date
            ).days / 7
        ).reshape(-1, 1)

        fitted = LinearRegression()

        fitted.fit(
            X_train,
            y
        )

        prediction = fitted.predict(
            X_future
        )

    # --------------------------------------------------------
    # MODEL 9: SEASONAL CALENDAR REGRESSION
    # --------------------------------------------------------

    elif model_name == "Seasonal calendar regression":

        first_date = train.index.min()

        X_train = calendar_features(
            train.index,
            first_date
        )

        X_future = calendar_features(
            forecast_dates,
            first_date
        )

        fitted = make_pipeline(
            StandardScaler(),
            Ridge(alpha=5.0)
        )

        fitted.fit(
            X_train,
            y
        )

        prediction = fitted.predict(
            X_future
        )

    else:

        raise ValueError(
            f"Unknown forecasting model: {model_name}"
        )

    prediction = np.asarray(
        prediction,
        dtype=float
    )

    assert len(prediction) == h

    assert np.isfinite(
        prediction
    ).all()

    # Preserve raw predictions for quality checks.
    # Operational demand forecasts cannot be negative.

    return prediction


# ============================================================
# 8. RUN ROLLING FORECAST VALIDATION
# ============================================================

results = []

print("\n3. MODEL EVALUATION")

for fold, origin in enumerate(
    origins,
    start=1
):

    forecast_dates = weeks[
        origin:origin + HORIZON
    ]

    training_end = weeks[origin - 1]

    print(
        f"\nFold {fold}: "
        f"Forecast origin {training_end.date()}"
    )

    for series, group in development.groupby(
        "Series"
    ):

        group = group.set_index(
            "Week"
        ).sort_index()

        historical = group.loc[
            group.index <= training_end,
            "Sales"
        ]

        train = active_training_history(
            historical
        )

        actual = group["Sales"].reindex(
            forecast_dates
        )

        assert actual.notna().all()

        print(
            f"  {series}: "
            f"{len(train)} active training weeks",
            flush=True
        )

        for model_name in MODELS:

            raw_prediction = forecast_model(
                model_name,
                train,
                forecast_dates
            )

            prediction = np.maximum(
                0,
                raw_prediction
            )

            for i, week in enumerate(
                forecast_dates
            ):

                results.append({
                    "Fold": fold,

                    "Forecast_Origin":
                        training_end,

                    "Week": week,

                    "Horizon_Week": i + 1,

                    "Series": series,

                    "Region":
                        group["Region"].iloc[0],

                    "Product_Group":
                        group["Product_Group"].iloc[0],

                    "Model": model_name,

                    "Training_Observations":
                        len(train),

                    "Actual":
                        float(actual.loc[week]),

                    "Forecast_Raw":
                        float(raw_prediction[i]),

                    "Forecast":
                        float(prediction[i])
                })

pred = pd.DataFrame(results)

# ============================================================
# 9. CALCULATE FORECAST ERRORS
# ============================================================

# Professor's convention:
# Positive error means overforecasting.

pred["Error"] = (
    pred["Forecast"]
    - pred["Actual"]
)

pred["Absolute_Error"] = (
    pred["Error"].abs()
)

pred["Squared_Error"] = (
    pred["Error"] ** 2
)

pred["Absolute_Percentage_Error"] = (
    pred["Absolute_Error"]
    / pred["Actual"].replace(0, np.nan)
    * 100
)


def calculate_metrics(group):

    actual = group["Actual"].to_numpy(
        dtype=float
    )

    error = group["Error"].to_numpy(
        dtype=float
    )

    absolute = np.abs(error)

    total_actual = actual.sum()

    positive = actual > 0

    return {
        "Observations": len(group),

        "MAPE_Observations":
            int(positive.sum()),

        "Zero_Actual_Observations":
            int((actual == 0).sum()),

        "MAPE_Pct": (
            np.mean(
                absolute[positive]
                / actual[positive]
            ) * 100
            if positive.any()
            else np.nan
        ),

        "WAPE_Pct": (
            absolute.sum()
            / total_actual * 100
            if total_actual > 0
            else np.nan
        ),

        "MAD":
            absolute.mean(),

        "MSE":
            np.mean(error ** 2),

        "Bias_Units":
            error.mean(),

        "Bias_Pct": (
            error.sum()
            / total_actual * 100
            if total_actual > 0
            else np.nan
        ),

        "Cumulative_Error":
            error.sum(),

        "Negative_Raw_Forecasts":
            int(
                group["Forecast_Raw"]
                .lt(0).sum()
            )
    }


def summarize_metrics(frame, grouping):

    rows = []

    for keys, group in frame.groupby(
        grouping
    ):

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(
            zip(grouping, keys)
        )

        row.update(
            calculate_metrics(group)
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 10. ACCURACY BY SERIES AND MODEL
# ============================================================

series_metrics = summarize_metrics(
    pred,
    ["Series", "Model"]
)

fold_metrics = summarize_metrics(
    pred,
    ["Fold", "Series", "Model"]
)

overall_metrics = summarize_metrics(
    pred,
    ["Model"]
)

overall_metrics = overall_metrics.sort_values(
    "MAPE_Pct"
)

print("\n4. OVERALL MODEL PERFORMANCE")

print(
    overall_metrics[
        [
            "Model",
            "Observations",
            "MAPE_Pct",
            "WAPE_Pct",
            "MAD",
            "MSE",
            "Bias_Units",
            "Bias_Pct"
        ]
    ].round(2).to_string(index=False)
)

# ============================================================
# 11. COMPARE MAPE AND WAPE
# ============================================================

mape_table = series_metrics.pivot(
    index="Series",
    columns="Model",
    values="MAPE_Pct"
)

wape_table = series_metrics.pivot(
    index="Series",
    columns="Model",
    values="WAPE_Pct"
)

print("\n5. MAPE BY SERIES (%)")

print(
    mape_table.round(2).to_string()
)

print("\n6. WAPE BY SERIES (%)")

print(
    wape_table.round(2).to_string()
)

# ============================================================
# 12. PROVISIONAL MODEL RANKINGS
# ============================================================

# These are development-stage winners.
# Final model selection will also include
# advanced price-based forecasting candidates.

best_mape = series_metrics.loc[
    series_metrics.groupby(
        "Series"
    )["MAPE_Pct"].idxmin()
].copy()

best_wape = series_metrics.loc[
    series_metrics.groupby(
        "Series"
    )["WAPE_Pct"].idxmin()
].copy()

winners = best_mape[
    [
        "Series",
        "Model",
        "MAPE_Pct",
        "WAPE_Pct",
        "Bias_Units"
    ]
].rename(
    columns={
        "Model": "Best_MAPE_Model",
        "WAPE_Pct": "WAPE_of_MAPE_Winner"
    }
)

winners = winners.merge(
    best_wape[
        [
            "Series",
            "Model",
            "WAPE_Pct"
        ]
    ].rename(
        columns={
            "Model": "Best_WAPE_Model",
            "WAPE_Pct": "Best_WAPE_Pct"
        }
    ),
    on="Series"
)

naive = series_metrics[
    series_metrics["Model"] == "Naive"
][
    ["Series", "MAPE_Pct"]
].rename(
    columns={
        "MAPE_Pct": "Naive_MAPE_Pct"
    }
)

winners = winners.merge(
    naive,
    on="Series"
)

winners["MAPE_Improvement_pp"] = (
    winners["Naive_MAPE_Pct"]
    - winners["MAPE_Pct"]
)

print("\n7. PROVISIONAL MODEL WINNERS")

print(
    winners.round(2).to_string(index=False)
)

# ============================================================
# 13. PERFORMANCE BY FORECAST HORIZON
# ============================================================

horizon_metrics = summarize_metrics(
    pred,
    ["Model", "Horizon_Week"]
)

# ============================================================
# 14. CREATE MODEL COMPARISON CHARTS
# ============================================================

fig1 = px.imshow(
    mape_table.reindex(columns=MODELS),
    text_auto=".1f",
    color_continuous_scale="RdYlGn_r",
    aspect="auto",
    title="Rolling Validation: MAPE by Model and Series",
    labels={
        "x": "Forecasting model",
        "y": "Demand series",
        "color": "MAPE (%)"
    },
    template="plotly_white",
    height=650
)

fig1.write_html(
    CHART / "18_forecast_mape_comparison.html"
)

fig2 = px.imshow(
    wape_table.reindex(columns=MODELS),
    text_auto=".1f",
    color_continuous_scale="RdYlGn_r",
    aspect="auto",
    title="Rolling Validation: WAPE by Model and Series",
    labels={
        "x": "Forecasting model",
        "y": "Demand series",
        "color": "WAPE (%)"
    },
    template="plotly_white",
    height=650
)

fig2.write_html(
    CHART / "19_forecast_wape_comparison.html"
)

fig3 = px.line(
    horizon_metrics,
    x="Horizon_Week",
    y="WAPE_Pct",
    color="Model",
    markers=True,
    title="Forecast Error by Prediction Horizon",
    labels={
        "Horizon_Week": "Weeks ahead",
        "WAPE_Pct": "WAPE (%)"
    },
    template="plotly_white"
)

fig3.write_html(
    CHART / "20_forecast_horizon_accuracy.html"
)

# ============================================================
# 15. VALIDATION
# ============================================================

expected_predictions = (
    len(origins)
    * HORIZON
    * df["Series"].nunique()
    * len(MODELS)
)

assert len(pred) == expected_predictions

assert not pred.duplicated(
    [
        "Fold",
        "Week",
        "Series",
        "Model"
    ]
).any()

# Ensure identical evaluation observations
# are used for every forecasting model.

assert pred.groupby(
    ["Fold", "Series", "Week"]
)["Model"].nunique().eq(
    len(MODELS)
).all()

# Confirm no forecast uses future training data.

assert (
    pred["Forecast_Origin"]
    < pred["Week"]
).all()

# Confirm the final test is unused.

assert pred["Week"].max() < (
    final_test_weeks.min()
)

assert pred["Forecast"].ge(0).all()

assert np.isfinite(
    pred["Forecast"]
).all()

# Independently verify the forecast metrics.

test = pd.DataFrame({
    "Actual": [100.0],
    "Forecast": [120.0],
    "Forecast_Raw": [120.0],
    "Error": [20.0]
})

test_metrics = calculate_metrics(test)

assert np.isclose(
    test_metrics["MAPE_Pct"], 20
)

assert np.isclose(
    test_metrics["WAPE_Pct"], 20
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

# Independently verify simple forecasting methods.

test_train = pd.Series(
    [10.0, 20.0, 30.0, 40.0],
    index=pd.date_range(
        "2020-01-06",
        periods=4,
        freq="W-MON"
    )
)

test_dates = pd.date_range(
    "2020-02-03",
    periods=3,
    freq="W-MON"
)

assert np.allclose(
    forecast_model(
        "Naive",
        test_train,
        test_dates
    ),
    [40, 40, 40]
)

assert np.allclose(
    forecast_model(
        "Moving average 4",
        test_train,
        test_dates
    ),
    [25, 25, 25]
)

print("\n8. VALIDATION")

print("PASS - All forecasting models included.")
print("PASS - Identical evaluation observations.")
print("PASS - Chronological training and validation.")
print("PASS - Reserved final test period unused.")
print("PASS - No duplicate predictions.")
print("PASS - All operational forecasts nonnegative.")
print("PASS - All predictions finite.")
print("PASS - Forecast error metrics verified.")
print("PASS - Independent model calculations verified.")

# ============================================================
# 16. EXPORT RESULTS
# ============================================================

pred.to_csv(
    OUT / "forecast_classical_predictions.csv",
    index=False
)

series_metrics.to_csv(
    OUT / "forecast_classical_series_metrics.csv",
    index=False
)

fold_metrics.to_csv(
    OUT / "forecast_classical_fold_metrics.csv",
    index=False
)

overall_metrics.to_csv(
    OUT / "forecast_classical_overall_metrics.csv",
    index=False
)

winners.to_csv(
    OUT / "forecast_classical_winners.csv",
    index=False
)

horizon_metrics.to_csv(
    OUT / "forecast_classical_horizon_metrics.csv",
    index=False
)

fold_summary.to_csv(
    OUT / "forecast_validation_periods.csv",
    index=False
)

print("\n" + "=" * 70)

print(
    "Classical forecasting evaluation "
    "completed successfully!"
)

print(
    "Total forecast predictions:",
    len(pred)
)

print("Results saved to:", OUT)
print("Charts saved to:", CHART)

print(
    "\nFinal historical test period "
    "remains reserved."
)

print(
    "Model rankings are provisional "
    "until advanced candidates are evaluated."
)

print("=" * 70)
