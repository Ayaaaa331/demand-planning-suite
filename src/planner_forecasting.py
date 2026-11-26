
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd

# Reuse our existing forecasting engines.

evaluation_engine = import_module(
    "12_final_model_evaluation"
)

advanced_engine = import_module(
    "11_advanced_forecasting"
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

MODELS = [
    "Naive",
    "Seasonal naive",
    "Moving average 4",
    "Moving average 13",
    "Moving median 13",
    "SES",
    "Holt linear",
    "Holt damped",
    "Linear regression",
    "Seasonal calendar regression",
    "Log calendar regression",
    "Price Ridge regression",
    "Log price Ridge regression",
    "Pooled log price regression",
    "ARIMAX (1,0,0)"
]

PRICE_MODELS = (
    evaluation_engine.PRICE_MODELS
)

REQUIRED_COLUMNS = [
    "Week",
    "Series",
    "Region",
    "Product_Group",
    "Sales",
    "Price"
]


# ============================================================
# 1. VALIDATE UPLOADED HISTORICAL DATA
# ============================================================

def validate_history(history):

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in history.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = history[
        REQUIRED_COLUMNS
    ].copy()

    df["Week"] = pd.to_datetime(
        df["Week"],
        errors="raise"
    )

    df["Sales"] = pd.to_numeric(
        df["Sales"],
        errors="raise"
    )

    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="raise"
    )

    if df.empty:

        raise ValueError(
            "No historical observations."
        )

    if df[
        ["Week", "Series", "Region", "Product_Group"]
    ].isna().any().any():

        raise ValueError(
            "Missing dates or series identifiers."
        )

    if df.duplicated(
        ["Week", "Series"]
    ).any():

        raise ValueError(
            "Duplicate series-week observations."
        )

    if df["Sales"].isna().any():

        raise ValueError(
            "Historical sales contain missing values."
        )

    if not np.isfinite(
        df["Sales"]
    ).all():

        raise ValueError(
            "Sales must be finite."
        )

    if df["Sales"].lt(0).any():

        raise ValueError(
            "Negative sales are not supported."
        )

    observed_prices = df["Price"].dropna()

    if not np.isfinite(
        observed_prices
    ).all():

        raise ValueError(
            "Prices must be finite."
        )

    if observed_prices.le(0).any():

        raise ValueError(
            "Observed prices must be positive."
        )

    if not df["Week"].dt.day_of_week.eq(0).all():

        raise ValueError(
            "Weekly dates must be Mondays."
        )

    df = df.sort_values(
        ["Series", "Week"]
    ).reset_index(drop=True)

    expected = pd.date_range(
        df["Week"].min(),
        df["Week"].max(),
        freq="W-MON"
    )

    for series, group in df.groupby("Series"):

        dates = pd.DatetimeIndex(
            group["Week"]
        )

        if not dates.equals(expected):

            raise ValueError(
                f"{series}: incomplete weekly history."
            )

        if group["Sales"].gt(0).sum() == 0:

            raise ValueError(
                f"{series}: no positive sales history."
            )

    return df


# ============================================================
# 2. SELECT RELEVANT PRODUCT HISTORY
# ============================================================

def product_history(history, series):

    history = validate_history(history)

    selected = history[
        history["Series"] == series
    ]

    if selected.empty:

        raise ValueError(
            f"Unknown demand series: {series}"
        )

    product = selected[
        "Product_Group"
    ].iloc[0]

    return history[
        history["Product_Group"] == product
    ].copy()


# ============================================================
# 3. DEFAULT FUTURE PRICE
# ============================================================

def default_price(history, series):

    selected = history[
        history["Series"] == series
    ]

    train = advanced_engine.prepare_training(
        selected,
        selected["Week"].max()
    )

    observed = train["Price"].where(
        train["Price"] > 0
    )

    recent = observed.tail(13).dropna()

    if not recent.empty:

        return float(recent.median())

    available = observed.dropna()

    if available.empty:

        raise ValueError(
            f"No observed prices for {series}."
        )

    return float(available.median())


# ============================================================
# 4. GENERATE STANDARD MODEL FORECASTS
# ============================================================

def forecast_candidates(
    history,
    series,
    models=None,
    horizon=13
):

    if models is None:

        models = MODELS

    if not 1 <= horizon <= 52:

        raise ValueError(
            "Forecast horizon must be 1–52 weeks."
        )

    unknown = set(models) - set(MODELS)

    if unknown:

        raise ValueError(
            f"Unknown models: {unknown}"
        )

    group = product_history(
        history,
        series
    )

    origin = group["Week"].max()

    dates = pd.date_range(
        origin + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON"
    )

    # The existing pooled model was designed
    # specifically for North and South.

    if "Pooled log price regression" in models:

        if set(group["Region"]) != {
            "North", "South"
        }:

            raise ValueError(
                "Pooled forecasts require "
                "North and South observations."
            )

    forecasts, audit, prices = (
        evaluation_engine.generate_forecasts(
            group,
            dates,
            list(models)
        )
    )

    forecasts = forecasts[
        forecasts["Series"] == series
    ].copy()

    audit = audit[
        audit["Series"] == series
    ].copy()

    prices = prices[
        prices["Series"] == series
    ].copy()

    return forecasts, audit, prices


# ============================================================
# 5. FORECAST WITH CUSTOM FUTURE PRICE
# ============================================================

def price_scenario(
    history,
    series,
    model,
    horizon,
    assumed_price
):

    if model not in PRICE_MODELS:

        raise ValueError(
            "Selected model does not use price."
        )

    if not 1 <= horizon <= 52:

        raise ValueError(
            "Forecast horizon must be 1–52 weeks."
        )

    assumed_price = float(
        assumed_price
    )

    if (
        not np.isfinite(assumed_price)
        or assumed_price <= 0
    ):

        raise ValueError(
            "Future price must be positive and finite."
        )

    group = product_history(
        history,
        series
    )

    origin = group["Week"].max()

    dates = pd.date_range(
        origin + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON"
    )

    pooled = (
        model == "Pooled log price regression"
    )

    if pooled:

        if set(group["Region"]) != {
            "North", "South"
        }:

            raise ValueError(
                "Pooled model requires both regions."
            )

    else:

        group = group[
            group["Series"] == series
        ]

    train_parts = []
    future_parts = []

    for name, frame in group.groupby("Series"):

        train = advanced_engine.prepare_training(
            frame,
            origin
        ).reset_index(drop=True)

        future = pd.DataFrame({
            "Week": dates,
            "Series": name,
            "Region": train["Region"].iloc[0],
            "Product_Group":
                train["Product_Group"].iloc[0]
        })

        train, future, _, _ = (
            advanced_engine.prepare_prices(
                train,
                future
            )
        )

        # Change only the selected series.
        # The other region retains its default price.

        if name == series:

            future["Model_Price"] = (
                assumed_price
            )

        train_parts.append(train)
        future_parts.append(future)

    train = pd.concat(
        train_parts,
        ignore_index=True
    )

    future = pd.concat(
        future_parts,
        ignore_index=True
    )

    raw, method = (
        advanced_engine.advanced_forecast(
            model,
            train,
            future
        )
    )

    future["Model"] = model

    future["Forecast_Raw"] = raw

    future["Forecast"] = np.maximum(
        0,
        raw
    )

    future["Forecast_Units"] = np.rint(
        future["Forecast"]
    ).astype(int)

    future["Assumed_Price"] = (
        future["Model_Price"]
    )

    future["Forecast_Origin"] = origin

    future["Fit_Method"] = method

    future["Price_Scenario"] = (
        "User-defined future price"
    )

    result = future[
        future["Series"] == series
    ].copy()

    assert len(result) == horizon

    assert np.isfinite(
        result["Forecast"]
    ).all()

    return result


# ============================================================
# 6. GENERATE ONE SELECTED FORECAST
# ============================================================

def forecast_series(
    history,
    series,
    model,
    horizon=13,
    assumed_price=None
):

    if assumed_price is not None:

        return price_scenario(
            history,
            series,
            model,
            horizon,
            assumed_price
        )

    predictions, audit, _ = (
        forecast_candidates(
            history,
            series,
            models=[model],
            horizon=horizon
        )
    )

    if len(predictions) != horizon:

        raise RuntimeError(
            f"Incomplete forecast for {model}. "
            f"Fit audit: {audit.to_dict('records')}"
        )

    predictions["Forecast_Units"] = (
        np.rint(predictions["Forecast"])
        .astype(int)
    )

    predictions["Price_Scenario"] = (
        "Default historical median price"
        if model in PRICE_MODELS
        else "Model does not use price"
    )

    return predictions.sort_values(
        "Week"
    ).reset_index(drop=True)


# ============================================================
# 7. HISTORICAL TEST FOR A SELECTED MODEL
# ============================================================

def historical_test(
    history,
    series,
    model,
    test_weeks=13
):

    history = validate_history(history)

    weeks = pd.DatetimeIndex(
        sorted(history["Week"].unique())
    )

    if len(weeks) <= test_weeks + 13:

        raise ValueError(
            "Insufficient history for testing."
        )

    test_dates = weeks[-test_weeks:]

    train = history[
        history["Week"] < test_dates.min()
    ]

    test = history[
        history["Week"].isin(test_dates)
        & history["Series"].eq(series)
    ]

    forecast = forecast_series(
        train,
        series,
        model,
        horizon=test_weeks
    )

    actual = test[
        ["Series", "Week", "Sales"]
    ].rename(
        columns={"Sales": "Actual"}
    )

    result = forecast.merge(
        actual,
        on=["Series", "Week"],
        validate="one_to_one"
    )

    assert len(result) == test_weeks

    result["Error"] = (
        result["Forecast"]
        - result["Actual"]
    )

    result["Cumulative_Error"] = (
        result["Error"].cumsum()
    )

    metrics = advanced_engine.metrics(
        result
    )

    return result, metrics


# ============================================================
# 8. DYNAMIC MODEL RECOMMENDATION
# ============================================================

def recommend_model(history, series):

    history = validate_history(history)

    weeks = pd.DatetimeIndex(
        sorted(history["Week"].unique())
    )

    # Reserve the final 13 weeks.
    # Use three development validation origins.

    development_end = len(weeks) - 13

    origins = [
        development_end - 19,
        development_end - 16,
        development_end - 13
    ]

    if min(origins) < 26:

        raise ValueError(
            "Insufficient history for automatic "
            "13-week model selection."
        )

    candidates = MODELS.copy()

    group = product_history(
        history,
        series
    )

    if set(group["Region"]) != {
        "North", "South"
    }:

        candidates.remove(
            "Pooled log price regression"
        )

    records = []

    for fold, origin in enumerate(
        origins,
        start=1
    ):

        train_dates = weeks[:origin]

        validation_dates = weeks[
            origin:origin + 13
        ]

        train = history[
            history["Week"].isin(train_dates)
        ]

        actual = history[
            history["Week"].isin(
                validation_dates
            )
            & history["Series"].eq(series)
        ][
            ["Series", "Week", "Sales"]
        ].rename(
            columns={"Sales": "Actual"}
        )

        pred, _, _ = forecast_candidates(
            train,
            series,
            models=candidates,
            horizon=13
        )

        result = pred.merge(
            actual,
            on=["Series", "Week"],
            validate="many_to_one"
        )

        result["Fold"] = fold

        records.append(result)

    predictions = pd.concat(
        records,
        ignore_index=True
    )

    ranking = []

    for name, group in predictions.groupby(
        "Model"
    ):

        # Only compare candidates with complete
        # predictions across all three folds.

        if len(group) != 39:

            continue

        if group.duplicated(
            ["Fold", "Week"]
        ).any():

            continue

        metrics = advanced_engine.metrics(
            group
        )

        ranking.append({
            "Model": name,
            **metrics
        })

    ranking = pd.DataFrame(ranking)

    if ranking.empty:

        raise RuntimeError(
            "No complete forecasting candidates."
        )

    ranking = ranking.sort_values(
        ["MAPE_Pct", "Model"]
    ).reset_index(drop=True)

    return ranking


# ============================================================
# 9. SMOKE TEST USING OUR VALIDATED PROJECT DATA
# ============================================================

def main():

    print("=" * 70)
    print("LIVE FORECASTING ENGINE - VALIDATION")
    print("=" * 70)

    history = pd.read_csv(
        OUT / "historical_data_clean.csv",
        parse_dates=["Week"]
    )

    history = validate_history(history)

    selection = pd.read_csv(
        OUT / "final_model_selection.csv"
    )

    expected = pd.read_csv(
        OUT / "final_forecast_13_weeks_detailed.csv",
        parse_dates=["Week"]
    )

    expected_test = pd.read_csv(
        OUT / "forecast_final_test_selected_metrics.csv"
    )

    print("\n1. VALIDATE HISTORICAL INPUT")

    print("Historical observations:", len(history))

    print(
        "Demand series:",
        history["Series"].nunique()
    )

    print("\n2. REPRODUCE SUBMISSION FORECASTS")

    for _, row in selection.iterrows():

        series = row["Series"]
        model = row["Selected_Model"]

        result = forecast_series(
            history,
            series,
            model,
            horizon=13
        )

        reference = expected[
            expected["Series"] == series
        ][
            ["Week", "Forecast_Units"]
        ]

        check = result.merge(
            reference,
            on="Week",
            suffixes=("_Live", "_Expected"),
            validate="one_to_one"
        )

        assert len(check) == 13

        assert np.array_equal(
            check["Forecast_Units_Live"],
            check["Forecast_Units_Expected"]
        ), f"Forecast mismatch for {series}"

        print(
            f"PASS - {series}: "
            f"{result['Forecast_Units'].sum():,} units"
        )

    print("\n3. REPRODUCE HISTORICAL TEST ACCURACY")

    for _, row in selection.iterrows():

        series = row["Series"]
        model = row["Selected_Model"]

        _, metrics = historical_test(
            history,
            series,
            model
        )

        expected_mape = expected_test.loc[
            expected_test["Series"] == series,
            "MAPE_Pct"
        ].iloc[0]

        assert np.isclose(
            metrics["MAPE_Pct"],
            expected_mape,
            atol=0.001
        )

        print(
            f"PASS - {series}: "
            f"MAPE {metrics['MAPE_Pct']:.2f}%"
        )

    print("\n4. TEST INTERACTIVE PRICE SCENARIOS")

    series = "North 2"
    model = "Log price Ridge regression"

    price = default_price(
        history,
        series
    )

    base = forecast_series(
        history,
        series,
        model,
        horizon=13
    )

    same = forecast_series(
        history,
        series,
        model,
        horizon=13,
        assumed_price=price
    )

    assert np.allclose(
        base["Forecast"],
        same["Forecast"]
    )

    lower = forecast_series(
        history,
        series,
        model,
        horizon=13,
        assumed_price=price * 0.90
    )

    higher = forecast_series(
        history,
        series,
        model,
        horizon=13,
        assumed_price=price * 1.10
    )

    scenarios = pd.DataFrame({
        "Scenario": [
            "Default",
            "10% lower price",
            "10% higher price"
        ],

        "Price_EUR": [
            price,
            price * 0.90,
            price * 1.10
        ],

        "Forecast_Units": [
            base["Forecast_Units"].sum(),
            lower["Forecast_Units"].sum(),
            higher["Forecast_Units"].sum()
        ]
    })

    print(
        scenarios.round(2).to_string(index=False)
    )

    print(
        "PASS - Default price scenario reproduced."
    )

    print("\n5. TEST 52-WEEK FORECAST")

    extended = forecast_series(
        history,
        series,
        model,
        horizon=52
    )

    assert len(extended) == 52

    assert np.allclose(
        extended["Forecast"].iloc[:13],
        base["Forecast"]
    )

    print(
        "PASS - 52-week forecast generated."
    )

    print(
        "PASS - First 13 weeks match "
        "the short-term forecast."
    )

    print("\n6. TEST AUTOMATIC MODEL RECOMMENDATION")

    ranking = recommend_model(
        history,
        series
    )

    print(
        ranking[
            [
                "Model",
                "MAPE_Pct",
                "WAPE_Pct",
                "Bias_Pct"
            ]
        ].round(2).to_string(index=False)
    )

    expected_model = selection.loc[
        selection["Series"] == series,
        "Selected_Model"
    ].iloc[0]

    assert (
        ranking.iloc[0]["Model"]
        == expected_model
    )

    print(
        "PASS - Automatic model recommendation "
        "reproduces our development selection."
    )

    print("\n" + "=" * 70)

    print(
        "Live forecasting engine "
        "validated successfully!"
    )

    print(
        "Ready for Streamlit integration."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
