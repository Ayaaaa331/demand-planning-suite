
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression

# Reuse the validated forecasting functions
# from script 11.

engine = import_module("11_advanced_forecasting")

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CHART = OUT / "charts"

CHART.mkdir(parents=True, exist_ok=True)

FINAL_TEST_WEEKS = 13

PRICE_MODELS = [
    "Price Ridge regression",
    "Log price Ridge regression",
    "Pooled log price regression",
    "ARIMAX (1,0,0)"
]

# ============================================================
# 2. REUSABLE FORECASTING ENGINE
# ============================================================

def generate_forecasts(
    history,
    forecast_dates,
    model_names
):

    history = history.copy()

    forecast_dates = pd.DatetimeIndex(
        forecast_dates
    )

    forecast_origin = history["Week"].max()

    assert forecast_origin < forecast_dates.min()

    assert not history.duplicated(
        ["Series", "Week"]
    ).any()

    assert history["Sales"].notna().all()

    results = []
    fit_records = []
    price_records = []

    training_parts = []
    future_parts = []

    # --------------------------------------------------------
    # Helper function to record forecasts.
    # --------------------------------------------------------

    def record(future, name, raw, method):

        raw = np.asarray(
            raw,
            dtype=float
        )

        assert len(raw) == len(future)
        assert np.isfinite(raw).all()

        part = future[
            [
                "Week",
                "Series",
                "Region",
                "Product_Group"
            ]
        ].copy()

        part["Forecast_Origin"] = forecast_origin

        part["Horizon_Week"] = (
            (part["Week"] - forecast_origin)
            .dt.days // 7
        )

        part["Model"] = name

        part["Forecast_Raw"] = raw

        part["Forecast"] = np.maximum(
            0,
            raw
        )

        part["Fit_Method"] = method

        part["Assumed_Price"] = (
            future["Model_Price"].to_numpy()
            if name in PRICE_MODELS
            else np.nan
        )

        results.append(part)

    # --------------------------------------------------------
    # Independent forecasts for each demand series.
    # --------------------------------------------------------

    for series, group in history.groupby("Series"):

        train = engine.prepare_training(
            group,
            forecast_origin
        )

        train = train.reset_index(drop=True)

        future = pd.DataFrame({
            "Week": forecast_dates,
            "Series": series,
            "Region": train["Region"].iloc[0],
            "Product_Group":
                train["Product_Group"].iloc[0]
        })

        train, future, price, source = (
            engine.prepare_prices(
                train,
                future
            )
        )

        training_parts.append(train)
        future_parts.append(future)

        price_records.append({
            "Series": series,
            "Forecast_Origin": forecast_origin,
            "Assumed_Price": price,
            "Price_Assumption": source,
            "Training_Observations": len(train)
        })

        y = train["Sales"].to_numpy(
            dtype=float
        )

        h = len(future)

        print(
            f"Forecasting {series}: "
            f"{len(train)} active training weeks",
            flush=True
        )

        for name in model_names:

            if name == "Pooled log price regression":
                continue

            try:

                if name == "Naive":

                    raw = np.repeat(y[-1], h)
                    method = "Last observation"

                elif name == "Seasonal naive":

                    lagged_dates = (
                        forecast_dates
                        - pd.Timedelta(weeks=52)
                    )

                    lagged = train.set_index(
                        "Week"
                    )["Sales"].reindex(
                        lagged_dates
                    )

                    if lagged.isna().any():

                        raise ValueError(
                            "Insufficient seasonal history."
                        )

                    raw = lagged.to_numpy()
                    method = "52-week lag"

                elif name == "Moving average 4":

                    raw = np.repeat(
                        np.mean(y[-4:]),
                        h
                    )

                    method = "4-week mean"

                elif name == "Moving average 13":

                    raw = np.repeat(
                        np.mean(y[-13:]),
                        h
                    )

                    method = "13-week mean"

                elif name in engine.SMOOTHING_MODELS:

                    raw, method = engine.fit_smoothing(
                        name,
                        y,
                        h
                    )

                elif name == "Linear regression":

                    origin = train["Week"].min()

                    X_train = (
                        (train["Week"] - origin)
                        .dt.days.to_numpy() / 7
                    ).reshape(-1, 1)

                    X_future = (
                        (future["Week"] - origin)
                        .dt.days.to_numpy() / 7
                    ).reshape(-1, 1)

                    fitted = LinearRegression()

                    fitted.fit(X_train, y)

                    raw = fitted.predict(
                        X_future
                    )

                    method = "OLS"

                elif name == (
                    "Seasonal calendar regression"
                ):

                    raw = engine.ridge_forecast(
                        train,
                        future,
                        use_price=False,
                        log_target=False
                    )

                    method = "Ridge"

                elif name in engine.ADVANCED_MODELS:

                    raw, method = (
                        engine.advanced_forecast(
                            name,
                            train,
                            future
                        )
                    )

                else:

                    raise ValueError(
                        f"Unknown model: {name}"
                    )

                record(
                    future,
                    name,
                    raw,
                    method
                )

                fit_records.append({
                    "Series": series,
                    "Model": name,
                    "Success": True,
                    "Message": method
                })

            except Exception as error:

                fit_records.append({
                    "Series": series,
                    "Model": name,
                    "Success": False,
                    "Message": str(error)
                })

                print(
                    f"FAILED - {series} | "
                    f"{name}: {error}"
                )

    # --------------------------------------------------------
    # Pooled regional model.
    # --------------------------------------------------------

    if "Pooled log price regression" in model_names:

        pooled_train = pd.concat(
            training_parts,
            ignore_index=True
        )

        pooled_future = pd.concat(
            future_parts,
            ignore_index=True
        )

        name = "Pooled log price regression"

        for product, future in pooled_future.groupby(
            "Product_Group"
        ):

            train = pooled_train[
                pooled_train["Product_Group"]
                == product
            ].copy()

            try:

                raw, method = (
                    engine.advanced_forecast(
                        name,
                        train,
                        future
                    )
                )

                record(
                    future,
                    name,
                    raw,
                    method
                )

                for series in future["Series"].unique():

                    fit_records.append({
                        "Series": series,
                        "Model": name,
                        "Success": True,
                        "Message": method
                    })

            except Exception as error:

                for series in future["Series"].unique():

                    fit_records.append({
                        "Series": series,
                        "Model": name,
                        "Success": False,
                        "Message": str(error)
                    })

    predictions = pd.concat(
        results,
        ignore_index=True
    )

    fit_audit = pd.DataFrame(
        fit_records
    )

    prices = pd.DataFrame(
        price_records
    )

    assert not predictions.duplicated(
        ["Series", "Week", "Model"]
    ).any()

    assert predictions["Forecast"].ge(0).all()

    return predictions, fit_audit, prices


# ============================================================
# 3. EVALUATION FUNCTIONS
# ============================================================

def add_errors(frame):

    frame = frame.copy()

    frame["Error"] = (
        frame["Forecast"]
        - frame["Actual"]
    )

    frame["Absolute_Error"] = (
        frame["Error"].abs()
    )

    frame["Squared_Error"] = (
        frame["Error"] ** 2
    )

    frame["Absolute_Percentage_Error"] = (
        frame["Absolute_Error"]
        / frame["Actual"].replace(0, np.nan)
        * 100
    )

    return frame


def summarize(frame, grouping):

    records = []

    for keys, group in frame.groupby(
        grouping
    ):

        if not isinstance(keys, tuple):

            keys = (keys,)

        row = dict(
            zip(grouping, keys)
        )

        row.update(
            engine.metrics(group)
        )

        row["Actual_Total"] = (
            group["Actual"].sum()
        )

        row["Forecast_Total"] = (
            group["Forecast"].sum()
        )

        records.append(row)

    return pd.DataFrame(records)


# ============================================================
# 4. MAIN WORKFLOW
# ============================================================

def main():

    print("=" * 70)
    print("DEMAND PLANNING - FINAL HISTORICAL TEST")
    print("=" * 70)

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

    development_metrics = pd.read_csv(
        OUT / "forecast_combined_series_metrics.csv"
    )

    winners = pd.read_csv(
        OUT / "forecast_combined_winners.csv"
    )

    # ========================================================
    # 5. FREEZE MODEL SELECTION BEFORE TESTING
    # ========================================================

    selection = winners[
        [
            "Series",
            "Best_MAPE_Model",
            "MAPE_Pct",
            "WAPE_Pct",
            "Bias_Pct",
            "Best_WAPE_Model",
            "Best_WAPE_Pct"
        ]
    ].copy()

    selection = selection.rename(
        columns={
            "Best_MAPE_Model":
                "Selected_Model",

            "MAPE_Pct":
                "Development_MAPE",

            "WAPE_Pct":
                "Development_WAPE",

            "Bias_Pct":
                "Development_Bias_Pct"
        }
    )

    selection["Selection_Rule"] = (
        "Lowest development MAPE"
    )

    selection["Planning_Warning"] = np.where(
        selection["Development_Bias_Pct"] < -50,
        "Severe development underforecasting",
        np.where(
            selection["Development_Bias_Pct"] < -25,
            "Material development underforecasting",
            ""
        )
    )

    assert selection["Series"].is_unique

    assert set(selection["Series"]) == set(
        history["Series"].unique()
    )

    selection.to_csv(
        OUT / "final_model_selection.csv",
        index=False
    )

    print("\n1. FROZEN MODEL SELECTION")

    print(
        selection[
            [
                "Series",
                "Selected_Model",
                "Development_MAPE",
                "Development_WAPE",
                "Development_Bias_Pct"
            ]
        ].round(2).to_string(index=False)
    )

    # ========================================================
    # 6. PREPARE RESERVED HISTORICAL TEST
    # ========================================================

    weeks = pd.DatetimeIndex(
        sorted(history["Week"].unique())
    )

    test_dates = weeks[-FINAL_TEST_WEEKS:]

    train = history[
        history["Week"] < test_dates.min()
    ].copy()

    test = history[
        history["Week"].isin(test_dates)
    ].copy()

    forecast_origin = train["Week"].max()

    assert forecast_origin < test_dates.min()

    assert train["Week"].nunique() == (
        len(weeks) - FINAL_TEST_WEEKS
    )

    assert test["Sales"].notna().all()

    print("\n2. HISTORICAL TEST DESIGN")

    print(
        "Training weeks:",
        train["Week"].nunique()
    )

    print(
        "Forecast origin:",
        forecast_origin.date()
    )

    print(
        "Test period:",
        test_dates.min().date(),
        "to",
        test_dates.max().date()
    )

    print(
        "Test observations:",
        len(test)
    )

    model_names = sorted(
        development_metrics["Model"].unique()
    )

    # ========================================================
    # 7. GENERATE TEST FORECASTS
    # ========================================================

    print("\n3. TEST FORECAST ESTIMATION")

    predictions, fit_audit, prices = (
        generate_forecasts(
            train,
            test_dates,
            model_names
        )
    )

    # Attach test actuals only after forecasts
    # have been produced.

    actuals = test[
        ["Series", "Week", "Sales"]
    ].rename(
        columns={"Sales": "Actual"}
    )

    predictions = predictions.merge(
        actuals,
        on=["Series", "Week"],
        how="left",
        validate="many_to_one"
    )

    assert predictions["Actual"].notna().all()

    predictions = add_errors(
        predictions
    )

    # ========================================================
    # 8. CHECK MODEL COMPLETENESS
    # ========================================================

    expected_keys = set(
        actuals[
            ["Series", "Week"]
        ].itertuples(
            index=False,
            name=None
        )
    )

    complete_models = []

    for name, group in predictions.groupby("Model"):

        keys = set(
            group[
                ["Series", "Week"]
            ].itertuples(
                index=False,
                name=None
            )
        )

        if (
            keys == expected_keys
            and len(group) == len(actuals)
        ):

            complete_models.append(name)

    complete = predictions[
        predictions["Model"].isin(
            complete_models
        )
    ].copy()

    print("\n4. MODEL FIT AUDIT")

    print(
        fit_audit.groupby("Model")["Success"]
        .agg(["count", "sum"])
        .to_string()
    )

    failures = fit_audit[
        ~fit_audit["Success"]
    ]

    fit_audit.to_csv(
        OUT / "forecast_final_test_fit_audit.csv",
        index=False
    )

    if not failures.empty:

        print("\nFit failures:")

        print(
            failures.to_string(index=False)
        )

    print(
        "\nComplete models:",
        len(complete_models)
    )

    # ========================================================
    # 9. EVALUATE ALL COMPLETE CANDIDATES
    # ========================================================

    overall = summarize(
        complete,
        ["Model"]
    ).sort_values(
        "MAPE_Pct"
    )

    series_metrics = summarize(
        complete,
        ["Series", "Model"]
    )

    print("\n5. HISTORICAL TEST - ALL MODELS")

    print(
        overall[
            [
                "Model",
                "Observations",
                "MAPE_Pct",
                "WAPE_Pct",
                "MAD",
                "MSE",
                "Bias_Pct"
            ]
        ].round(2).to_string(index=False)
    )

    # These results are diagnostic only.
    # Do not select new models from this table.

    # ========================================================
    # 10. EXTRACT PRESELECTED FORECASTING STRATEGIES
    # ========================================================

    joined = predictions.merge(
        selection[
            [
                "Series",
                "Selected_Model",
                "Best_WAPE_Model"
            ]
        ],
        on="Series",
        validate="many_to_one"
    )

    selected = joined[
        joined["Model"]
        == joined["Selected_Model"]
    ].copy()

    selected["Strategy"] = (
        "Development MAPE selection"
    )

    wape_selected = joined[
        joined["Model"]
        == joined["Best_WAPE_Model"]
    ].copy()

    wape_selected["Strategy"] = (
        "Development WAPE selection"
    )

    naive = joined[
        joined["Model"] == "Naive"
    ].copy()

    naive["Strategy"] = (
        "Naive benchmark"
    )

    strategies = pd.concat(
        [
            selected,
            wape_selected,
            naive
        ],
        ignore_index=True
    )

    # No silent fallback if a selected
    # model fails to generate forecasts.

    for name, group in strategies.groupby(
        "Strategy"
    ):

        assert len(group) == len(actuals), (
            f"Incomplete selected strategy: {name}"
        )

        assert not group.duplicated(
            ["Series", "Week"]
        ).any()

    assert strategies["Strategy"].nunique() == 3

    # ========================================================
    # 11. COMPARE FORECASTING STRATEGIES
    # ========================================================

    strategy_metrics = summarize(
        strategies,
        ["Strategy"]
    )

    strategy_series_metrics = summarize(
        strategies,
        ["Strategy", "Series"]
    )

    print("\n6. FINAL FORECASTING STRATEGY COMPARISON")

    print(
        strategy_metrics[
            [
                "Strategy",
                "Observations",
                "MAPE_Pct",
                "WAPE_Pct",
                "MAD",
                "MSE",
                "Bias_Pct",
                "Actual_Total",
                "Forecast_Total"
            ]
        ].round(2).to_string(index=False)
    )

    selected_metrics = summarize(
        selected,
        ["Series", "Model"]
    )

    selected_metrics = selected_metrics.merge(
        selection[
            [
                "Series",
                "Development_MAPE",
                "Development_WAPE",
                "Development_Bias_Pct"
            ]
        ],
        on="Series",
        validate="one_to_one"
    )

    selected_metrics["MAPE_Change_pp"] = (
        selected_metrics["MAPE_Pct"]
        - selected_metrics["Development_MAPE"]
    )

    print("\n7. SELECTED MODELS - FINAL TEST RESULTS")

    print(
        selected_metrics[
            [
                "Series",
                "Model",
                "Development_MAPE",
                "MAPE_Pct",
                "WAPE_Pct",
                "MAD",
                "Bias_Pct",
                "Actual_Total",
                "Forecast_Total"
            ]
        ].round(2).to_string(index=False)
    )

    # ========================================================
    # 12. COMPARE AGAINST NAIVE BENCHMARK
    # ========================================================

    naive_metrics = summarize(
        naive,
        ["Series"]
    )

    comparison = selected_metrics.merge(
        naive_metrics[
            [
                "Series",
                "MAPE_Pct",
                "WAPE_Pct"
            ]
        ],
        on="Series",
        suffixes=("", "_Naive"),
        validate="one_to_one"
    )

    comparison["MAPE_Improvement_pp"] = (
        comparison["MAPE_Pct_Naive"]
        - comparison["MAPE_Pct"]
    )

    print("\n8. SELECTED MODELS VS NAIVE BENCHMARK")

    print(
        comparison[
            [
                "Series",
                "Model",
                "MAPE_Pct",
                "MAPE_Pct_Naive",
                "MAPE_Improvement_pp",
                "WAPE_Pct",
                "WAPE_Pct_Naive"
            ]
        ].round(2).to_string(index=False)
    )

    # ========================================================
    # 13. CUMULATIVE FORECAST ERROR
    # ========================================================

    selected = selected.sort_values(
        ["Series", "Week"]
    )

    selected["Cumulative_Error"] = (
        selected.groupby("Series")["Error"]
        .cumsum()
    )

    # ========================================================
    # 14. VISUALIZATION - ACTUAL VS FORECAST
    # ========================================================

    chart_data = selected.melt(
        id_vars=["Week", "Series"],
        value_vars=["Actual", "Forecast"],
        var_name="Measure",
        value_name="Units"
    )

    fig1 = px.line(
        chart_data.sort_values("Week"),
        x="Week",
        y="Units",
        color="Measure",
        facet_col="Series",
        facet_col_wrap=2,
        markers=True,
        title="Reserved Historical Test - Actual vs Forecast",
        color_discrete_map={
            "Actual": "#365C8D",
            "Forecast": "#46A094"
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
        CHART / "24_final_test_actual_vs_forecast.html"
    )

    # ========================================================
    # 15. VISUALIZATION - MAPE COMPARISON
    # ========================================================

    fig2 = px.bar(
        strategy_series_metrics,
        x="Series",
        y="MAPE_Pct",
        color="Strategy",
        barmode="group",
        title="Reserved Test - Forecasting Strategy Accuracy",
        labels={
            "MAPE_Pct": "MAPE (%)"
        },
        template="plotly_white"
    )

    fig2.write_html(
        CHART / "25_final_test_accuracy.html"
    )

    # ========================================================
    # 16. VISUALIZATION - BIAS
    # ========================================================

    fig3 = px.bar(
        selected_metrics,
        x="Series",
        y="Bias_Pct",
        title="Reserved Historical Test - Forecast Bias",
        labels={
            "Bias_Pct": "Aggregate forecast bias (%)"
        },
        template="plotly_white"
    )

    fig3.add_hline(y=0)

    fig3.write_html(
        CHART / "26_final_test_bias.html"
    )

    # ========================================================
    # 17. VISUALIZATION - CUMULATIVE ERROR
    # ========================================================

    fig4 = px.line(
        selected,
        x="Week",
        y="Cumulative_Error",
        color="Series",
        title="Reserved Test - Cumulative Forecast Error",
        labels={
            "Cumulative_Error": "Cumulative error (units)"
        },
        template="plotly_white"
    )

    fig4.add_hline(y=0)

    fig4.write_html(
        CHART / "27_final_test_cumulative_error.html"
    )

    # ========================================================
    # 18. VALIDATION
    # ========================================================

    expected_observations = (
        history["Series"].nunique()
        * FINAL_TEST_WEEKS
    )

    assert len(selected) == expected_observations

    assert len(naive) == expected_observations

    assert not predictions.duplicated(
        ["Series", "Week", "Model"]
    ).any()

    assert (
        predictions["Forecast_Origin"]
        < predictions["Week"]
    ).all()

    assert predictions["Forecast"].ge(0).all()

    assert np.isfinite(
        predictions["Forecast"]
    ).all()

    assert selected["Model"].eq(
        selected["Selected_Model"]
    ).all()

    assert selected["Week"].nunique() == (
        FINAL_TEST_WEEKS
    )

    assert prices["Assumed_Price"].gt(0).all()

    assert np.isclose(
        selected["Actual"].sum(),
        actuals["Actual"].sum()
    )

    # Confirm development validation ended
    # before the final historical test began.

    development_predictions = pd.read_csv(
        OUT / "forecast_combined_predictions.csv",
        parse_dates=["Week"]
    )

    assert development_predictions["Week"].max() < (
        test_dates.min()
    )

    print("\n9. VALIDATION")

    print("PASS - Model selection frozen before testing.")
    print("PASS - Chronological separation preserved.")
    print("PASS - All selected forecasts available.")
    print("PASS - Identical test observations.")
    print("PASS - No duplicate forecasts.")
    print("PASS - Forecasts finite and nonnegative.")
    print("PASS - Price assumptions valid.")
    print("PASS - Actual sales totals reconciled.")
    print("PASS - Development and final test separated.")

    # ========================================================
    # 19. EXPORT RESULTS
    # ========================================================

    predictions.to_csv(
        OUT / "forecast_final_test_all_predictions.csv",
        index=False
    )

    overall.to_csv(
        OUT / "forecast_final_test_all_model_metrics.csv",
        index=False
    )

    series_metrics.to_csv(
        OUT / "forecast_final_test_series_metrics.csv",
        index=False
    )

    selected.to_csv(
        OUT / "forecast_final_test_selected_predictions.csv",
        index=False
    )

    selected_metrics.to_csv(
        OUT / "forecast_final_test_selected_metrics.csv",
        index=False
    )

    strategy_metrics.to_csv(
        OUT / "forecast_final_test_strategy_comparison.csv",
        index=False
    )

    strategy_series_metrics.to_csv(
        OUT / "forecast_final_test_strategy_series_metrics.csv",
        index=False
    )

    comparison.to_csv(
        OUT / "forecast_final_test_naive_comparison.csv",
        index=False
    )

    prices.to_csv(
        OUT / "forecast_final_test_price_assumptions.csv",
        index=False
    )

    print("\n" + "=" * 70)

    print(
        "Final historical test completed successfully!"
    )

    print(
        "Selected forecasts:",
        len(selected)
    )

    print(
        "All model forecasts:",
        len(predictions)
    )

    print("Results saved to:", OUT)
    print("Charts saved to:", CHART)

    print(
        "\nModel selection remains based on "
        "development validation only."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
