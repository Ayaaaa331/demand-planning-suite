
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px

from dateutil.easter import easter

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt
)

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CHART = OUT / "charts"

CHART.mkdir(parents=True, exist_ok=True)

ALPHA = 5.0
PRICE_WINDOW = 13
HORIZON = 13

SMOOTHING_MODELS = [
    "SES",
    "Holt linear",
    "Holt damped"
]

ADVANCED_MODELS = [
    "Moving median 13",
    "Log calendar regression",
    "Price Ridge regression",
    "Log price Ridge regression",
    "Pooled log price regression",
    "ARIMAX (1,0,0)"
]

# ============================================================
# 2. CALENDAR FEATURES
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

    black_friday = monday(
        thanksgiving
    )

    cyber_monday = (
        thanksgiving + pd.Timedelta(days=4)
    )

    christmas = monday(
        pd.Timestamp(year, 12, 25)
    )

    easter_monday = (
        pd.Timestamp(easter(year))
        + pd.Timedelta(days=1)
    )

    return [
        int(
            week in [
                black_friday,
                cyber_monday
            ]
        ),
        int(
            week in [
                christmas - pd.Timedelta(weeks=1),
                christmas
            ]
        ),
        int(
            week in [
                easter_monday - pd.Timedelta(weeks=1),
                easter_monday
            ]
        )
    ]


def build_features(
    frame,
    origin,
    use_price=False,
    pooled=False
):

    dates = pd.DatetimeIndex(
        frame["Week"]
    )

    t = np.asarray(
        (dates - origin).days / 7,
        dtype=float
    )

    angle = 2 * np.pi * t / 52.1775

    events = np.array([
        event_flags(w)
        for w in dates
    ])

    features = [
        t,
        np.sin(angle),
        np.cos(angle),
        events
    ]

    if use_price:

        features.append(
            np.log(
                frame["Model_Price"]
                .to_numpy(dtype=float)
            )
        )

    if pooled:

        features.append(
            frame["Region"].eq("South")
            .astype(float)
            .to_numpy()
        )

    return np.column_stack(
        features
    ).astype(float)


# ============================================================
# 3. PREPARE ACTIVE TRAINING HISTORY
# ============================================================

def prepare_training(group, forecast_origin):

    train = group[
        group["Week"] <= forecast_origin
    ].copy()

    train = train.sort_values(
        "Week"
    )

    positive_sales = train.loc[
        train["Sales"] > 0,
        "Week"
    ]

    if positive_sales.empty:

        raise ValueError(
            "No positive training sales."
        )

    first_sale = positive_sales.min()

    train = train[
        train["Week"] >= first_sale
    ].copy()

    assert train["Sales"].notna().all()

    assert train["Week"].max() <= forecast_origin

    return train


# ============================================================
# 4. PREPARE PRICE ASSUMPTIONS
# ============================================================

def prepare_prices(train, future):

    train = train.copy()
    future = future.copy()

    observed = train["Price"].where(
        train["Price"] > 0
    )

    available = observed.dropna()

    if available.empty:

        raise ValueError(
            "No historical prices available."
        )

    # Estimate future prices using the most
    # recent 13 calendar weeks in training.

    recent = observed.tail(
        PRICE_WINDOW
    ).dropna()

    if recent.empty:

        assumed_price = available.median()

        assumption_source = (
            "All available training prices"
        )

    else:

        assumed_price = recent.median()

        assumption_source = (
            "Median of preceding 13 weeks"
        )

    # Missing historical prices are filled
    # using preceding observations.
    #
    # Any leading missing values use the
    # median of the training sample only.

    train["Model_Price"] = (
        observed.ffill()
        .fillna(available.median())
    )

    future["Model_Price"] = assumed_price

    assert train["Model_Price"].gt(0).all()
    assert future["Model_Price"].gt(0).all()

    return (
        train,
        future,
        float(assumed_price),
        assumption_source
    )


# ============================================================
# 5. AUDIT EXPONENTIAL SMOOTHING
# ============================================================

def fit_smoothing(model_name, y, horizon):

    y = np.asarray(
        y,
        dtype=float
    )

    # Scaling improves numerical conditioning.

    scale = max(
        float(np.mean(y)),
        1.0
    )

    scaled = y / scale

    optimizers = [
        "L-BFGS-B",
        "SLSQP",
        "least_squares"
    ]

    failures = []

    for optimizer in optimizers:

        try:

            with warnings.catch_warnings(
                record=True
            ) as caught:

                warnings.simplefilter(
                    "always",
                    ConvergenceWarning
                )

                if model_name == "SES":

                    fitted = SimpleExpSmoothing(
                        scaled,
                        initialization_method="estimated"
                    ).fit(
                        optimized=True,
                        method=optimizer
                    )

                else:

                    fitted = Holt(
                        scaled,
                        damped_trend=(
                            model_name == "Holt damped"
                        ),
                        initialization_method="estimated"
                    ).fit(
                        optimized=True,
                        method=optimizer
                    )

            convergence_warning = any(
                issubclass(
                    w.category,
                    ConvergenceWarning
                )
                for w in caught
            )

            optimizer_result = (
                fitted.mle_retvals
            )

            converged = (
                optimizer_result is not None
                and bool(
                    optimizer_result.get(
                        "success",
                        False
                    )
                )
            )

            prediction = (
                np.asarray(
                    fitted.forecast(horizon),
                    dtype=float
                ) * scale
            )

            if (
                converged
                and not convergence_warning
                and np.isfinite(prediction).all()
            ):

                return (
                    prediction,
                    optimizer
                )

            failures.append(
                f"{optimizer}: did not converge"
            )

        except Exception as error:

            failures.append(
                f"{optimizer}: {error}"
            )

    raise RuntimeError(
        "; ".join(failures)
    )


# ============================================================
# 6. RIDGE REGRESSION
# ============================================================

def ridge_forecast(
    train,
    future,
    use_price=False,
    log_target=False,
    pooled=False
):

    origin = train["Week"].min()

    X_train = build_features(
        train,
        origin,
        use_price,
        pooled
    )

    X_future = build_features(
        future,
        origin,
        use_price,
        pooled
    )

    y = train["Sales"].to_numpy(
        dtype=float
    )

    target = (
        np.log1p(y)
        if log_target
        else y
    )

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=ALPHA)
    )

    model.fit(
        X_train,
        target
    )

    raw_prediction = model.predict(
        X_future
    )

    if log_target:

        # Correct the log-to-unit transformation
        # using training residuals only.

        residual = (
            target
            - model.predict(X_train)
        )

        smearing = pd.Series(
            np.exp(residual),
            index=train.index
        )

        region_factors = smearing.groupby(
            train["Region"]
        ).mean()

        factors = future["Region"].map(
            region_factors
        ).to_numpy(dtype=float)

        raw_prediction = (
            np.exp(raw_prediction)
            * factors - 1
        )

    return np.asarray(
        raw_prediction,
        dtype=float
    )


# ============================================================
# 7. ARIMAX MODEL
# ============================================================

def arimax_forecast(train, future):

    origin = train["Week"].min()

    X_train = build_features(
        train,
        origin,
        use_price=True
    )

    X_future = build_features(
        future,
        origin,
        use_price=True
    )

    # Remove constant training features.
    #
    # Example: a holiday indicator that has
    # never occurred in the training period.

    nonconstant = (
        np.std(X_train, axis=0) > 1e-10
    )

    X_train = X_train[:, nonconstant]
    X_future = X_future[:, nonconstant]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_future = scaler.transform(
        X_future
    )

    y = train["Sales"].to_numpy(
        dtype=float
    )

    scale = max(
        float(y.mean()),
        1.0
    )

    optimizers = [
        "lbfgs",
        "powell"
    ]

    failures = []

    for optimizer in optimizers:

        try:

            with warnings.catch_warnings(
                record=True
            ) as caught:

                warnings.simplefilter(
                    "always",
                    ConvergenceWarning
                )

                model = SARIMAX(
                    y / scale,
                    exog=X_train,
                    order=(1, 0, 0),
                    trend="c",
                    enforce_stationarity=True,
                    enforce_invertibility=True
                )

                fitted = model.fit(
                    method=optimizer,
                    maxiter=300,
                    disp=False
                )

            converged = bool(
                fitted.mle_retvals.get(
                    "converged",
                    False
                )
            )

            convergence_warning = any(
                issubclass(
                    w.category,
                    ConvergenceWarning
                )
                for w in caught
            )

            prediction = (
                np.asarray(
                    fitted.forecast(
                        steps=len(future),
                        exog=X_future
                    ),
                    dtype=float
                ) * scale
            )

            if (
                converged
                and not convergence_warning
                and np.isfinite(prediction).all()
            ):

                return (
                    prediction,
                    optimizer
                )

            failures.append(
                f"{optimizer}: did not converge"
            )

        except Exception as error:

            failures.append(
                f"{optimizer}: {error}"
            )

    raise RuntimeError(
        "; ".join(failures)
    )


# ============================================================
# 8. ADVANCED FORECAST DISPATCHER
# ============================================================

def advanced_forecast(
    model_name,
    train,
    future
):

    if model_name == "Moving median 13":

        raw = np.repeat(
            train["Sales"].tail(13).median(),
            len(future)
        )

        method = "Median"

    elif model_name == "Log calendar regression":

        raw = ridge_forecast(
            train,
            future,
            use_price=False,
            log_target=True
        )

        method = "Ridge"

    elif model_name == "Price Ridge regression":

        raw = ridge_forecast(
            train,
            future,
            use_price=True,
            log_target=False
        )

        method = "Ridge"

    elif model_name == "Log price Ridge regression":

        raw = ridge_forecast(
            train,
            future,
            use_price=True,
            log_target=True
        )

        method = "Ridge"

    elif model_name == "Pooled log price regression":

        raw = ridge_forecast(
            train,
            future,
            use_price=True,
            log_target=True,
            pooled=True
        )

        method = "Pooled Ridge"

    elif model_name == "ARIMAX (1,0,0)":

        raw, method = arimax_forecast(
            train,
            future
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    raw = np.asarray(
        raw,
        dtype=float
    )

    assert len(raw) == len(future)
    assert np.isfinite(raw).all()

    return raw, method


# ============================================================
# 9. FORECAST ERROR METRICS
# ============================================================

def metrics(group):

    actual = group["Actual"].to_numpy(
        dtype=float
    )

    forecast = group["Forecast"].to_numpy(
        dtype=float
    )

    error = forecast - actual

    positive = actual > 0

    total_actual = actual.sum()

    return {
        "Observations": len(group),

        "MAPE_Observations": int(
            positive.sum()
        ),

        "MAPE_Pct": (
            np.mean(
                np.abs(error[positive])
                / actual[positive]
            ) * 100
            if positive.any()
            else np.nan
        ),

        "WAPE_Pct": (
            np.abs(error).sum()
            / total_actual * 100
            if total_actual > 0
            else np.nan
        ),

        "MAD": np.abs(error).mean(),

        "MSE": np.mean(error ** 2),

        "Bias_Units": error.mean(),

        "Bias_Pct": (
            error.sum()
            / total_actual * 100
            if total_actual > 0
            else np.nan
        ),

        "Cumulative_Error": error.sum(),

        "Negative_Raw_Forecasts": int(
            group["Forecast_Raw"].lt(0).sum()
        )
    }


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
            metrics(group)
        )

        records.append(row)

    return pd.DataFrame(records)


# ============================================================
# 10. MAIN EVALUATION WORKFLOW
# ============================================================

def main():

    print("=" * 70)

    print(
        "DEMAND PLANNING - ADVANCED FORECASTING"
    )

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

    classical = pd.read_csv(
        OUT / "forecast_classical_predictions.csv",
        parse_dates=[
            "Week",
            "Forecast_Origin"
        ]
    )

    periods = pd.read_csv(
        OUT / "forecast_validation_periods.csv",
        parse_dates=[
            "Training_Start",
            "Training_End",
            "Validation_Start",
            "Validation_End"
        ]
    )

    # Derive exactly the same validation
    # observations as in script 10.

    evaluation = classical[
        classical["Model"] == "Naive"
    ].copy()

    key = [
        "Fold",
        "Series",
        "Week"
    ]

    assert not evaluation.duplicated(
        key
    ).any()

    historical_weeks = sorted(
        history["Week"].unique()
    )

    final_test_start = pd.Timestamp(
        historical_weeks[-13]
    )

    assert evaluation["Week"].max() < (
        final_test_start
    )

    print("\n1. EVALUATION DESIGN")

    print(
        "Common validation observations:",
        len(evaluation)
    )

    print(
        "Reserved final test begins:",
        final_test_start.date()
    )

    print(
        "Additional candidate models:",
        len(ADVANCED_MODELS)
    )

    print(
        "Smoothing models to audit:",
        len(SMOOTHING_MODELS)
    )

    results = []
    fit_log = []
    price_log = []

    # --------------------------------------------------------
    # Store forecasts with consistent metadata.
    # --------------------------------------------------------

    def record(
        future,
        model_name,
        raw,
        method,
        assumed_price=np.nan
    ):

        raw = np.asarray(
            raw,
            dtype=float
        )

        assert len(raw) == len(future)

        assert np.isfinite(raw).all()

        part = future[
            [
                "Fold",
                "Forecast_Origin",
                "Week",
                "Horizon_Week",
                "Series",
                "Region",
                "Product_Group",
                "Actual",
                "Training_Observations"
            ]
        ].copy()

        part["Model"] = model_name

        part["Forecast_Raw"] = raw

        part["Forecast"] = np.maximum(
            0,
            raw
        )

        part["Fit_Method"] = method

        part["Assumed_Price"] = (
            assumed_price
        )

        results.append(part)

    def log_fit(
        fold,
        series,
        model,
        success,
        message
    ):

        fit_log.append({
            "Fold": fold,
            "Series": series,
            "Model": model,
            "Success": success,
            "Message": message
        })

    # --------------------------------------------------------
    # Chronological rolling validation.
    # --------------------------------------------------------

    print("\n2. MODEL ESTIMATION")

    for _, period in periods.iterrows():

        fold = int(
            period["Fold"]
        )

        forecast_origin = (
            period["Training_End"]
        )

        print(
            f"\nFold {fold}: "
            f"{forecast_origin.date()}",
            flush=True
        )

        fold_evaluation = evaluation[
            evaluation["Fold"] == fold
        ].copy()

        training_parts = []
        future_parts = []

        for series, future in fold_evaluation.groupby(
            "Series"
        ):

            future = future.sort_values(
                "Week"
            ).copy().reset_index(drop=True)

            group = history[
                history["Series"] == series
            ]

            train = prepare_training(
                group,
                forecast_origin
            )

            assert (
                train["Week"].max()
                < future["Week"].min()
            )

            assert len(future) == HORIZON

            assert future[
                "Training_Observations"
            ].eq(len(train)).all()

            train, future, assumed_price, source = (
                prepare_prices(
                    train,
                    future
                )
            )

            price_log.append({
                "Fold": fold,
                "Series": series,
                "Forecast_Origin": forecast_origin,
                "Assumed_Price": assumed_price,
                "Price_Assumption": source,
                "Training_Missing_Prices": int(
                    train["Price"].isna().sum()
                )
            })

            training_parts.append(train)
            future_parts.append(future)

            print(
                f"  {series}: "
                f"assumed price "
                f"EUR {assumed_price:.2f}",
                flush=True
            )

            # --------------------------------------------
            # Audit the original smoothing methods.
            # --------------------------------------------

            for model_name in SMOOTHING_MODELS:

                try:

                    raw, method = fit_smoothing(
                        model_name,
                        train["Sales"],
                        len(future)
                    )

                    record(
                        future,
                        model_name,
                        raw,
                        method
                    )

                    log_fit(
                        fold,
                        series,
                        model_name,
                        True,
                        method
                    )

                except Exception as error:

                    log_fit(
                        fold,
                        series,
                        model_name,
                        False,
                        str(error)
                    )

                    print(
                        f"    FAILED: "
                        f"{model_name}: {error}"
                    )

            # --------------------------------------------
            # Independent advanced models.
            # --------------------------------------------

            for model_name in ADVANCED_MODELS:

                if model_name == (
                    "Pooled log price regression"
                ):

                    continue

                try:

                    raw, method = advanced_forecast(
                        model_name,
                        train,
                        future
                    )

                    uses_price = model_name not in [
                        "Moving median 13",
                        "Log calendar regression"
                    ]

                    record(
                        future,
                        model_name,
                        raw,
                        method,
                        assumed_price
                        if uses_price
                        else np.nan
                    )

                    log_fit(
                        fold,
                        series,
                        model_name,
                        True,
                        method
                    )

                except Exception as error:

                    log_fit(
                        fold,
                        series,
                        model_name,
                        False,
                        str(error)
                    )

                    print(
                        f"    FAILED: "
                        f"{model_name}: {error}"
                    )

        # ----------------------------------------------------
        # Joint regional forecasting.
        # ----------------------------------------------------

        pooled_train = pd.concat(
            training_parts,
            ignore_index=True
        )

        pooled_future = pd.concat(
            future_parts,
            ignore_index=True
        )

        for product, future in pooled_future.groupby(
            "Product_Group"
        ):

            train = pooled_train[
                pooled_train["Product_Group"]
                == product
            ].copy()

            future = future.copy()

            assert train["Week"].max() <= (
                forecast_origin
            )

            name = (
                "Pooled log price regression"
            )

            try:

                raw, method = advanced_forecast(
                    name,
                    train,
                    future
                )

                record(
                    future,
                    name,
                    raw,
                    method,
                    future["Model_Price"].to_numpy()
                )

                for series in future["Series"].unique():

                    log_fit(
                        fold,
                        series,
                        name,
                        True,
                        method
                    )

            except Exception as error:

                for series in future["Series"].unique():

                    log_fit(
                        fold,
                        series,
                        name,
                        False,
                        str(error)
                    )

    # ========================================================
    # 11. FIT AUDIT
    # ========================================================

    fit_audit = pd.DataFrame(
        fit_log
    )

    assumed_prices = pd.DataFrame(
        price_log
    )

    new_predictions = pd.concat(
        results,
        ignore_index=True
    )

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

    print("\n3. MODEL CONVERGENCE AUDIT")

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

    # ========================================================
    # 12. COMBINE CLASSICAL AND ADVANCED FORECASTS
    # ========================================================

    # Replace the original smoothing forecasts
    # with the audited, refitted forecasts.
    #
    # Keep the original script 10 files unchanged.

    retained = classical[
        ~classical["Model"].isin(
            SMOOTHING_MODELS
        )
    ].copy()

    retained["Fit_Method"] = (
        "Original classical model"
    )

    retained["Assumed_Price"] = np.nan

    combined = pd.concat(
        [
            retained,
            new_predictions
        ],
        ignore_index=True
    )

    # A candidate is eligible for ranking only
    # when it predicts every common observation.

    expected_keys = set(
        evaluation[key].itertuples(
            index=False,
            name=None
        )
    )

    complete_models = []
    excluded_models = []

    expected_models = set(
        classical["Model"].unique()
    ) | set(
        ADVANCED_MODELS
    )

    for model_name in sorted(expected_models):

        group = combined[
            combined["Model"] == model_name
        ]

        model_keys = set(
            group[key].itertuples(
                index=False,
                name=None
            )
        )

        complete = (
            len(group) == len(evaluation)
            and model_keys == expected_keys
            and not group.duplicated(key).any()
            and np.isfinite(
                group["Forecast"]
            ).all()
        )

        if complete:

            complete_models.append(model_name)

        else:

            excluded_models.append(model_name)

    pred = combined[
        combined["Model"].isin(
            complete_models
        )
    ].copy()

    print("\n4. MODEL COMPARISON COVERAGE")

    print(
        "Complete models:",
        len(complete_models)
    )

    print(
        "Excluded incomplete models:",
        excluded_models
        if excluded_models
        else "None"
    )

    # Recalculate errors because some classical
    # forecasts have been refitted.

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

    # ========================================================
    # 13. PERFORMANCE COMPARISON
    # ========================================================

    overall = summarize(
        pred,
        ["Model"]
    ).sort_values(
        "MAPE_Pct"
    )

    series_metrics = summarize(
        pred,
        ["Series", "Model"]
    )

    fold_metrics = summarize(
        pred,
        ["Fold", "Series", "Model"]
    )

    print("\n5. OVERALL MODEL PERFORMANCE")

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

    mape = series_metrics.pivot(
        index="Series",
        columns="Model",
        values="MAPE_Pct"
    )

    wape = series_metrics.pivot(
        index="Series",
        columns="Model",
        values="WAPE_Pct"
    )

    print("\n6. MAPE BY SERIES (%)")

    print(
        mape.round(2).to_string()
    )

    print("\n7. WAPE BY SERIES (%)")

    print(
        wape.round(2).to_string()
    )

    # ========================================================
    # 14. PROVISIONAL MODEL SELECTION
    # ========================================================

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
            "Bias_Pct"
        ]
    ].rename(
        columns={
            "Model": "Best_MAPE_Model"
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

    original_winners = pd.read_csv(
        OUT / "forecast_classical_winners.csv"
    )

    original_winners = original_winners[
        [
            "Series",
            "Best_MAPE_Model",
            "MAPE_Pct"
        ]
    ].rename(
        columns={
            "Best_MAPE_Model":
                "Original_Best_Model",

            "MAPE_Pct":
                "Original_Best_MAPE"
        }
    )

    winners = winners.merge(
        original_winners,
        on="Series"
    )

    winners["MAPE_Improvement_pp"] = (
        winners["Original_Best_MAPE"]
        - winners["MAPE_Pct"]
    )

    print("\n8. UPDATED PROVISIONAL WINNERS")

    print(
        winners.round(2).to_string(index=False)
    )

    # ========================================================
    # 15. AUDIT CHANGES TO SMOOTHING FORECASTS
    # ========================================================

    old_smoothing = classical[
        classical["Model"].isin(
            SMOOTHING_MODELS
        )
    ]

    new_smoothing = new_predictions[
        new_predictions["Model"].isin(
            SMOOTHING_MODELS
        )
    ]

    smoothing_comparison = old_smoothing[
        key + ["Model", "Forecast"]
    ].merge(
        new_smoothing[
            key + ["Model", "Forecast"]
        ],
        on=key + ["Model"],
        suffixes=("_Original", "_Audited"),
        validate="one_to_one"
    )

    smoothing_comparison[
        "Absolute_Change"
    ] = (
        smoothing_comparison[
            "Forecast_Audited"
        ]
        - smoothing_comparison[
            "Forecast_Original"
        ]
    ).abs()

    print("\n9. SMOOTHING REFIT COMPARISON")

    print(
        smoothing_comparison.groupby(
            "Model"
        )["Absolute_Change"].agg(
            ["mean", "max"]
        ).round(2).to_string()
    )

    # ========================================================
    # 16. VISUALIZATIONS
    # ========================================================

    fig1 = px.imshow(
        mape,
        text_auto=".1f",
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="Combined Forecast Model Comparison - MAPE",
        labels={
            "x": "Forecasting model",
            "y": "Demand series",
            "color": "MAPE (%)"
        },
        template="plotly_white",
        height=700
    )

    fig1.write_html(
        CHART / "21_combined_forecast_mape.html"
    )

    fig2 = px.imshow(
        wape,
        text_auto=".1f",
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="Combined Forecast Model Comparison - WAPE",
        labels={
            "x": "Forecasting model",
            "y": "Demand series",
            "color": "WAPE (%)"
        },
        template="plotly_white",
        height=700
    )

    fig2.write_html(
        CHART / "22_combined_forecast_wape.html"
    )

    fig3 = px.bar(
        overall.sort_values("Bias_Pct"),
        x="Model",
        y="Bias_Pct",
        title="Systematic Forecast Bias by Model",
        labels={
            "Bias_Pct": "Aggregate bias (%)"
        },
        template="plotly_white"
    )

    fig3.add_hline(y=0)

    fig3.write_html(
        CHART / "23_combined_forecast_bias.html"
    )

    # ========================================================
    # 17. VALIDATION
    # ========================================================

    assert "Naive" in complete_models

    assert not pred.duplicated(
        key + ["Model"]
    ).any()

    assert len(pred) == (
        len(evaluation)
        * len(complete_models)
    )

    assert pred.groupby(
        key
    )["Model"].nunique().eq(
        len(complete_models)
    ).all()

    assert (
        pred["Forecast_Origin"]
        < pred["Week"]
    ).all()

    assert pred["Week"].max() < (
        final_test_start
    )

    assert pred["Forecast"].ge(0).all()

    assert np.isfinite(
        pred["Forecast"]
    ).all()

    # Confirm that actual validation observations
    # remain identical to script 10.

    check = pred.merge(
        evaluation[
            key + ["Actual"]
        ],
        on=key,
        suffixes=("", "_Expected"),
        validate="many_to_one"
    )

    assert np.allclose(
        check["Actual"],
        check["Actual_Expected"]
    )

    assert assumed_prices[
        "Assumed_Price"
    ].gt(0).all()

    # Independent price assumption test.

    test_train = pd.DataFrame({
        "Price": [
            10.0, 20.0, 30.0
        ]
    })

    test_future = pd.DataFrame(
        index=range(3)
    )

    _, test_future, test_price, _ = (
        prepare_prices(
            test_train,
            test_future
        )
    )

    assert np.isclose(
        test_price,
        20.0
    )

    assert test_future[
        "Model_Price"
    ].eq(20.0).all()

    # Independent forecast error calculation.

    test = pd.DataFrame({
        "Actual": [100.0],
        "Forecast": [120.0],
        "Forecast_Raw": [120.0]
    })

    test_metrics = metrics(test)

    assert np.isclose(
        test_metrics["MAPE_Pct"],
        20.0
    )

    assert np.isclose(
        test_metrics["Bias_Pct"],
        20.0
    )

    print("\n10. VALIDATION")

    print(
        "PASS - Complete candidates use "
        "identical evaluation observations."
    )

    print(
        "PASS - Actual validation sales "
        "match script 10."
    )

    print(
        "PASS - Chronological training preserved."
    )

    print(
        "PASS - Reserved test period unused."
    )

    print(
        "PASS - No duplicate forecasts."
    )

    print(
        "PASS - All operational forecasts finite "
        "and nonnegative."
    )

    print(
        "PASS - Independent price assumption "
        "test verified."
    )

    print(
        "PASS - Forecast metrics verified."
    )

    # ========================================================
    # 18. EXPORT
    # ========================================================

    pred.to_csv(
        OUT / "forecast_combined_predictions.csv",
        index=False
    )

    overall.to_csv(
        OUT / "forecast_combined_overall_metrics.csv",
        index=False
    )

    series_metrics.to_csv(
        OUT / "forecast_combined_series_metrics.csv",
        index=False
    )

    fold_metrics.to_csv(
        OUT / "forecast_combined_fold_metrics.csv",
        index=False
    )

    winners.to_csv(
        OUT / "forecast_combined_winners.csv",
        index=False
    )

    fit_audit.to_csv(
        OUT / "forecast_model_fit_audit.csv",
        index=False
    )

    fit_summary.to_csv(
        OUT / "forecast_convergence_summary.csv",
        index=False
    )

    assumed_prices.to_csv(
        OUT / "forecast_validation_price_assumptions.csv",
        index=False
    )

    smoothing_comparison.to_csv(
        OUT / "forecast_smoothing_refit_comparison.csv",
        index=False
    )

    print("\n" + "=" * 70)

    print(
        "Advanced forecasting evaluation "
        "completed successfully!"
    )

    print(
        "Complete forecasting models:",
        len(complete_models)
    )

    print(
        "Total comparable forecasts:",
        len(pred)
    )

    print("Results saved to:", OUT)
    print("Charts saved to:", CHART)

    print(
        "\nFuture prices estimated using "
        "training observations only."
    )

    print(
        "Final historical test remains reserved."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
