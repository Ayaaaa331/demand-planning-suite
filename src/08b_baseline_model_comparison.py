
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px

from dateutil.easter import easter
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CHART = OUT / "charts"

CHART.mkdir(parents=True, exist_ok=True)

ALPHA = 5.0
MIN_TRAIN = 30

# Reuse exactly the assessment sample and
# validation blocks from script 08.

data = pd.read_csv(
    OUT / "baseline_demand_estimates.csv",
    parse_dates=["Week"]
)

source = pd.read_csv(
    OUT / "price_analysis_enriched.csv",
    parse_dates=["Week"]
)

origin = source["Week"].min()

data = data.sort_values(
    ["Series", "Week"]
).reset_index(drop=True)

assert not data.duplicated(
    ["Series", "Week"]
).any()

assert data["Price"].gt(0).all()
assert data["Reference_Price"].gt(0).all()
assert data["Sales"].gt(0).all()

assert data["Validation_Block"].eq(
    (data["Week"] - origin).dt.days // 28
).all()

# ============================================================
# 2. CALENDAR FEATURES
# ============================================================

# Use the same calendar definitions as script 08.

def monday(date):

    return date - pd.Timedelta(
        days=date.weekday()
    )


def event_flags(week):

    y = week.year

    thursdays = pd.date_range(
        f"{y}-11-01",
        f"{y}-11-30",
        freq="W-THU"
    )

    thanksgiving = thursdays[3]

    black_friday_week = monday(thanksgiving)

    cyber_monday = (
        thanksgiving + pd.Timedelta(days=4)
    )

    christmas_week = monday(
        pd.Timestamp(y, 12, 25)
    )

    easter_monday = (
        pd.Timestamp(easter(y))
        + pd.Timedelta(days=1)
    )

    return (
        int(
            week in [
                black_friday_week,
                cyber_monday
            ]
        ),
        int(
            week in [
                christmas_week - pd.Timedelta(weeks=1),
                christmas_week
            ]
        ),
        int(
            week in [
                easter_monday - pd.Timedelta(weeks=1),
                easter_monday
            ]
        )
    )


def features(
    frame,
    price_mode,
    pooled,
    reference=False
):

    t = (
        (frame["Week"] - origin).dt.days / 7
    )

    angle = 2 * np.pi * t / 52.1775

    X = pd.DataFrame(
        {
            "Trend": t,
            "Annual_Sin": np.sin(angle),
            "Annual_Cos": np.cos(angle)
        },
        index=frame.index
    )

    X[
        ["Shopping_Event", "Christmas", "Easter"]
    ] = np.array(
        [event_flags(w) for w in frame["Week"]]
    )

    if price_mode != "none":

        relative = np.log(
            frame["Price"]
            / frame["Reference_Price"]
        )

        X["Log_Relative_Price"] = (
            0.0 if reference else relative
        )

        if price_mode == "quadratic":

            X["Relative_Price_Squared"] = (
                X["Log_Relative_Price"] ** 2
            )

    if pooled:

        # Region-specific intercepts,
        # common trend and price response.

        X["South"] = (
            frame["Region"].eq("South")
            .astype(float)
        )

    return X.astype(float)


# ============================================================
# 3. DEFINE CANDIDATE MODELS
# ============================================================

# Model specification:
# Name, pooled regions, log target, price mode.
#
# Settings are fixed before model comparison.

SPECS = [
    (
        "Independent level + price",
        False,
        False,
        "linear"
    ),
    (
        "Independent log + price",
        False,
        True,
        "linear"
    ),
    (
        "Pooled level + price",
        True,
        False,
        "linear"
    ),
    (
        "Pooled log + price",
        True,
        True,
        "linear"
    ),
    (
        "Pooled log + nonlinear price",
        True,
        True,
        "quadratic"
    ),
    (
        "Independent log calendar only",
        False,
        True,
        "none"
    ),
    (
        "Pooled log calendar only",
        True,
        True,
        "none"
    )
]

BENCHMARK = SPECS[0][0]

KEY = ["Series", "Week"]

KEEP = [
    "Week",
    "Series",
    "Region",
    "Product_Group",
    "Sales",
    "Price",
    "Reference_Price",
    "Discount_Pct",
    "Potential_Promotion",
    "Validation_Block"
]

# ============================================================
# 4. MODEL TRAINING AND PREDICTION FUNCTION
# ============================================================

def fit_predict(
    train,
    test,
    log_target,
    price_mode,
    pooled
):

    X_train = features(
        train,
        price_mode,
        pooled
    )

    X_test = features(
        test,
        price_mode,
        pooled
    )

    X_ref = features(
        test,
        price_mode,
        pooled,
        reference=True
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

    actual_raw = model.predict(X_test)

    baseline_raw = model.predict(X_ref)

    if log_target:

        # Duan smearing correction.
        #
        # Correct the log-to-unit retransformation
        # towards arithmetic mean demand.
        #
        # Use training residuals only.

        residual = (
            target - model.predict(X_train)
        )

        smear = pd.Series(
            np.exp(residual),
            index=train.index
        )

        factors = smear.groupby(
            train["Region"]
        ).mean()

        factor = test["Region"].map(
            factors
        ).to_numpy(dtype=float)

        actual_raw = (
            np.exp(actual_raw) * factor - 1
        )

        baseline_raw = (
            np.exp(baseline_raw) * factor - 1
        )

    return actual_raw, baseline_raw


print("=" * 70)
print("DEMAND PLANNING - BASELINE MODEL COMPARISON")
print("=" * 70)

print(
    f"Common observations: {len(data)}"
)

print(
    f"Promotion candidates: "
    f"{data['Potential_Promotion'].sum()}"
)

# ============================================================
# 5. RETROSPECTIVE OUT-OF-BLOCK PREDICTIONS
# ============================================================

results = []

for name, pooled, log_target, price_mode in SPECS:

    print(
        "Evaluating:",
        name,
        flush=True
    )

    grouping = (
        "Product_Group"
        if pooled
        else "Series"
    )

    for _, group in data.groupby(grouping):

        blocks = sorted(
            group["Validation_Block"].unique()
        )

        for block in blocks:

            train = group[
                group["Validation_Block"] != block
            ]

            test = group[
                group["Validation_Block"] == block
            ]

            assert len(train) >= MIN_TRAIN

            # Both regions' observations from the
            # test block stay out of training.

            assert set(
                train["Week"]
            ).isdisjoint(
                test["Week"]
            )

            raw, baseline_raw = fit_predict(
                train,
                test,
                log_target,
                price_mode,
                pooled
            )

            part = test[KEEP].copy()

            part["Model"] = name

            part["Uses_Price"] = (
                price_mode != "none"
            )

            part["Actual_Prediction_Raw"] = raw

            part["Baseline_Raw"] = baseline_raw

            part["Predicted_At_Actual_Price"] = (
                np.maximum(0, raw)
            )

            part["Baseline_Demand"] = (
                np.maximum(0, baseline_raw)
            )

            train_ratio = np.log(
                train["Price"]
                / train["Reference_Price"]
            )

            test_ratio = np.log(
                test["Price"]
                / test["Reference_Price"]
            )

            part[
                "Actual_Price_Within_Training_Range"
            ] = test_ratio.between(
                train_ratio.min(),
                train_ratio.max()
            )

            part[
                "Reference_Within_Training_Range"
            ] = (
                train_ratio.min()
                <= 0
                <= train_ratio.max()
            )

            results.append(part)

pred = pd.concat(
    results,
    ignore_index=True
)

assert len(pred) == (
    len(data) * len(SPECS)
)

assert not pred.duplicated(
    ["Model"] + KEY
).any()

assert np.isfinite(
    pred[
        [
            "Predicted_At_Actual_Price",
            "Baseline_Demand"
        ]
    ]
).all().all()

# ============================================================
# 6. REPRODUCE ORIGINAL BENCHMARK
# ============================================================

# Confirm that the comparison reproduces
# the original script 08 predictions.

replay = pred[
    pred["Model"] == BENCHMARK
].merge(
    data[
        KEY + [
            "Predicted_At_Actual_Price",
            "Baseline_Demand"
        ]
    ],
    on=KEY,
    validate="one_to_one",
    suffixes=("_new", "_old")
)

for col in [
    "Predicted_At_Actual_Price",
    "Baseline_Demand"
]:

    assert np.allclose(
        replay[col + "_new"],
        replay[col + "_old"],
        rtol=1e-6,
        atol=1e-5
    ), (
        f"Script 08 benchmark does not match: "
        f"{col}. Check settings before continuing."
    )

print(
    "PASS - Original script 08 benchmark reproduced."
)

# ============================================================
# 7. ACCURACY METRICS
# ============================================================

def metrics(group):

    actual = group["Sales"].to_numpy(
        dtype=float
    )

    prediction = group[
        "Predicted_At_Actual_Price"
    ].to_numpy(dtype=float)

    # Professor's error convention:
    # Positive error = overforecasting.

    error = prediction - actual

    nonzero = actual > 0

    return {
        "N": len(actual),

        "MAPE_Pct": (
            100 * np.mean(
                np.abs(
                    error[nonzero]
                    / actual[nonzero]
                )
            )
        ),

        "MAD": np.mean(
            np.abs(error)
        ),

        "MSE": np.mean(
            error ** 2
        ),

        "WAPE_Pct": (
            100
            * np.abs(error).sum()
            / actual.sum()
        ),

        "Bias_Units": error.mean(),

        "Zero_Baselines": int(
            group["Baseline_Demand"].eq(0).sum()
        )
    }


rows = []

samples = [
    (
        "All assessable",
        pred
    ),
    (
        "Non-flagged",
        pred[~pred["Potential_Promotion"]]
    ),
    (
        "Flagged",
        pred[pred["Potential_Promotion"]]
    )
]

for label, sample in samples:

    for (model, series), group in sample.groupby(
        ["Model", "Series"]
    ):

        rows.append({
            "Sample": label,
            "Model": model,
            "Series": series,
            **metrics(group)
        })

accuracy = pd.DataFrame(rows)

all_accuracy = accuracy[
    accuracy["Sample"] == "All assessable"
]

regular_accuracy = accuracy[
    accuracy["Sample"] == "Non-flagged"
]

assert all_accuracy.groupby(
    "Series"
)["N"].nunique().eq(1).all()

# ============================================================
# 8. PRINT MODEL COMPARISON RESULTS
# ============================================================

print("\n1. MAPE (%) - ALL ASSESSABLE WEEKS")

print(
    all_accuracy.pivot(
        index="Series",
        columns="Model",
        values="MAPE_Pct"
    ).round(1).to_string()
)

print("\n2. WAPE (%) - ALL ASSESSABLE WEEKS")

print(
    all_accuracy.pivot(
        index="Series",
        columns="Model",
        values="WAPE_Pct"
    ).round(1).to_string()
)

print("\n3. WAPE (%) - NON-FLAGGED WEEKS")

print(
    regular_accuracy.pivot(
        index="Series",
        columns="Model",
        values="WAPE_Pct"
    ).round(1).to_string()
)

# ============================================================
# 9. IDENTIFY LOWEST-ERROR MODELS
# ============================================================

# These are descriptive comparison winners,
# not independently validated model selections.

best_wape = all_accuracy.loc[
    all_accuracy.groupby(
        "Series"
    )["WAPE_Pct"].idxmin()
]

best_mape = all_accuracy.loc[
    all_accuracy.groupby(
        "Series"
    )["MAPE_Pct"].idxmin()
]

base = all_accuracy[
    all_accuracy["Model"] == BENCHMARK
][
    [
        "Series",
        "WAPE_Pct",
        "MAPE_Pct"
    ]
]

winners = best_wape[
    [
        "Series",
        "Model",
        "WAPE_Pct"
    ]
].rename(
    columns={
        "Model": "Best_WAPE_Model"
    }
)

winners = winners.merge(
    best_mape[
        [
            "Series",
            "Model",
            "MAPE_Pct"
        ]
    ].rename(
        columns={
            "Model": "Best_MAPE_Model"
        }
    ),
    on="Series"
)

winners = winners.merge(
    base,
    on="Series",
    suffixes=("", "_Benchmark")
)

winners["WAPE_Improvement_pp"] = (
    winners["WAPE_Pct_Benchmark"]
    - winners["WAPE_Pct"]
)

print("\n4. LOWEST OBSERVED ERRORS BY SERIES")

print(
    winners.round(2).to_string(index=False)
)

# ============================================================
# 10. ROBUSTNESS OF COMMERCIAL EFFECTS
# ============================================================

# Compare only price-aware models.
#
# Calendar-only models cannot identify
# a model-implied price response.
#
# All models must use identical
# promotion observations.

promo = pred[
    pred["Potential_Promotion"]
    & pred["Uses_Price"]
].copy()

promo["Comparable"] = (
    promo["Baseline_Raw"].gt(0)
    & promo["Reference_Within_Training_Range"]
)

# Strict sample also excludes
# observed price extrapolation.

promo["Strict"] = (
    promo["Comparable"]
    & promo["Actual_Price_Within_Training_Range"]
)

promo["Actual_Revenue_Gap"] = (
    promo["Sales"] * promo["Price"]
    - promo["Baseline_Demand"]
    * promo["Reference_Price"]
)

promo["Model_Revenue_Effect"] = (
    promo["Predicted_At_Actual_Price"]
    * promo["Price"]
    - promo["Baseline_Demand"]
    * promo["Reference_Price"]
)

promo["Model_Price_Effect_Units"] = (
    promo["Predicted_At_Actual_Price"]
    - promo["Baseline_Demand"]
)

robustness = []

comparison_samples = [
    (
        "Common positive baseline",
        "Comparable"
    ),
    (
        "Strict common sample",
        "Strict"
    )
]

for label, flag in comparison_samples:

    common_keys = promo.groupby(
        KEY
    )[flag].all().rename(
        "Keep"
    ).reset_index()

    common = promo.merge(
        common_keys[
            common_keys["Keep"]
        ][KEY],
        on=KEY,
        how="inner"
    )

    print(
        f"\n{label}: "
        f"{len(common_keys[common_keys['Keep']])} "
        f"of {len(common_keys)} promotions"
    )

    for (product, model), group in common.groupby(
        ["Product_Group", "Model"]
    ):

        robustness.append({
            "Sample": label,

            "Product_Group": product,

            "Model": model,

            "N": len(group),

            "Average_Model_Price_Effect_Units": (
                group["Model_Price_Effect_Units"]
                .mean()
            ),

            "Actual_Revenue_Gap": (
                group["Actual_Revenue_Gap"]
                .sum()
            ),

            "Model_Revenue_Effect": (
                group["Model_Revenue_Effect"]
                .sum()
            )
        })

robustness = pd.DataFrame(robustness)

print(
    "\n5. MODEL-IMPLIED REVENUE EFFECT "
    "BY PRODUCT (EUR)"
)

if not robustness.empty:

    strict = robustness[
        robustness["Sample"]
        == "Strict common sample"
    ]

    print(
        strict.pivot(
            index="Product_Group",
            columns="Model",
            values="Model_Revenue_Effect"
        ).round(0).to_string()
    )

# ============================================================
# 11. EXPORT ANALYTICAL RESULTS
# ============================================================

accuracy.to_csv(
    OUT / "baseline_model_comparison_metrics.csv",
    index=False
)

pred.to_csv(
    OUT / "baseline_candidate_predictions.csv",
    index=False
)

winners.to_csv(
    OUT / "baseline_model_comparison_winners.csv",
    index=False
)

robustness.to_csv(
    OUT / "baseline_model_robustness.csv",
    index=False
)

# ============================================================
# 12. CREATE INTERACTIVE COMPARISON CHARTS
# ============================================================

chart_settings = [
    (
        "WAPE_Pct",
        "13b_baseline_wape.html"
    ),
    (
        "MAPE_Pct",
        "13c_baseline_mape.html"
    )
]

for metric, filename in chart_settings:

    fig = px.bar(
        all_accuracy,
        x="Series",
        y=metric,
        color="Model",
        barmode="group",
        title=(
            f"Baseline comparison: "
            f"{metric.replace('_Pct', ' (%)')}"
        ),
        template="plotly_white",
        height=650
    )

    fig.update_layout(
        xaxis_title="Demand series",
        legend_title="Model specification"
    )

    fig.write_html(
        CHART / filename
    )

# ============================================================
# 13. VALIDATION AND COMPLETION
# ============================================================

print("\n6. VALIDATION")

print(
    "PASS - All candidates evaluated "
    "on identical observations."
)

print(
    "PASS - Both regions held out "
    "together in pooled models."
)

print(
    "PASS - Log retransformation uses "
    "training residuals only."
)

print(
    "PASS - All predictions are finite."
)

print("\n" + "=" * 70)

print(
    "Model comparison completed successfully!"
)

print("Analytical tables saved to:", OUT)

print("Interactive charts saved to:", CHART)

print(
    "Retrospective diagnostics only; "
    "not causal estimates or future forecast accuracy."
)

print("=" * 70)
