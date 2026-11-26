
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from dateutil.easter import easter

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CHART = OUT / "charts"

CHART.mkdir(parents=True, exist_ok=True)

ALPHA = 5.0
MIN_PRICES = 6
PROMOTION_THRESHOLD = 10
BOOTSTRAP_REPETITIONS = 200
RANDOM_SEED = 42

df = pd.read_csv(
    OUT / "price_analysis_enriched.csv",
    parse_dates=["Week"]
)

df = df.sort_values(
    ["Series", "Week"]
).reset_index(drop=True)

origin = df["Week"].min()

print("=" * 70)
print("DEMAND PLANNING - REFERENCE PRICE SENSITIVITY")
print("=" * 70)

# ============================================================
# 2. ALTERNATIVE REFERENCE PRICES
# ============================================================

REFERENCE_SPECS = {
    "Median 8 weeks": (8, 0.50),
    "Median 13 weeks": (13, 0.50),
    "Median 26 weeks": (26, 0.50),
    "75th percentile 13 weeks": (13, 0.75)
}

for name, (window, quantile) in REFERENCE_SPECS.items():

    df[name] = df.groupby("Series")["Price"].transform(
        lambda x: (
            x.shift(1)
            .rolling(
                window=window,
                min_periods=MIN_PRICES
            )
            .quantile(quantile)
        )
    )

# Confirm that the original reference price
# is reproduced exactly.

assert np.allclose(
    df["Median 13 weeks"],
    df["Reference_Price"],
    equal_nan=True
)

print("PASS - Original reference prices reproduced.")

# ============================================================
# 3. CREATE COMMON COMPARISON SAMPLE
# ============================================================

# Use identical observations for every reference
# definition and model specification.

common_mask = df["Use_For_Price_Analysis"].copy()

for name in REFERENCE_SPECS:

    common_mask &= (
        df[name].notna()
        & df[name].gt(0)
    )

data = df[common_mask].copy().reset_index(drop=True)

data["Row_ID"] = np.arange(len(data))

data["Block"] = (
    (data["Week"] - origin).dt.days // 28
)

data["Original_Promotion"] = (
    data["Potential_Promotion"]
)

original_candidates = int(
    df["Potential_Promotion"].sum()
)

print("\n1. COMPARISON SAMPLE")

print(
    "Original assessable observations:",
    df["Price_Analysis_Eligible"].sum()
)

print(
    "Common observations:",
    len(data)
)

print(
    "Original promotion candidates:",
    original_candidates
)

print(
    "Original candidates in common sample:",
    data["Original_Promotion"].sum()
)

# ============================================================
# 4. PROMOTION FREQUENCY SENSITIVITY
# ============================================================

frequency_records = []

for reference in REFERENCE_SPECS:

    discount = (
        1 - data["Price"] / data[reference]
    ) * 100

    for product, group in data.groupby("Product_Group"):

        group_discount = discount.loc[group.index]

        count = group_discount.ge(
            PROMOTION_THRESHOLD
        ).sum()

        frequency_records.append({
            "Reference": reference,
            "Product_Group": product,
            "Assessable_Weeks": len(group),
            "Promotion_Candidates": count,
            "Frequency_Pct": (
                count / len(group) * 100
            )
        })

frequency = pd.DataFrame(frequency_records)

print("\n2. PROMOTION COUNTS BY REFERENCE PRICE")

print(
    frequency.pivot(
        index="Product_Group",
        columns="Reference",
        values="Promotion_Candidates"
    ).to_string()
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

    black_friday = monday(thanksgiving)

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


t = (
    (data["Week"] - origin).dt.days / 7
).to_numpy()

angle = 2 * np.pi * t / 52.1775

events = np.array([
    event_flags(w)
    for w in data["Week"]
])

calendar = np.column_stack([
    t,
    np.sin(angle),
    np.cos(angle),
    events
]).astype(float)

region = data["Region"].to_numpy()
sales = data["Sales"].to_numpy(dtype=float)
prices = data["Price"].to_numpy(dtype=float)
blocks = data["Block"].to_numpy()

# ============================================================
# 6. MODEL SPECIFICATIONS
# ============================================================

# Name, pooled, log target, nonlinear price.

SPECS = [
    (
        "Independent level + price",
        False, False, False
    ),
    (
        "Independent log + price",
        False, True, False
    ),
    (
        "Pooled level + price",
        True, False, False
    ),
    (
        "Pooled log + price",
        True, True, False
    ),
    (
        "Pooled log + nonlinear price",
        True, True, True
    )
]

# ============================================================
# 7. FEATURE MATRICES
# ============================================================

def make_features(reference, pooled, nonlinear):

    ratio = np.log(
        prices / data[reference].to_numpy()
    )

    actual_features = [
        calendar,
        ratio
    ]

    reference_features = [
        calendar,
        np.zeros(len(data))
    ]

    if nonlinear:

        actual_features.append(ratio ** 2)

        reference_features.append(
            np.zeros(len(data))
        )

    if pooled:

        south = (
            region == "South"
        ).astype(float)

        actual_features.append(south)
        reference_features.append(south)

    return (
        np.column_stack(actual_features),
        np.column_stack(reference_features),
        ratio
    )

# ============================================================
# 8. WEIGHTED RIDGE REGRESSION
# ============================================================

# Supports normal model estimation and
# block-bootstrap resampling.
#
# Standardization and log retransformation
# use training observations only.

def fit_predict(
    X,
    X_ref,
    train,
    test,
    log_target,
    weights
):

    w = weights[train]

    if np.count_nonzero(w) < 15:
        raise ValueError(
            "Insufficient distinct training rows."
        )

    X_train = X[train]

    y = sales[train]

    target = (
        np.log1p(y)
        if log_target
        else y
    )

    mean = np.average(
        X_train,
        axis=0,
        weights=w
    )

    variance = np.average(
        (X_train - mean) ** 2,
        axis=0,
        weights=w
    )

    scale = np.sqrt(variance)

    scale[scale < 1e-12] = 1.0

    Z = (X_train - mean) / scale

    target_mean = np.average(
        target,
        weights=w
    )

    lhs = (
        Z.T @ (w[:, None] * Z)
        + ALPHA * np.eye(Z.shape[1])
    )

    rhs = (
        Z.T @ (w * (target - target_mean))
    )

    beta = np.linalg.solve(lhs, rhs)

    actual_prediction = (
        (X[test] - mean) / scale
    ) @ beta + target_mean

    baseline_prediction = (
        (X_ref[test] - mean) / scale
    ) @ beta + target_mean

    if log_target:

        training_prediction = (
            Z @ beta + target_mean
        )

        residual = (
            target - training_prediction
        )

        smear = np.exp(residual)

        actual_units = np.zeros(len(test))
        baseline_units = np.zeros(len(test))

        for reg in np.unique(region[test]):

            train_region = (
                (region[train] == reg)
                & (w > 0)
            )

            test_region = (
                region[test] == reg
            )

            if not train_region.any():

                raise ValueError(
                    "Missing bootstrap region."
                )

            factor = np.average(
                smear[train_region],
                weights=w[train_region]
            )

            actual_units[test_region] = (
                np.exp(
                    actual_prediction[test_region]
                ) * factor - 1
            )

            baseline_units[test_region] = (
                np.exp(
                    baseline_prediction[test_region]
                ) * factor - 1
            )

        actual_prediction = actual_units
        baseline_prediction = baseline_units

    return (
        actual_prediction,
        baseline_prediction
    )

# ============================================================
# 9. RETROSPECTIVE OUT-OF-BLOCK ESTIMATION
# ============================================================

def crossfit(
    reference,
    pooled,
    log_target,
    nonlinear,
    weights=None,
    prepared=None
):

    if weights is None:

        weights = np.ones(len(data))

    if prepared is None:

        X, X_ref, ratio = make_features(
            reference,
            pooled,
            nonlinear
        )

    else:

        X, X_ref, ratio = prepared

    grouping = (
        "Product_Group"
        if pooled
        else "Series"
    )

    actual_raw = np.full(len(data), np.nan)
    baseline_raw = np.full(len(data), np.nan)

    support = np.zeros(
        len(data),
        dtype=bool
    )

    for _, indices in data.groupby(
        grouping
    ).indices.items():

        indices = np.asarray(indices)

        for block in np.unique(blocks[indices]):

            train = indices[
                blocks[indices] != block
            ]

            test = indices[
                blocks[indices] == block
            ]

            if weights[train].sum() < 30:

                raise ValueError(
                    "Insufficient training weight."
                )

            raw, base = fit_predict(
                X,
                X_ref,
                train,
                test,
                log_target,
                weights
            )

            actual_raw[test] = raw
            baseline_raw[test] = base

            used_train = train[
                weights[train] > 0
            ]

            low = ratio[used_train].min()
            high = ratio[used_train].max()

            support[test] = (
                (ratio[test] >= low)
                & (ratio[test] <= high)
                & (low <= 0)
                & (high >= 0)
            )

    assert np.isfinite(actual_raw).all()
    assert np.isfinite(baseline_raw).all()

    return actual_raw, baseline_raw, support

# ============================================================
# 10. ESTIMATE ALL MODEL / REFERENCE COMBINATIONS
# ============================================================

results = []

for reference in REFERENCE_SPECS:

    for name, pooled, log_target, nonlinear in SPECS:

        print(
            f"Evaluating: {reference} | {name}",
            flush=True
        )

        raw, baseline_raw, support = crossfit(
            reference,
            pooled,
            log_target,
            nonlinear
        )

        part = data[
            [
                "Row_ID",
                "Week",
                "Series",
                "Region",
                "Product_Group",
                "Sales",
                "Price",
                "Original_Promotion"
            ]
        ].copy()

        part["Reference_Definition"] = reference
        part["Model"] = name
        part["Reference_Price"] = data[reference]

        part["Discount_Pct"] = (
            1 - part["Price"] / part["Reference_Price"]
        ) * 100

        part["Promotion"] = (
            part["Discount_Pct"]
            >= PROMOTION_THRESHOLD
        )

        part["Predicted_Actual_Price"] = (
            np.maximum(0, raw)
        )

        part["Baseline_Demand"] = (
            np.maximum(0, baseline_raw)
        )

        part["Supported"] = (
            support & (baseline_raw > 0)
        )

        part["Actual_Revenue_Gap"] = (
            part["Sales"] * part["Price"]
            - part["Baseline_Demand"]
            * part["Reference_Price"]
        )

        part["Model_Revenue_Effect"] = (
            part["Predicted_Actual_Price"]
            * part["Price"]
            - part["Baseline_Demand"]
            * part["Reference_Price"]
        )

        error = (
            part["Predicted_Actual_Price"]
            - part["Sales"]
        )

        part["Absolute_Error"] = error.abs()

        results.append(part)

pred = pd.concat(
    results,
    ignore_index=True
)

# ============================================================
# 11. FIXED PROMOTION SAMPLE
# ============================================================

# Compare exactly the same original promotion
# observations under every model and reference.
#
# Some observations may no longer qualify as a
# promotion under a different reference price.
# They remain here for controlled comparison.

supported_all = pred.groupby(
    "Row_ID"
)["Supported"].all()

fixed_ids = data.loc[
    data["Original_Promotion"]
    & data["Row_ID"].map(supported_all),
    "Row_ID"
]

fixed = pred[
    pred["Row_ID"].isin(fixed_ids)
].copy()

print("\n3. FIXED STRICT PROMOTION SAMPLE")

print(
    "Common supported promotion observations:",
    len(fixed_ids)
)

print(
    "Excluded original candidates:",
    original_candidates - len(fixed_ids)
)

assert not fixed.empty

fixed_summary = fixed.groupby([
    "Product_Group",
    "Reference_Definition",
    "Model"
]).agg(
    N=("Row_ID", "count"),
    Actual_Revenue_Gap=("Actual_Revenue_Gap", "sum"),
    Model_Revenue_Effect=("Model_Revenue_Effect", "sum")
).reset_index()

# ============================================================
# 12. MODEL AND REFERENCE UNCERTAINTY RANGES
# ============================================================

robustness = fixed_summary.groupby(
    "Product_Group"
).agg(
    Specifications=("Model_Revenue_Effect", "count"),
    Evaluated_Promotions=("N", "first"),
    Lowest_Estimate=("Model_Revenue_Effect", "min"),
    Median_Estimate=("Model_Revenue_Effect", "median"),
    Highest_Estimate=("Model_Revenue_Effect", "max"),
    Positive_Estimates=(
        "Model_Revenue_Effect",
        lambda x: (x > 0).sum()
    ),
    Negative_Estimates=(
        "Model_Revenue_Effect",
        lambda x: (x < 0).sum()
    )
).reset_index()

robustness["Conclusion"] = np.select(
    [
        robustness["Lowest_Estimate"] > 0,
        robustness["Highest_Estimate"] < 0
    ],
    [
        "Consistently positive",
        "Consistently negative"
    ],
    default="Sensitive to assumptions"
)

print("\n4. REVENUE EFFECT ROBUSTNESS - FIXED SAMPLE")

print(
    robustness.round(2).to_string(index=False)
)

# ============================================================
# 13. REFERENCE-SPECIFIC PROMOTION SAMPLES
# ============================================================

# Also evaluate promotions according to each
# reference price's own candidate definition.

supported_reference = pred.groupby([
    "Reference_Definition",
    "Row_ID"
])["Supported"].transform("all")

native = pred[
    pred["Promotion"]
    & supported_reference
].copy()

native_summary = native.groupby([
    "Product_Group",
    "Reference_Definition",
    "Model"
]).agg(
    N=("Row_ID", "count"),
    Actual_Revenue_Gap=("Actual_Revenue_Gap", "sum"),
    Model_Revenue_Effect=("Model_Revenue_Effect", "sum")
).reset_index()

native_robustness = native_summary.groupby([
    "Product_Group",
    "Reference_Definition"
]).agg(
    Evaluated_Promotions=("N", "first"),
    Lowest_Estimate=("Model_Revenue_Effect", "min"),
    Highest_Estimate=("Model_Revenue_Effect", "max"),
    Positive_Estimates=(
        "Model_Revenue_Effect",
        lambda x: (x > 0).sum()
    ),
    Negative_Estimates=(
        "Model_Revenue_Effect",
        lambda x: (x < 0).sum()
    )
).reset_index()

print("\n5. REFERENCE-SPECIFIC PROMOTION RESULTS")

print(
    native_robustness.round(2).to_string(index=False)
)

# ============================================================
# 14. BLOCK BOOTSTRAP
# ============================================================

# Bootstrap one parsimonious price-aware model:
# pooled log regression with the original
# 13-week median reference price.
#
# Both regions share the same sampled time-block
# weights. Each evaluation block is still
# excluded from its corresponding training fit.
#
# Reference prices and evaluation observations
# remain fixed during bootstrap estimation.

BOOT_REFERENCE = "Median 13 weeks"
BOOT_MODEL = "Pooled log + price"

prepared = make_features(
    BOOT_REFERENCE,
    pooled=True,
    nonlinear=False
)

reference_values = (
    data[BOOT_REFERENCE].to_numpy()
)

unique_blocks = np.unique(blocks)

block_positions = np.searchsorted(
    unique_blocks,
    blocks
)

fixed_mask = data["Row_ID"].isin(
    fixed_ids
).to_numpy()

products = data["Product_Group"].to_numpy()

rng = np.random.default_rng(
    RANDOM_SEED
)

bootstrap_records = []
successful_repetitions = 0

print("\n6. BLOCK BOOTSTRAP")

for repetition in range(BOOTSTRAP_REPETITIONS):

    sampled = rng.integers(
        0,
        len(unique_blocks),
        size=len(unique_blocks)
    )

    counts = np.bincount(
        sampled,
        minlength=len(unique_blocks)
    )

    weights = counts[block_positions].astype(float)

    try:

        actual_raw, baseline_raw, _ = crossfit(
            BOOT_REFERENCE,
            pooled=True,
            log_target=True,
            nonlinear=False,
            weights=weights,
            prepared=prepared
        )

    except (ValueError, np.linalg.LinAlgError):

        continue

    actual_prediction = np.maximum(
        0, actual_raw
    )

    baseline = np.maximum(
        0, baseline_raw
    )

    model_effect = (
        actual_prediction * prices
        - baseline * reference_values
    )

    actual_gap = (
        sales * prices
        - baseline * reference_values
    )

    if not (
        np.isfinite(model_effect).all()
        and np.isfinite(actual_gap).all()
    ):

        continue

    successful_repetitions += 1

    for product in np.unique(products[fixed_mask]):

        mask = (
            fixed_mask
            & (products == product)
        )

        bootstrap_records.append({
            "Repetition": repetition + 1,
            "Product_Group": product,
            "Model_Revenue_Effect": (
                model_effect[mask].sum()
            ),
            "Actual_Revenue_Gap": (
                actual_gap[mask].sum()
            )
        })

    if (repetition + 1) % 25 == 0:

        print(
            f"Bootstrap progress: "
            f"{repetition + 1}/"
            f"{BOOTSTRAP_REPETITIONS}",
            flush=True
        )

bootstrap = pd.DataFrame(
    bootstrap_records
)

print(
    "Successful repetitions:",
    successful_repetitions
)

assert successful_repetitions >= (
    BOOTSTRAP_REPETITIONS * 0.80
), "Too many failed bootstrap repetitions."

# ============================================================
# 15. BOOTSTRAP UNCERTAINTY SUMMARY
# ============================================================

interval_records = []

for product, group in bootstrap.groupby(
    "Product_Group"
):

    effects = group[
        "Model_Revenue_Effect"
    ]

    actual_gaps = group[
        "Actual_Revenue_Gap"
    ]

    point = fixed_summary.loc[
        (
            fixed_summary["Product_Group"]
            == product
        )
        & (
            fixed_summary["Reference_Definition"]
            == BOOT_REFERENCE
        )
        & (
            fixed_summary["Model"]
            == BOOT_MODEL
        ),
        "Model_Revenue_Effect"
    ].iloc[0]

    low = effects.quantile(0.05)
    high = effects.quantile(0.95)

    if low > 0:

        conclusion = "Positive across central range"

    elif high < 0:

        conclusion = "Negative across central range"

    else:

        conclusion = "Range crosses zero"

    interval_records.append({
        "Product_Group": product,
        "Point_Estimate": point,
        "Bootstrap_Median": effects.median(),
        "Lower_5th_Percentile": low,
        "Upper_95th_Percentile": high,
        "Positive_Draws_Pct": (
            effects.gt(0).mean() * 100
        ),
        "Actual_Gap_Lower": (
            actual_gaps.quantile(0.05)
        ),
        "Actual_Gap_Upper": (
            actual_gaps.quantile(0.95)
        ),
        "Conclusion": conclusion
    })

intervals = pd.DataFrame(
    interval_records
)

print("\n7. BOOTSTRAP UNCERTAINTY RESULTS")

print(
    intervals.round(2).to_string(index=False)
)

# ============================================================
# 16. MODEL / REFERENCE PREDICTIVE DIAGNOSTICS
# ============================================================

accuracy = pred.groupby([
    "Product_Group",
    "Reference_Definition",
    "Model"
]).agg(
    Absolute_Error=("Absolute_Error", "sum"),
    Actual_Sales=("Sales", "sum"),
    Observations=("Row_ID", "count")
).reset_index()

accuracy["WAPE_Pct"] = (
    accuracy["Absolute_Error"]
    / accuracy["Actual_Sales"] * 100
)

# ============================================================
# 17. VISUALIZATIONS
# ============================================================

fig1 = px.bar(
    frequency,
    x="Product_Group",
    y="Promotion_Candidates",
    color="Reference",
    barmode="group",
    title="Sensitivity of Promotion Identification",
    labels={
        "Product_Group": "Product group",
        "Promotion_Candidates": "Flagged observations"
    },
    template="plotly_white"
)

fig1.write_html(
    CHART / "13d_reference_price_sensitivity.html"
)

fig2 = px.scatter(
    fixed_summary,
    x="Model_Revenue_Effect",
    y="Product_Group",
    color="Reference_Definition",
    symbol="Model",
    hover_data=["N"],
    title="Revenue Effect Across Model and Price Assumptions",
    labels={
        "Model_Revenue_Effect": "Estimated revenue effect (€)",
        "Product_Group": "Product group"
    },
    template="plotly_white"
)

fig2.add_vline(
    x=0,
    line_dash="dash"
)

fig2.write_html(
    CHART / "13e_revenue_robustness.html"
)

chart_intervals = intervals.copy()

chart_intervals["Upper_Error"] = (
    chart_intervals["Upper_95th_Percentile"]
    - chart_intervals["Bootstrap_Median"]
)

chart_intervals["Lower_Error"] = (
    chart_intervals["Bootstrap_Median"]
    - chart_intervals["Lower_5th_Percentile"]
)

chart_intervals["Product"] = (
    "Group "
    + chart_intervals["Product_Group"].astype(str)
)

fig3 = px.scatter(
    chart_intervals,
    x="Bootstrap_Median",
    y="Product",
    error_x="Upper_Error",
    error_x_minus="Lower_Error",
    title="Conditional Bootstrap Revenue Uncertainty",
    labels={
        "Bootstrap_Median": "Estimated revenue effect (€)"
    },
    template="plotly_white"
)

fig3.add_vline(
    x=0,
    line_dash="dash"
)

fig3.update_traces(
    marker=dict(size=10)
)

fig3.write_html(
    CHART / "13f_bootstrap_uncertainty.html"
)

# ============================================================
# 18. VALIDATION
# ============================================================

expected_predictions = (
    len(data)
    * len(REFERENCE_SPECS)
    * len(SPECS)
)

assert len(pred) == expected_predictions

assert not pred.duplicated([
    "Row_ID",
    "Reference_Definition",
    "Model"
]).any()

assert fixed_summary.groupby(
    "Product_Group"
)["N"].nunique().eq(1).all()

assert robustness["Specifications"].eq(
    len(REFERENCE_SPECS) * len(SPECS)
).all()

assert intervals[
    "Lower_5th_Percentile"
].le(
    intervals["Upper_95th_Percentile"]
).all()

assert np.isfinite(
    pred[
        [
            "Baseline_Demand",
            "Predicted_Actual_Price",
            "Model_Revenue_Effect"
        ]
    ]
).all().all()

print("\n8. VALIDATION")

print("PASS - Original reference prices reproduced.")
print("PASS - All model/reference combinations included.")
print("PASS - Identical fixed evaluation observations.")
print("PASS - No duplicate predictions.")
print("PASS - All predictions are finite.")
print("PASS - Bootstrap uncertainty ranges are valid.")

# ============================================================
# 19. EXPORT RESULTS
# ============================================================

frequency.to_csv(
    OUT / "reference_price_promotion_frequency.csv",
    index=False
)

pred.to_csv(
    OUT / "reference_price_candidate_predictions.csv",
    index=False
)

fixed_summary.to_csv(
    OUT / "reference_price_fixed_sample_results.csv",
    index=False
)

robustness.to_csv(
    OUT / "reference_price_robustness_summary.csv",
    index=False
)

native_summary.to_csv(
    OUT / "reference_specific_promotion_results.csv",
    index=False
)

native_robustness.to_csv(
    OUT / "reference_specific_robustness_summary.csv",
    index=False
)

accuracy.to_csv(
    OUT / "reference_price_model_accuracy.csv",
    index=False
)

bootstrap.to_csv(
    OUT / "promotion_bootstrap_draws.csv",
    index=False
)

intervals.to_csv(
    OUT / "promotion_uncertainty_summary.csv",
    index=False
)

print("\n" + "=" * 70)

print(
    "Reference price sensitivity completed successfully!"
)

print("Results saved to:", OUT)
print("Charts saved to:", CHART)

print(
    "Bootstrap ranges are conditional on the "
    "selected model, fixed reference prices and "
    "historical sample."
)

print(
    "They are not causal confidence intervals."
)

print("=" * 70)
