
from importlib import import_module
import calendar as cal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import planner_forecasting as planner

engine = import_module("11_advanced_forecasting")


# ============================================================
# 1. CONFIGURATION AND DESIGN
# ============================================================

BLUE = "#365C8D"
TEAL = "#46A094"
ORANGE = "#C77C35"
GREY = "#8492A6"
RED = "#C56858"

REGION_COLORS = {
    "North": BLUE,
    "South": TEAL
}

REFERENCES = {
    "8-week median": (8, 0.50),
    "13-week median": (13, 0.50),
    "26-week median": (26, 0.50),
    "13-week upper quartile": (13, 0.75)
}

PRIMARY_REF = "13-week median"


# ============================================================
# 2. DISPLAY AND EXPORT HELPERS
# ============================================================

def show_chart(fig):

    fig.update_layout(
        template="plotly_white",
        margin=dict(
            l=20,
            r=20,
            t=65,
            b=20
        ),
        font=dict(
            color="#243447",
            size=12
        ),
        title=dict(
            font=dict(size=17)
        ),
        legend_title_text="",
        hovermode="closest"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def download(frame, name, label=None):

    if label is None:

        label = (
            "Download "
            + name.replace("_", " ")
        )

    st.download_button(
        label=label,
        data=frame.to_csv(
            index=False,
            date_format="%Y-%m-%d"
        ).encode("utf-8"),
        file_name=name + ".csv",
        mime="text/csv",
        key=name
    )


def euro(value):

    if pd.isna(value):
        return "N/A"

    if value < 0:
        return f"-€{abs(value):,.0f}"

    return f"€{value:,.0f}"


def percentage(value):

    return (
        f"{value:.2f}%"
        if pd.notna(value)
        else "N/A"
    )


# ============================================================
# 3. DYNAMIC PROMOTION ANALYSIS
# ============================================================

@st.cache_data(show_spinner=False)
def promotion_analysis(history):

    h = planner.validate_history(history)

    # --------------------------------------------------------
    # Identify active product histories.
    # --------------------------------------------------------

    first = h[
        h.Sales.gt(0)
    ].groupby(
        "Series"
    ).Week.min()

    h = h[
        h.Week.ge(
            h.Series.map(first)
        )
    ].copy()

    h = h.sort_values(
        ["Series", "Week"]
    ).reset_index(drop=True)

    origin = h.Week.min()

    # --------------------------------------------------------
    # Calculate alternative reference prices.
    #
    # Only preceding prices are used.
    # --------------------------------------------------------

    for name, (window, q) in REFERENCES.items():

        h[name] = h.groupby(
            "Series"
        ).Price.transform(
            lambda s: (
                s.shift(1)
                .rolling(
                    window,
                    min_periods=6
                )
                .quantile(q)
            )
        )

    observed = (
        h.Sales.gt(0)
        & h.Price.gt(0)
    )

    original = (
        observed
        & h[PRIMARY_REF].notna()
        & (
            100 * (
                1 - h.Price / h[PRIMARY_REF]
            ) >= 10
        )
    )

    candidates = h.loc[
        original,
        [
            "Week",
            "Series",
            "Region",
            "Product_Group"
        ]
    ].copy()

    # Identical observations across all
    # reference-price definitions.

    common = (
        observed
        & h[list(REFERENCES)]
        .notna().all(axis=1)
    )

    d = h.loc[
        common
    ].copy().reset_index(drop=True)

    if d.empty:

        raise ValueError(
            "Not enough observed prices to estimate "
            "reference prices."
        )

    d["Row_ID"] = d.index

    d["Block"] = (
        (d.Week - origin).dt.days // 28
    )

    d["Original"] = (
        100 * (
            1 - d.Price / d[PRIMARY_REF]
        ) >= 10
    )

    # --------------------------------------------------------
    # Calendar features.
    # --------------------------------------------------------

    t = (
        (d.Week - origin)
        .dt.days.to_numpy() / 7
    )

    angle = (
        2 * np.pi * t / 52.1775
    )

    calendar_features = np.column_stack([
        t,

        np.sin(angle),

        np.cos(angle),

        np.array([
            engine.event_flags(w)
            for w in d.Week
        ])
    ])

    # --------------------------------------------------------
    # Candidate model specifications.
    #
    # Name, pooled, log target, nonlinear price.
    # --------------------------------------------------------

    specs = [
        (
            "Independent level",
            False,
            False,
            False
        ),

        (
            "Independent log",
            False,
            True,
            False
        )
    ]

    paired = all(
        set(g.Region) == {"North", "South"}
        and g.Series.nunique() == 2

        for _, g in h.groupby(
            "Product_Group"
        )
    )

    if paired:

        specs += [
            (
                "Pooled level",
                True,
                False,
                False
            ),

            (
                "Pooled log",
                True,
                True,
                False
            ),

            (
                "Pooled nonlinear log",
                True,
                True,
                True
            )
        ]

    primary_model = (
        "Pooled log"
        if paired
        else "Independent log"
    )

    results = []

    # --------------------------------------------------------
    # Estimate all model/reference combinations.
    # --------------------------------------------------------

    for ref in REFERENCES:

        ratio = np.log(
            d.Price / d[ref]
        ).to_numpy()

        for name, pooled, logged, nonlinear in specs:

            X = np.column_stack([
                calendar_features,
                ratio
            ])

            X0 = np.column_stack([
                calendar_features,
                np.zeros(len(d))
            ])

            if nonlinear:

                X = np.column_stack([
                    X,
                    ratio ** 2
                ])

                X0 = np.column_stack([
                    X0,
                    np.zeros(len(d))
                ])

            if pooled:

                south = (
                    d.Region.eq("South")
                    .astype(float)
                    .to_numpy()
                )

                X = np.column_stack([
                    X,
                    south
                ])

                X0 = np.column_stack([
                    X0,
                    south
                ])

            p = np.full(
                len(d),
                np.nan
            )

            b = p.copy()

            supported = np.zeros(
                len(d),
                dtype=bool
            )

            grouping = (
                "Product_Group"
                if pooled
                else "Series"
            )

            # ------------------------------------------------
            # Four-week blocked retrospective estimation.
            #
            # Pooled models exclude both regions
            # from the held-out block.
            # ------------------------------------------------

            for _, g in d.groupby(grouping):

                ids = g.index.to_numpy()

                for block in g.Block.unique():

                    tr = ids[
                        d.Block.iloc[ids].to_numpy()
                        != block
                    ]

                    te = ids[
                        d.Block.iloc[ids].to_numpy()
                        == block
                    ]

                    if len(tr) < 30:
                        continue

                    y = d.Sales.iloc[
                        tr
                    ].to_numpy(
                        dtype=float
                    )

                    target = (
                        np.log1p(y)
                        if logged
                        else y
                    )

                    fit = make_pipeline(
                        StandardScaler(),
                        Ridge(alpha=5.0)
                    ).fit(
                        X[tr],
                        target
                    )

                    a = fit.predict(
                        X[te]
                    )

                    base = fit.predict(
                        X0[te]
                    )

                    # Training-only logarithmic
                    # retransformation correction.

                    if logged:

                        smear = pd.Series(
                            np.exp(
                                target
                                - fit.predict(X[tr])
                            ),
                            index=tr
                        )

                        factors = smear.groupby(
                            d.Region.iloc[tr]
                        ).mean()

                        factor = d.Region.iloc[
                            te
                        ].map(
                            factors
                        ).to_numpy()

                        a = (
                            np.exp(a)
                            * factor - 1
                        )

                        base = (
                            np.exp(base)
                            * factor - 1
                        )

                    p[te] = np.maximum(
                        a,
                        0
                    )

                    b[te] = np.maximum(
                        base,
                        0
                    )

                    low = ratio[tr].min()

                    high = ratio[tr].max()

                    supported[te] = (
                        np.isfinite(a)
                        & np.isfinite(base)
                        & (base > 0)
                        & (ratio[te] >= low)
                        & (ratio[te] <= high)
                        & (low <= 0)
                        & (high >= 0)
                    )

            # ------------------------------------------------
            # Store model predictions.
            # ------------------------------------------------

            part = d[
                [
                    "Row_ID",
                    "Week",
                    "Series",
                    "Region",
                    "Product_Group",
                    "Sales",
                    "Price",
                    "Original"
                ]
            ].copy()

            part["Reference"] = ref

            part["Model"] = name

            part["Reference_Price"] = d[ref]

            part["Supported"] = supported

            part["Baseline"] = b

            part["Predicted"] = p

            part["Discount_Pct"] = (
                100 * (
                    1
                    - part.Price
                    / part.Reference_Price
                )
            )

            part["Required_Uplift_Pct"] = (
                100 * (
                    part.Reference_Price
                    / part.Price - 1
                )
            )

            positive_baseline = (
                part.Baseline.where(
                    part.Baseline > 0
                )
            )

            part["Actual_Uplift_Pct"] = (
                100 * (
                    part.Sales
                    / positive_baseline - 1
                )
            )

            part["Model_Uplift_Pct"] = (
                100 * (
                    part.Predicted
                    / positive_baseline - 1
                )
            )

            part["Observed_Gap"] = (
                part.Sales * part.Price
                - part.Baseline
                * part.Reference_Price
            )

            part["Model_Effect"] = (
                part.Predicted * part.Price
                - part.Baseline
                * part.Reference_Price
            )

            results.append(part)

    pred = pd.concat(
        results,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Strict common evaluation sample.
    #
    # Retain original promotion candidates supported
    # by every model and reference-price specification.
    # --------------------------------------------------------

    keep = pred.groupby(
        "Row_ID"
    ).Supported.all()

    strict = pred[
        pred.Original
        & pred.Row_ID.map(keep)
    ].copy()

    primary = strict[
        strict.Reference.eq(PRIMARY_REF)
        & strict.Model.eq(primary_model)
    ].copy()

    if primary.empty:

        raise ValueError(
            "No promotion candidates are supported "
            "by all specifications."
        )

    # --------------------------------------------------------
    # Validation.
    # --------------------------------------------------------

    assert not primary.duplicated(
        ["Series", "Week"]
    ).any()

    assert len(strict) == (
        len(primary)
        * len(REFERENCES)
        * len(specs)
    )

    assert primary.Baseline.gt(0).all()

    assert primary.Price.gt(0).all()

    assert primary.Reference_Price.gt(0).all()

    assert np.isfinite(
        primary[
            [
                "Baseline",
                "Predicted",
                "Observed_Gap",
                "Model_Effect"
            ]
        ]
    ).all().all()

    return (
        candidates,
        primary,
        strict,
        primary_model
    )


# ============================================================
# 4. PROMOTION EFFECTIVENESS PAGE
# ============================================================

def render_promotions(history, series):

    st.title("Promotion effectiveness")

    st.caption(
        "Assess whether additional sales during "
        "discounted weeks compensate for lower prices."
    )

    try:

        with st.spinner(
            "Estimating regular-price demand "
            "and evaluating promotions..."
        ):

            candidates, primary, strict, model = (
                promotion_analysis(history)
            )

    except Exception as e:

        st.warning(
            f"Promotion analysis unavailable: {e}"
        )

        return

    primary = primary.copy()

    st.caption(
        f"Primary specification: {model} regression. "
        "Reference price: preceding 13-week median. "
        "Potential promotion: at least 10% below "
        "the estimated reference price."
    )

    # --------------------------------------------------------
    # Portfolio-level commercial indicators.
    # --------------------------------------------------------

    a, b, c, d = st.columns(4)

    a.metric(
        "Potential promotions",
        len(candidates)
    )

    b.metric(
        "Supported observations",
        len(primary)
    )

    c.metric(
        "Observed revenue gap",
        euro(
            primary.Observed_Gap.sum()
        )
    )

    d.metric(
        "Model-implied revenue effect",
        euro(
            primary.Model_Effect.sum()
        )
    )

    coverage = (
        len(primary) / len(candidates) * 100
        if len(candidates) > 0
        else np.nan
    )

    st.caption(
        f"Evaluation coverage: {coverage:.2f}%. "
        "Unsupported observations are excluded "
        "from the strict evaluation, not classified "
        "as unsuccessful promotions."
    )

    primary["Break_Even_Units"] = (
        primary.Baseline
        * primary.Reference_Price
        / primary.Price
    )

    primary["Observed_Break_Even"] = (
        primary.Observed_Gap.ge(0)
    )

    primary["Model_Break_Even"] = (
        primary.Model_Effect.ge(0)
    )

    a, b = st.columns(2)

    a.metric(
        "Observed revenue break-even rate",
        percentage(
            primary.Observed_Break_Even.mean()
            * 100
        )
    )

    b.metric(
        "Model-implied break-even rate",
        percentage(
            primary.Model_Break_Even.mean()
            * 100
        )
    )

    with st.expander(
        "How should these indicators be interpreted?"
    ):

        st.markdown(
            """
            **Observed revenue gap**

            Actual revenue during discounted weeks
            minus estimated revenue at the reference
            price. This includes price effects and
            other unexplained demand differences.

            **Model-implied revenue effect**

            Predicted revenue at the observed price
            minus predicted revenue at the reference
            price, holding modelled calendar factors
            constant.

            **Revenue break-even**

            The additional demand required to
            compensate for a lower selling price.

            These estimates do not establish causal
            price effects or promotional profitability.
            Product costs and margins are unavailable.
            """
        )

    # --------------------------------------------------------
    # Aggregate commercial results.
    # --------------------------------------------------------

    totals = primary.groupby(
        "Product_Group"
    )[
        [
            "Observed_Gap",
            "Model_Effect"
        ]
    ].sum().reset_index()

    uplift = primary.groupby(
        "Product_Group"
    ).agg(
        Actual=("Sales", "sum"),

        Baseline=("Baseline", "sum"),

        Predicted=("Predicted", "sum"),

        Required=("Break_Even_Units", "sum"),

        Observed_Break_Even_Rate=(
            "Observed_Break_Even",
            "mean"
        ),

        Model_Break_Even_Rate=(
            "Model_Break_Even",
            "mean"
        )
    )

    for source, name in [
        ("Actual", "Observed_Uplift"),
        ("Predicted", "Model_Uplift"),
        ("Required", "Required_Uplift")
    ]:

        uplift[name] = (
            100 * (
                uplift[source]
                / uplift.Baseline - 1
            )
        )

    for col in [
        "Observed_Break_Even_Rate",
        "Model_Break_Even_Rate"
    ]:

        uplift[col] *= 100

    counts = primary.groupby(
        "Product_Group"
    ).size().rename(
        "Evaluated"
    )

    total_candidates = candidates.groupby(
        "Product_Group"
    ).size().rename(
        "Candidates"
    )

    # --------------------------------------------------------
    # Specification robustness.
    # --------------------------------------------------------

    spec = strict.groupby(
        [
            "Product_Group",
            "Reference",
            "Model"
        ]
    ).Model_Effect.sum().reset_index()

    ranges = spec.groupby(
        "Product_Group"
    ).Model_Effect.agg(
        Lowest="min",
        Highest="max"
    )

    summary = (
        total_candidates.to_frame()
        .join(counts)
        .join(
            totals.set_index(
                "Product_Group"
            )
        )
        .join(ranges)
    )

    summary["Evaluated"] = (
        summary.Evaluated
        .fillna(0)
        .astype(int)
    )

    summary["Direction"] = np.select(
        [
            summary.Lowest.gt(0),

            summary.Highest.lt(0),

            summary.Evaluated.eq(0)
        ],

        [
            "Consistently positive",

            "Consistently negative",

            "Not assessable"
        ],

        default="Sensitive to assumptions"
    )

    # --------------------------------------------------------
    # Commercial results by product.
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Commercial results by product"
    )

    chart = totals.melt(
        "Product_Group",
        var_name="Measure",
        value_name="EUR"
    )

    chart["Product"] = (
        "Group "
        + chart.Product_Group.astype(str)
    )

    chart["Measure"] = (
        chart["Measure"].replace({
            "Observed_Gap":
                "Observed revenue gap",

            "Model_Effect":
                "Model-implied revenue effect"
        })
    )

    fig = px.bar(
        chart,
        x="Product",
        y="EUR",
        color="Measure",
        barmode="group",
        title="Observed versus model-implied revenue effects",
        labels={
            "Product": "Product group",
            "EUR": "Revenue difference (€)"
        },
        color_discrete_map={
            "Observed revenue gap": BLUE,
            "Model-implied revenue effect": TEAL
        }
    )

    fig.add_hline(
        y=0,
        line_color=GREY
    )

    show_chart(fig)

    st.dataframe(
        summary.reset_index(),
        hide_index=True,
        use_container_width=True,
        column_config={

            "Product_Group":
                "Product group",

            "Candidates": st.column_config.NumberColumn(
                "Potential promotions",
                format="%d"
            ),

            "Evaluated": st.column_config.NumberColumn(
                "Evaluated promotions",
                format="%d"
            ),

            "Observed_Gap": st.column_config.NumberColumn(
                "Observed revenue gap (€)",
                format="%.2f"
            ),

            "Model_Effect": st.column_config.NumberColumn(
                "Model-implied effect (€)",
                format="%.2f"
            ),

            "Lowest": st.column_config.NumberColumn(
                "Lowest estimate (€)",
                format="%.2f"
            ),

            "Highest": st.column_config.NumberColumn(
                "Highest estimate (€)",
                format="%.2f"
            ),

            "Direction":
                "Specification agreement"
        }
    )

    st.caption(
        "Lowest and highest estimates represent "
        "variation across model specifications "
        "and reference-price definitions. "
        "They are not confidence intervals."
    )

    # --------------------------------------------------------
    # Model and reference-price sensitivity chart.
    # --------------------------------------------------------

    with st.expander(
        "Explore model and reference-price sensitivity"
    ):

        sensitivity = spec.copy()

        sensitivity["Product"] = (
            "Group "
            + sensitivity.Product_Group.astype(str)
        )

        fig = px.scatter(
            sensitivity,
            x="Model_Effect",
            y="Product",
            color="Reference",
            symbol="Model",
            title="Estimated revenue effects across specifications",
            labels={
                "Model_Effect":
                    "Estimated revenue effect (€)",

                "Product":
                    "Product group",

                "Reference":
                    "Reference price"
            },
            hover_data=[
                "Model"
            ]
        )

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color=GREY
        )

        fig.update_traces(
            marker=dict(size=9)
        )

        show_chart(fig)

        st.caption(
            "Each point represents one model and "
            "reference-price combination evaluated "
            "on the same supported observations."
        )

        download(
            spec,
            "promotion_specification_sensitivity",
            "Download specification sensitivity"
        )

    # --------------------------------------------------------
    # Sales uplift and revenue break-even.
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Sales uplift versus required uplift"
    )

    uplift_cols = [
        "Required_Uplift",
        "Observed_Uplift",
        "Model_Uplift"
    ]

    uplift_chart = uplift.reset_index().melt(
        "Product_Group",
        value_vars=uplift_cols,
        var_name="Measure",
        value_name="Uplift"
    )

    uplift_chart["Product"] = (
        "Group "
        + uplift_chart.Product_Group.astype(str)
    )

    uplift_chart["Measure"] = (
        uplift_chart["Measure"].replace({
            "Required_Uplift":
                "Required uplift",

            "Observed_Uplift":
                "Observed uplift",

            "Model_Uplift":
                "Model-implied uplift"
        })
    )

    fig = px.bar(
        uplift_chart,
        x="Product",
        y="Uplift",
        color="Measure",
        barmode="group",
        title="Did additional sales compensate for discounts?",
        labels={
            "Uplift":
                "Aggregate unit uplift (%)",

            "Product":
                "Product group"
        },
        color_discrete_map={
            "Required uplift": GREY,

            "Observed uplift": BLUE,

            "Model-implied uplift": TEAL
        }
    )

    fig.add_hline(
        y=0,
        line_color=GREY
    )

    show_chart(fig)

    st.caption(
        "Use euro-based revenue comparisons for "
        "aggregate break-even decisions, because "
        "selling prices differ across weeks."
    )

    # --------------------------------------------------------
    # Regional commercial comparison.
    # --------------------------------------------------------

    st.divider()

    st.subheader("Regional comparison")

    regional = primary.groupby(
        [
            "Product_Group",
            "Region"
        ]
    ).Model_Effect.sum().reset_index()

    regional["Product"] = (
        "Group "
        + regional.Product_Group.astype(str)
    )

    fig = px.bar(
        regional,
        x="Product",
        y="Model_Effect",
        color="Region",
        barmode="group",
        title="Estimated promotional revenue effect by region",
        labels={
            "Model_Effect":
                "Model-implied revenue effect (€)",

            "Product":
                "Product group"
        },
        color_discrete_map=REGION_COLORS
    )

    fig.add_hline(
        y=0,
        line_color=GREY
    )

    show_chart(fig)

    # --------------------------------------------------------
    # Selected-series promotion details.
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        f"Break-even analysis — {series}"
    )

    sel = primary[
        primary.Series.eq(series)
    ].copy()

    if sel.empty:

        st.info(
            "No supported promotion observations "
            "are available for this series."
        )

    else:

        a, b, c = st.columns(3)

        a.metric(
            "Evaluated promotions",
            len(sel)
        )

        b.metric(
            "Average discount",
            percentage(
                sel.Discount_Pct.mean()
            )
        )

        c.metric(
            "Model-implied revenue effect",
            euro(
                sel.Model_Effect.sum()
            )
        )

        # ----------------------------------------------------
        # Price and reference-price history.
        # ----------------------------------------------------

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=sel["Week"],
                y=sel["Reference_Price"],
                name="Estimated reference price",
                mode="lines+markers",
                line=dict(
                    color=BLUE,
                    dash="dash"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=sel["Week"],
                y=sel["Price"],
                name="Observed discounted price",
                mode="markers",
                marker=dict(
                    color=ORANGE,
                    size=9
                )
            )
        )

        fig.update_layout(
            title="Prices during supported promotion weeks",
            xaxis_title="Week",
            yaxis_title="Selling price (€)"
        )

        show_chart(fig)

        # ----------------------------------------------------
        # Weekly revenue break-even chart.
        # ----------------------------------------------------

        fig = px.scatter(
            sel,
            x="Required_Uplift_Pct",
            y="Actual_Uplift_Pct",
            color="Observed_Break_Even",
            hover_data=[
                "Week",
                "Price",
                "Reference_Price"
            ],
            title="Observed sales uplift versus break-even requirement",
            labels={
                "Required_Uplift_Pct":
                    "Required uplift (%)",

                "Actual_Uplift_Pct":
                    "Observed uplift (%)",

                "Observed_Break_Even":
                    "Revenue break-even"
            },
            color_discrete_map={
                True: TEAL,
                False: ORANGE
            }
        )

        limit = sel.Required_Uplift_Pct.max()

        fig.add_trace(
            go.Scatter(
                x=[0, limit],
                y=[0, limit],
                mode="lines",
                name="Revenue break-even threshold",
                line=dict(
                    color=GREY,
                    dash="dash"
                )
            )
        )

        show_chart(fig)

        st.caption(
            "Observations above the diagonal line "
            "exceeded estimated regular-price revenue. "
            "Observed uplift also includes other "
            "demand drivers."
        )

        st.subheader(
            "Detailed promotion observations"
        )

        cols = [
            "Week",
            "Sales",
            "Price",
            "Reference_Price",
            "Discount_Pct",
            "Baseline",
            "Predicted",
            "Required_Uplift_Pct",
            "Actual_Uplift_Pct",
            "Model_Uplift_Pct",
            "Observed_Gap",
            "Model_Effect"
        ]

        st.dataframe(
            sel[cols],
            hide_index=True,
            use_container_width=True,
            column_config={

                "Week": st.column_config.DateColumn(
                    "Week",
                    format="DD MMM YYYY"
                ),

                "Sales": st.column_config.NumberColumn(
                    "Actual sales",
                    format="%d"
                ),

                "Price": st.column_config.NumberColumn(
                    "Observed price (€)",
                    format="%.2f"
                ),

                "Reference_Price":
                    st.column_config.NumberColumn(
                        "Reference price (€)",
                        format="%.2f"
                    ),

                "Discount_Pct":
                    st.column_config.NumberColumn(
                        "Discount (%)",
                        format="%.2f"
                    ),

                "Baseline":
                    st.column_config.NumberColumn(
                        "Baseline demand",
                        format="%.1f"
                    ),

                "Predicted":
                    st.column_config.NumberColumn(
                        "Predicted demand",
                        format="%.1f"
                    ),

                "Required_Uplift_Pct":
                    st.column_config.NumberColumn(
                        "Required uplift (%)",
                        format="%.2f"
                    ),

                "Actual_Uplift_Pct":
                    st.column_config.NumberColumn(
                        "Observed uplift (%)",
                        format="%.2f"
                    ),

                "Model_Uplift_Pct":
                    st.column_config.NumberColumn(
                        "Model-implied uplift (%)",
                        format="%.2f"
                    ),

                "Observed_Gap":
                    st.column_config.NumberColumn(
                        "Observed revenue gap (€)",
                        format="%.2f"
                    ),

                "Model_Effect":
                    st.column_config.NumberColumn(
                        "Model-implied effect (€)",
                        format="%.2f"
                    )
            }
        )

        if len(sel) < 10:

            st.warning(
                "Fewer than ten observations are "
                "available for this series. "
                "Avoid strong commercial conclusions."
            )

    # --------------------------------------------------------
    # Export promotion analysis.
    # --------------------------------------------------------

    st.divider()

    st.subheader("Export promotion analysis")

    export_summary = summary.join(
        uplift[
            uplift_cols + [
                "Observed_Break_Even_Rate",
                "Model_Break_Even_Rate"
            ]
        ]
    ).reset_index()

    a, b = st.columns(2)

    with a:

        download(
            primary,
            "promotion_details",
            "Download detailed promotion analysis"
        )

    with b:

        download(
            export_summary,
            "promotion_summary",
            "Download promotion summary"
        )

    st.warning(
        "Promotion estimates reflect historical "
        "statistical associations, not proven causal "
        "effects. Confirmed promotion schedules, "
        "inventory, product mix, costs and margins "
        "are unavailable."
    )

    st.caption(
        "The live analysis compares alternative "
        "model and reference-price assumptions. "
        "Statistical bootstrap uncertainty is "
        "assessed separately in the project analysis."
    )


# ============================================================
# 5. SEASONALITY AND REGIONAL DEMAND
# ============================================================

def render_seasonality(history, series):

    st.divider()

    st.subheader(
        "Seasonality and regional demand"
    )

    active = history[
        history.Active_Demand
    ].copy()

    if active.empty:

        st.info(
            "No active sales history is available."
        )

        return

    active["Year"] = (
        active.Week.dt.year
    )

    active["Month"] = (
        active.Week.dt.month
    )

    monthly = active.groupby(
        [
            "Series",
            "Year",
            "Month"
        ]
    ).Sales.agg(
        [
            "mean",
            "count"
        ]
    ).reset_index()

    # --------------------------------------------------------
    # Retain complete calendar months only.
    # --------------------------------------------------------

    def expected(row):

        start = pd.Timestamp(
            year=int(row.Year),
            month=int(row.Month),
            day=1
        )

        return len(
            pd.date_range(
                start,
                start + pd.offsets.MonthEnd(0),
                freq="W-MON"
            )
        )

    monthly = monthly[
        monthly["count"].eq(
            monthly.apply(
                expected,
                axis=1
            )
        )
    ].copy()

    # --------------------------------------------------------
    # Monthly demand indices.
    # --------------------------------------------------------

    if not monthly.empty:

        means = active.groupby(
            "Series"
        ).Sales.mean()

        monthly["Index"] = (
            100
            * monthly["mean"]
            / monthly.Series.map(means)
        )

        profile = monthly.groupby(
            [
                "Series",
                "Month"
            ]
        ).Index.mean().unstack()

        profile = profile.reindex(
            columns=range(1, 13)
        )

        profile.columns = [
            cal.month_abbr[m]
            for m in profile.columns
        ]

        fig = px.imshow(
            profile,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Monthly demand seasonality across all series",
            labels={
                "color": "Demand index",
                "x": "Calendar month",
                "y": "Demand series"
            }
        )

        fig.update_layout(
            height=max(
                420,
                32 * len(profile) + 100
            )
        )

        show_chart(fig)

        st.caption(
            "Index 100 represents the historical "
            "average weekly demand of each series. "
            "An index of 150 indicates demand 50% "
            "above that average. Only complete "
            "calendar months are included."
        )

        st.caption(
            "Observed monthly patterns may reflect "
            "growth, promotions and other demand drivers, "
            "rather than recurring seasonality alone."
        )

        # ----------------------------------------------------
        # Strongest month for the selected series.
        # ----------------------------------------------------

        if series in profile.index:

            values = profile.loc[
                series
            ].dropna()

            if not values.empty:

                peak_month = values.idxmax()

                peak_index = values.max()

                a, b = st.columns(2)

                a.metric(
                    "Highest observed demand month",
                    peak_month
                )

                b.metric(
                    "Peak monthly demand index",
                    f"{peak_index:.0f}"
                )

    else:

        st.info(
            "Not enough complete calendar months "
            "to estimate monthly demand patterns."
        )

    # --------------------------------------------------------
    # Compare regions for the selected product.
    # --------------------------------------------------------

    product = history.loc[
        history.Series.eq(series),
        "Product_Group"
    ].iloc[0]

    region = active[
        active.Product_Group.eq(product)
    ].copy()

    if region.empty:

        st.info(
            "No active regional observations."
        )

        return

    st.subheader(
        f"Regional demand comparison — Group {product}"
    )

    fig = px.line(
        region.sort_values("Week"),
        x="Week",
        y="Sales",
        color="Region",
        title=(
            f"Historical regional sales — "
            f"Product group {product}"
        ),
        labels={
            "Sales": "Weekly sales units",
            "Week": "Week"
        },
        color_discrete_map=REGION_COLORS
    )

    show_chart(fig)

    paired = region.pivot(
        index="Week",
        columns="Region",
        values="Sales"
    )

    if paired.shape[1] > 1:

        correlation = paired.corr()

        st.subheader(
            "Regional demand correlation"
        )

        if (
            "North" in correlation.columns
            and "South" in correlation.columns
        ):

            value = correlation.loc[
                "North",
                "South"
            ]

            matched = paired[
                ["North", "South"]
            ].dropna()

            a, b = st.columns(2)

            a.metric(
                "North–South correlation",
                f"{value:.3f}"
                if pd.notna(value)
                else "N/A"
            )

            b.metric(
                "Matched historical weeks",
                len(matched)
            )

        st.dataframe(
            correlation.round(3),
            use_container_width=True
        )

        st.caption(
            "Correlation measures the strength of "
            "the historical linear relationship "
            "between regional sales. It does not "
            "establish a causal relationship."
        )


# ============================================================
# 6. DYNAMIC PORTFOLIO FORECASTING
# ============================================================

@st.cache_data(show_spinner=False)
def portfolio_plan(history):

    history = planner.validate_history(
        history
    )

    frames = []
    records = []
    tests = []

    # --------------------------------------------------------
    # Automatically select and refit forecasting models.
    # --------------------------------------------------------

    for series in sorted(
        history.Series.unique()
    ):

        ranking = planner.recommend_model(
            history,
            series
        ).dropna(
            subset=["MAPE_Pct"]
        )

        if ranking.empty:

            raise ValueError(
                f"No assessable forecasting model "
                f"for {series}."
            )

        winner = ranking.iloc[0]

        model = winner.Model

        # Fit the selected method using all
        # available historical observations.

        forecast = planner.forecast_series(
            history,
            series,
            model,
            horizon=52
        )

        forecast = forecast.sort_values(
            "Week"
        ).reset_index(drop=True)

        # Separate historical evaluation.

        test, m = planner.historical_test(
            history,
            series,
            model
        )

        frames.append(
            forecast
        )

        tests.append(
            test
        )

        records.append({

            "Series": series,

            "Model": model,

            "Next_13_Weeks": int(
                forecast.head(13)
                .Forecast_Units.sum()
            ),

            "Next_52_Weeks": int(
                forecast.Forecast_Units.sum()
            ),

            "Assumed_Price":
                forecast.Assumed_Price.iloc[0],

            "Development_MAPE":
                winner.MAPE_Pct,

            "Development_Bias":
                winner.Bias_Pct,

            "Test_MAPE":
                m["MAPE_Pct"],

            "Test_WAPE":
                m["WAPE_Pct"],

            "Test_Bias":
                m["Bias_Pct"]
        })

    full = pd.concat(
        frames,
        ignore_index=True
    )

    full = full.sort_values(
        ["Series", "Week"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Validate the generated portfolio.
    # --------------------------------------------------------

    assert not full.duplicated(
        ["Week", "Series"]
    ).any()

    assert len(full) == (
        52 * history.Series.nunique()
    )

    assert full.groupby(
        "Series"
    ).Week.nunique().eq(52).all()

    assert full.Forecast_Units.ge(0).all()

    assert np.isfinite(
        full.Forecast
    ).all()

    assert full.Week.min() > (
        history.Week.max()
    )

    # Portfolio-wide historical accuracy.
    #
    # Calculate from all evaluation observations,
    # rather than averaging rounded series results.

    metrics = engine.metrics(
        pd.concat(
            tests,
            ignore_index=True
        )
    )

    return (
        full,
        pd.DataFrame(records),
        metrics
    )


# ============================================================
# 7. PORTFOLIO FORECAST PAGE
# ============================================================

def render_portfolio(history, dataset_id):

    st.title(
        "Portfolio forecast and export"
    )

    st.caption(
        "Generate a consolidated demand plan "
        "for all product groups and regions."
    )

    st.write(
        "Forecasting models are selected using "
        "development MAPE and refitted using all "
        "available historical observations."
    )

    with st.expander(
        "Forecasting methodology and assumptions"
    ):

        st.markdown(
            """
            **Model selection**

            Available forecasting methods are compared
            using chronological 13-week development
            validation periods.

            **Historical evaluation**

            The latest 13 historical weeks are reserved
            for evaluating the selected models.

            **Future prices**

            Price-aware models use a constant selling
            price equal to the median observed price
            over the preceding 13 weeks, with a
            historical fallback if necessary.

            **Forecast horizon**

            The primary planning horizon is 13 weeks.
            The extended 52-week horizon is intended
            for indicative longer-term planning.

            **Forecast interpretation**

            The forecasts represent expected weekly
            sales, not final purchasing quantities.
            """
        )

    key = "portfolio_" + dataset_id

    if st.button(
        "Generate complete portfolio forecast",
        key="generate_" + dataset_id,
        type="primary"
    ):

        try:

            with st.spinner(
                "Comparing forecasting methods "
                "and generating the complete demand plan..."
            ):

                st.session_state[key] = (
                    portfolio_plan(history)
                )

        except Exception as e:

            st.error(
                f"Portfolio forecast unavailable: {e}"
            )

    if key not in st.session_state:

        st.info(
            "Generate the portfolio forecast to "
            "obtain selected forecasting models, "
            "planning quantities and downloadable "
            "CSV files for all demand series."
        )

        return

    full, summary, metrics = (
        st.session_state[key]
    )

    full = full.copy()

    summary = summary.copy()

    short = full[
        full.Week.isin(
            sorted(
                full.Week.unique()
            )[:13]
        )
    ].copy()

    # --------------------------------------------------------
    # Portfolio KPIs.
    # --------------------------------------------------------

    a, b, c, d = st.columns(4)

    a.metric(
        "13-week forecast units",
        f"{short.Forecast_Units.sum():,}"
    )

    b.metric(
        "52-week forecast units",
        f"{full.Forecast_Units.sum():,}"
    )

    c.metric(
        "Historical MAPE",
        percentage(
            metrics["MAPE_Pct"]
        )
    )

    d.metric(
        "Historical WAPE",
        percentage(
            metrics["WAPE_Pct"]
        )
    )

    st.caption(
        "Historical accuracy refers to the reserved "
        "13-week historical test. It does not represent "
        "measured accuracy of future forecasts."
    )

    # --------------------------------------------------------
    # Selected models and forecast quantities.
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Selected models and planning quantities"
    )

    # Display formatting only.
    # Original numerical values are preserved
    # in the exported CSV.

    display = summary.copy()

    display["Assumed_Price"] = (
        display["Assumed_Price"].apply(
            lambda value:
                f"€{value:,.2f}"
                if pd.notna(value)
                else "Not used"
        )
    )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={

            "Series":
                "Demand series",

            "Model":
                "Selected model",

            "Next_13_Weeks":
                st.column_config.NumberColumn(
                    "13-week units",
                    format="%d"
                ),

            "Next_52_Weeks":
                st.column_config.NumberColumn(
                    "52-week units",
                    format="%d"
                ),

            "Assumed_Price":
                "Assumed price",

            "Development_MAPE":
                st.column_config.NumberColumn(
                    "Validation MAPE (%)",
                    format="%.2f"
                ),

            "Development_Bias":
                st.column_config.NumberColumn(
                    "Validation bias (%)",
                    format="%.2f"
                ),

            "Test_MAPE":
                st.column_config.NumberColumn(
                    "Test MAPE (%)",
                    format="%.2f"
                ),

            "Test_WAPE":
                st.column_config.NumberColumn(
                    "Test WAPE (%)",
                    format="%.2f"
                ),

            "Test_Bias":
                st.column_config.NumberColumn(
                    "Test bias (%)",
                    format="%.2f"
                )
        }
    )

    st.caption(
        "'Not used' means that the selected "
        "forecasting method does not require "
        "selling price information."
    )

    # --------------------------------------------------------
    # Flag high-risk demand series.
    # --------------------------------------------------------

    risk = summary[
        summary.Test_WAPE.gt(50)
        | summary.Test_MAPE.gt(100)
        | summary.Test_Bias.lt(-25)
        | summary.Development_Bias.lt(-50)
    ]

    if not risk.empty:

        names = ", ".join(
            risk.Series.tolist()
        )

        st.warning(
            f"Additional planning review recommended "
            f"for: {names}. These series exhibit "
            "high historical errors or substantial "
            "underforecasting."
        )

        st.caption(
            "These warnings use descriptive screening "
            "thresholds rather than formal statistical "
            "confidence limits."
        )

    # --------------------------------------------------------
    # Forecast by product and region.
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "13-week demand outlook"
    )

    by_product = short.groupby(
        [
            "Product_Group",
            "Region"
        ]
    ).Forecast_Units.sum().reset_index()

    by_product["Product"] = (
        "Group "
        + by_product.Product_Group.astype(str)
    )

    fig = px.bar(
        by_product,
        x="Product",
        y="Forecast_Units",
        color="Region",
        barmode="group",
        title="Forecast demand by product and region",
        labels={
            "Forecast_Units":
                "Forecast units",

            "Product":
                "Product group"
        },
        color_discrete_map=REGION_COLORS
    )

    show_chart(fig)

    # --------------------------------------------------------
    # Portfolio demand composition.
    # --------------------------------------------------------

    product_totals = short.groupby(
        "Product_Group"
    ).Forecast_Units.sum().reset_index()

    product_totals["Product"] = (
        "Group "
        + product_totals.Product_Group.astype(str)
    )

    left, right = st.columns(2)

    with left:

        fig = px.pie(
            product_totals,
            names="Product",
            values="Forecast_Units",
            hole=0.55,
            title="Forecast demand composition",
            color_discrete_sequence=[
                BLUE,
                TEAL,
                ORANGE,
                GREY,
                "#7797BD"
            ]
        )

        show_chart(fig)

    with right:

        regional_totals = short.groupby(
            "Region"
        ).Forecast_Units.sum().reset_index()

        regional_totals[
            "Average_Weekly_Units"
        ] = (
            regional_totals.Forecast_Units / 13
        )

        regional_totals[
            "Demand_Share_Pct"
        ] = (
            100
            * regional_totals.Forecast_Units
            / regional_totals.Forecast_Units.sum()
            if regional_totals.Forecast_Units.sum() > 0
            else np.nan
        )

        st.markdown(
            "**Regional planning totals**"
        )

        st.dataframe(
            regional_totals,
            hide_index=True,
            use_container_width=True,
            column_config={

                "Region":
                    "Region",

                "Forecast_Units":
                    st.column_config.NumberColumn(
                        "13-week units",
                        format="%d"
                    ),

                "Average_Weekly_Units":
                    st.column_config.NumberColumn(
                        "Weekly average",
                        format="%.0f"
                    ),

                "Demand_Share_Pct":
                    st.column_config.NumberColumn(
                        "Demand share (%)",
                        format="%.2f"
                    )
            }
        )

    # --------------------------------------------------------
    # Historical versus future total weekly demand.
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Extended portfolio outlook"
    )

    historical = history.groupby(
        "Week"
    ).Sales.sum(
        min_count=1
    ).reset_index()

    future = full.groupby(
        "Week"
    ).Forecast_Units.sum().reset_index()

    first_13 = future.head(13)

    remaining = future.iloc[12:]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=historical.Week,
            y=historical.Sales,
            name="Historical sales",
            mode="lines",
            line=dict(
                color=BLUE
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=first_13.Week,
            y=first_13.Forecast_Units,
            name="13-week forecast",
            mode="lines",
            line=dict(
                color=TEAL,
                width=3
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=remaining.Week,
            y=remaining.Forecast_Units,
            name="Extended planning scenario",
            mode="lines",
            line=dict(
                color=ORANGE,
                dash="dash"
            )
        )
    )

    fig.update_layout(
        title="Total historical and forecast weekly demand",
        xaxis_title="Week",
        yaxis_title="Total weekly units"
    )

    fig.add_vrect(
        x0=future.Week.min(),
        x1=future.Week.max(),
        fillcolor=TEAL,
        opacity=0.05,
        line_width=0
    )

    show_chart(fig)

    st.caption(
        "The 52-week forecast extends beyond the "
        "validated 13-week forecasting horizon. "
        "Longer-term trend and seasonality projections "
        "should therefore be treated cautiously."
    )

    # --------------------------------------------------------
    # Forecast exports.
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Export planning forecasts"
    )

    st.write(
        "Download forecast quantities for all "
        "demand series in the original wide "
        "workbook structure."
    )

    wide_13 = short.pivot(
        index="Week",
        columns="Series",
        values="Forecast_Units"
    ).sort_index().reset_index()

    wide_13.columns.name = None

    wide_52 = full.pivot(
        index="Week",
        columns="Series",
        values="Forecast_Units"
    ).sort_index().reset_index()

    wide_52.columns.name = None

    # Confirm export completeness.

    assert len(wide_13) == 13

    assert len(wide_52) == 52

    assert wide_13.notna().all().all()

    assert wide_52.notna().all().all()

    assert (
        wide_13.drop(
            columns="Week"
        ).to_numpy().sum()
        == short.Forecast_Units.sum()
    )

    assert (
        wide_52.drop(
            columns="Week"
        ).to_numpy().sum()
        == full.Forecast_Units.sum()
    )

    with st.expander(
        "Preview the 13-week submission forecast"
    ):

        st.dataframe(
            wide_13,
            hide_index=True,
            use_container_width=True,
            column_config={

                "Week": st.column_config.DateColumn(
                    "Week",
                    format="DD MMM YYYY"
                )
            }
        )

    a, b, c = st.columns(3)

    with a:

        download(
            wide_13,
            "portfolio_forecast_13_weeks",
            "Download 13-week forecast"
        )

    with b:

        download(
            wide_52,
            "portfolio_forecast_52_weeks",
            "Download 52-week forecast"
        )

    with c:

        download(
            summary,
            "portfolio_forecast_assumptions",
            "Download forecast assumptions"
        )

    st.warning(
        "Expected sales are not purchase orders. "
        "Review historical forecast errors, inventory "
        "levels, lead times, safety stock and supplier "
        "constraints before negotiating quantities."
    )
