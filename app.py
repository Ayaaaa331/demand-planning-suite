
from pathlib import Path
from io import BytesIO
import sys
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(ROOT / "src")
)

import planner_forecasting as planner
import planner_insights as insights

st.set_page_config(
    page_title="Hafen & Home | Demand Planning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BLUE = "#365C8D"
TEAL = "#46A094"
ORANGE = "#C77C35"
GREY = "#8492A6"

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_DESCRIPTIONS = {
    "Naive":
        "Repeats the most recent sales observation.",

    "Seasonal naive":
        "Uses sales from the corresponding week last year.",

    "Moving average 4":
        "Average sales over the previous four weeks.",

    "Moving average 13":
        "Average sales over the previous 13 weeks.",

    "Moving median 13":
        "Median sales over the previous 13 weeks.",

    "SES":
        "Exponential smoothing of the demand level.",

    "Holt linear":
        "Exponential smoothing with a linear trend.",

    "Holt damped":
        "Trend forecasting with gradually declining growth.",

    "Linear regression":
        "Projects the historical linear demand trend.",

    "Seasonal calendar regression":
        "Models trend, annual seasonality and calendar events.",

    "Log calendar regression":
        "Models proportional demand changes and calendar effects.",

    "Price Ridge regression":
        "Regularized regression with calendar and price information.",

    "Log price Ridge regression":
        "Models proportional demand changes and price relationships.",

    "Pooled log price regression":
        "Learns common price-demand relationships across regions.",

    "ARIMAX (1,0,0)":
        "Regression with price, calendar and autoregressive errors."
}

# ============================================================
# 2. WORKBOOK IMPORT
# ============================================================

@st.cache_data(show_spinner=False)
def load_workbook(content, filename):

    engine = (
        "xlrd"
        if filename.lower().endswith(".xls")
        else "openpyxl"
    )

    book = pd.ExcelFile(
        BytesIO(content),
        engine=engine
    )

    required_sheets = [
        "Sales in unit",
        "Prices in Euro"
    ]

    missing = [
        sheet for sheet in required_sheets
        if sheet not in book.sheet_names
    ]

    if missing:

        raise ValueError(
            f"Missing worksheets: {missing}"
        )

    sales = pd.read_excel(
        book,
        sheet_name="Sales in unit"
    )

    prices = pd.read_excel(
        book,
        sheet_name="Prices in Euro"
    )

    for frame in [sales, prices]:

        frame.columns = (
            frame.columns.astype(str).str.strip()
        )

        if "Week" not in frame.columns:

            raise ValueError(
                "Both worksheets require a Week column."
            )

        frame["Week"] = pd.to_datetime(
            frame["Week"],
            errors="raise"
        ).dt.normalize()

        if frame["Week"].isna().any():

            raise ValueError(
                "Missing weekly dates."
            )

        if frame["Week"].duplicated().any():

            raise ValueError(
                "Duplicate weekly dates."
            )

        if not frame["Week"].dt.day_of_week.eq(0).all():

            raise ValueError(
                "Weekly dates must be Mondays."
            )

    if set(sales.columns) != set(prices.columns):

        raise ValueError(
            "Sales and price columns do not match."
        )

    if len(sales.columns) < 2:

        raise ValueError(
            "No product-region series found."
        )

    sales_long = sales.melt(
        id_vars="Week",
        var_name="Series",
        value_name="Sales"
    )

    prices_long = prices.melt(
        id_vars="Week",
        var_name="Series",
        value_name="Price"
    )

    df = sales_long.merge(
        prices_long,
        on=["Week", "Series"],
        how="outer",
        validate="one_to_one",
        indicator=True
    )

    if not df["_merge"].eq("both").all():

        raise ValueError(
            "Sales and price dates do not match."
        )

    df = df.drop(columns="_merge")

    identifiers = df["Series"].str.extract(
        r"^(.+)\s+(\d+)$"
    )

    if identifiers.isna().any().any():

        raise ValueError(
            "Series names must follow the format "
            "'Region ProductNumber', e.g. North 1."
        )

    df["Region"] = identifiers[0]

    df["Product_Group"] = (
        identifiers[1].astype(int)
    )

    for column in ["Sales", "Price"]:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

        observed = df[column].dropna()

        if not np.isfinite(observed).all():

            raise ValueError(
                f"{column} contains non-finite values."
            )

    if df["Sales"].lt(0).any():

        raise ValueError(
            "Negative sales are not supported."
        )

    if df["Price"].dropna().le(0).any():

        raise ValueError(
            "Observed prices must be positive."
        )

    historical_dates = df.loc[
        df["Sales"].notna(),
        "Week"
    ]

    if historical_dates.empty:

        raise ValueError(
            "No historical sales found."
        )

    last_historical_week = historical_dates.max()

    df["Period"] = np.where(
        df["Week"] <= last_historical_week,
        "Historical",
        "Future"
    )

    df["Revenue"] = (
        df["Sales"] * df["Price"]
    )

    df.loc[
        df["Sales"].eq(0),
        "Revenue"
    ] = 0

    return df.sort_values(
        ["Series", "Week"]
    ).reset_index(drop=True)


# ============================================================
# 3. HISTORICAL DATA FLAGS
# ============================================================

@st.cache_data(show_spinner=False)
def enrich_history(history):

    history = history.copy()

    first_sales = history.loc[
        history["Sales"] > 0
    ].groupby(
        "Series"
    )["Week"].min()

    history["First_Sale"] = (
        history["Series"].map(first_sales)
    )

    history["Pre_Launch"] = (
        history["First_Sale"].notna()
        & (
            history["Week"]
            < history["First_Sale"]
        )
    )

    history["Active_Demand"] = (
        history["First_Sale"].notna()
        & ~history["Pre_Launch"]
        & history["Sales"].notna()
    )

    history["Potential_Availability_Issue"] = (
        history["Active_Demand"]
        & history["Sales"].eq(0)
        & history["Price"].isna()
    )

    history["Sales_Outlier"] = False

    for _, group in history.groupby("Series"):

        active = group[
            group["Active_Demand"]
        ]

        if active.empty:
            continue

        q1 = active["Sales"].quantile(0.25)
        q3 = active["Sales"].quantile(0.75)

        iqr = q3 - q1

        lower = max(
            0,
            q1 - 1.5 * iqr
        )

        upper = q3 + 1.5 * iqr

        flags = (
            active["Sales"].lt(lower)
            | active["Sales"].gt(upper)
        )

        history.loc[
            active.index,
            "Sales_Outlier"
        ] = flags

    return history


# ============================================================
# 4. CACHED FORECASTING FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def cached_recommendation(history, series):

    return planner.recommend_model(
        history,
        series
    )


@st.cache_data(show_spinner=False)
def cached_forecast(
    history,
    series,
    model,
    horizon,
    assumed_price=None
):

    return planner.forecast_series(
        history,
        series,
        model,
        horizon,
        assumed_price
    )


@st.cache_data(show_spinner=False)
def cached_historical_test(
    history,
    series,
    model
):

    return planner.historical_test(
        history,
        series,
        model,
        test_weeks=13
    )


# ============================================================
# 5. DISPLAY HELPERS
# ============================================================

def format_units(value):

    return (
        f"{value:,.0f}"
        if pd.notna(value)
        else "N/A"
    )


def format_percent(value):

    return (
        f"{value:.1f}%"
        if pd.notna(value)
        else "N/A"
    )


def format_euro(value):

    return (
        f"€{value:,.2f}"
        if pd.notna(value)
        else "N/A"
    )


def show_chart(fig):

    fig.update_layout(
        template="plotly_white",
        margin=dict(
            l=15,
            r=15,
            t=45,
            b=15
        ),
        legend_title_text=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def csv_download(
    frame,
    label,
    filename,
    key
):

    st.download_button(
        label=label,
        data=frame.to_csv(
            index=False,
            date_format="%Y-%m-%d"
        ).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key
    )


# ============================================================
# 6. SIDEBAR AND DATA LOADING
# ============================================================

st.sidebar.title("Hafen & Home")

st.sidebar.caption(
    "Demand Planning Suite · v1.0"
)

uploaded = st.sidebar.file_uploader(
    "Upload planning workbook",
    type=["xls", "xlsx"],
    help=(
        "Workbook containing the worksheets "
        "'Sales in unit' and 'Prices in Euro'."
    )
)

if uploaded is None:

    st.title("Demand Planning Suite")

    st.markdown(
        "Explore historical demand, evaluate forecasting "
        "methods and prepare future planning scenarios."
    )

    st.info(
        "Upload your sales and price workbook "
        "using the sidebar to begin."
    )

    with st.expander("Required workbook format"):

        st.markdown(
            """
            - Sales worksheet: `Sales in unit`
            - Price worksheet: `Prices in Euro`
            - Weekly date column: `Week`
            - Matching product-region columns
            - Example series: `North 1`, `South 1`
            - Weekly observations dated on Mondays
            """
        )

    st.stop()

content = uploaded.getvalue()

dataset_id = hashlib.sha256(
    content
).hexdigest()[:16]

try:

    data = load_workbook(
        content,
        uploaded.name
    )

except Exception as error:

    st.error(
        f"Unable to import workbook: {error}"
    )

    st.stop()

history = data[
    data["Period"] == "Historical"
].copy()

future_input = data[
    data["Period"] == "Future"
].copy()

history = enrich_history(history)

history_start = history["Week"].min()

history_end = history["Week"].max()

st.sidebar.success("Workbook loaded")

st.sidebar.caption(
    f"{history_start:%d %b %Y} – "
    f"{history_end:%d %b %Y}"
)

st.sidebar.caption(
    f"{history['Week'].nunique()} historical weeks"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Workspace",
    [
        "Business overview",
        "Data quality",
        "Demand history",
        "Promotion effectiveness",
        "Forecast accuracy",
        "Demand planning",
        "Portfolio forecast"
    ]
)

st.sidebar.divider()

regions = sorted(
    history["Region"].unique()
)

selected_region = st.sidebar.selectbox(
    "Region",
    regions
)

products = sorted(
    history.loc[
        history["Region"] == selected_region,
        "Product_Group"
    ].unique()
)

selected_product = st.sidebar.selectbox(
    "Product group",
    products,
    format_func=lambda x: f"Group {x}"
)

selected_history = history[
    history["Region"].eq(selected_region)
    & history["Product_Group"].eq(selected_product)
].copy()

selected_series = selected_history[
    "Series"
].iloc[0]

selected_history = selected_history.sort_values(
    "Week"
)

st.sidebar.divider()

st.sidebar.caption(
    "Forecasts represent expected sales, "
    "not final purchasing quantities."
)

# ============================================================
# 7. BUSINESS OVERVIEW
# ============================================================

if page == "Business overview":

    st.title("Business overview")

    st.caption(
        "Historical performance across all products "
        "and regions in the uploaded workbook."
    )

    total_sales = history["Sales"].sum()

    observed_revenue = history[
        "Revenue"
    ].sum(min_count=1)

    known_revenue_sales = history.loc[
        history["Revenue"].notna(),
        "Sales"
    ].sum()

    coverage = (
        known_revenue_sales / total_sales * 100
        if total_sales > 0
        else np.nan
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total units sold",
        format_units(total_sales)
    )

    c2.metric(
        "Observed revenue",
        f"€{observed_revenue:,.0f}"
        if pd.notna(observed_revenue)
        else "N/A"
    )

    c3.metric(
        "Historical weeks",
        history["Week"].nunique()
    )

    c4.metric(
        "Revenue coverage",
        format_percent(coverage)
    )

    st.caption(
        "Revenue coverage measures the share of sales "
        "units for which revenue can be calculated."
    )

    st.subheader("Sales by product and region")

    summary = history.groupby(
        ["Product_Group", "Region"]
    ).agg(
        Units=("Sales", "sum")
    ).reset_index()

    summary["Product"] = (
        "Group "
        + summary["Product_Group"].astype(str)
    )

    fig = px.bar(
        summary,
        x="Product",
        y="Units",
        color="Region",
        barmode="group",
        color_discrete_sequence=[
            BLUE, TEAL, ORANGE
        ]
    )

    show_chart(fig)

    st.subheader("Product portfolio")

    portfolio = history.groupby(
        "Product_Group"
    ).agg(
        Units=("Sales", "sum"),
        Observed_Revenue=(
            "Revenue",
            lambda x: x.sum(min_count=1)
        )
    ).reset_index()

    portfolio["Sales_Share_Pct"] = (
        portfolio["Units"] / total_sales * 100
        if total_sales > 0
        else np.nan
    )

    left, right = st.columns([1, 1])

    with left:

        fig = px.pie(
            portfolio,
            names="Product_Group",
            values="Units",
            hole=0.55
        )

        show_chart(fig)

    with right:

        st.dataframe(
            portfolio.style.format({
                "Units": "{:,.0f}",
                "Observed_Revenue": "€{:,.2f}",
                "Sales_Share_Pct": "{:.1f}%"
            }),
            hide_index=True,
            use_container_width=True
        )

    st.subheader("Total weekly sales")

    weekly = history.groupby(
        "Week"
    )["Sales"].sum(min_count=1).reset_index()

    weekly["Moving_Average"] = (
        weekly["Sales"].rolling(
            13,
            min_periods=13
        ).mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=weekly["Week"],
            y=weekly["Sales"],
            name="Actual sales",
            line=dict(color=BLUE)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=weekly["Week"],
            y=weekly["Moving_Average"],
            name="13-week moving average",
            line=dict(color=TEAL, width=3)
        )
    )

    fig.update_yaxes(
        title="Weekly units"
    )

    show_chart(fig)

    with st.expander("Highest-demand weeks"):

        st.dataframe(
            weekly.nlargest(10, "Sales"),
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# 8. DATA QUALITY
# ============================================================

elif page == "Data quality":

    st.title("Data quality")

    st.caption(
        "Review data completeness and potential "
        "anomalies before forecasting."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Demand series",
        history["Series"].nunique()
    )

    c2.metric(
        "Missing prices",
        int(history["Price"].isna().sum())
    )

    c3.metric(
        "Zero-sales weeks",
        int(history["Sales"].eq(0).sum())
    )

    c4.metric(
        "Potential sales outliers",
        int(history["Sales_Outlier"].sum())
    )

    records = []

    expected_dates = pd.date_range(
        history_start,
        history_end,
        freq="W-MON"
    )

    for series, group in history.groupby(
        "Series"
    ):

        records.append({
            "Series": series,

            "Missing_Weeks": len(
                expected_dates.difference(
                    group["Week"]
                )
            ),

            "Missing_Sales": int(
                group["Sales"].isna().sum()
            ),

            "Missing_Prices": int(
                group["Price"].isna().sum()
            ),

            "Zero_Sales": int(
                group["Sales"].eq(0).sum()
            ),

            "Before_First_Sale": int(
                group["Pre_Launch"].sum()
            ),

            "Potential_Availability_Issues": int(
                group[
                    "Potential_Availability_Issue"
                ].sum()
            ),

            "Potential_Outliers": int(
                group["Sales_Outlier"].sum()
            )
        })

    quality = pd.DataFrame(records)

    st.subheader("Quality assessment by series")

    st.dataframe(
        quality,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Data treatment")

    st.markdown(
        """
        - Missing historical prices remain missing in observed revenue.
        - Forecasting models may fill missing training prices.
        - Weeks before the first recorded sale are excluded from model fitting.
        - Zero sales within the active period are retained.
        - Potential sales outliers are flagged rather than deleted.
        """
    )

    st.caption(
        "The first recorded sale is a lifecycle proxy, "
        "not confirmation of the actual launch date."
    )

    missing_prices = history[
        history["Price"].isna()
    ][
        [
            "Week",
            "Series",
            "Sales",
            "Pre_Launch",
            "Potential_Availability_Issue"
        ]
    ]

    with st.expander("Missing-price observations"):

        st.dataframe(
            missing_prices,
            hide_index=True,
            use_container_width=True
        )

    with st.expander("Potential sales outliers"):

        st.dataframe(
            history.loc[
                history["Sales_Outlier"],
                ["Week", "Series", "Sales", "Price"]
            ],
            hide_index=True,
            use_container_width=True
        )

    try:

        planner.validate_history(history)

        st.success(
            "Historical data passed forecasting input validation."
        )

    except Exception as error:

        st.warning(
            f"Forecasting input issue: {error}"
        )

    csv_download(
        quality,
        "Download data quality report",
        "data_quality_report.csv",
        "quality_download"
    )


# ============================================================
# 9. DEMAND HISTORY
# ============================================================

elif page == "Demand history":

    st.title("Demand history")

    st.caption(
        f"{selected_region} · Product group "
        f"{selected_product}"
    )

    active = selected_history[
        selected_history["Active_Demand"]
    ].copy()

    mean = active["Sales"].mean()

    cv = (
        active["Sales"].std() / mean * 100
        if pd.notna(mean) and mean > 0
        else np.nan
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total units",
        format_units(
            selected_history["Sales"].sum()
        )
    )

    c2.metric(
        "Mean weekly demand",
        format_units(mean)
    )

    c3.metric(
        "Median weekly demand",
        format_units(
            active["Sales"].median()
        )
    )

    c4.metric(
        "Demand variability (CV)",
        format_percent(cv)
    )

    window = st.slider(
        "Moving-average window",
        min_value=4,
        max_value=26,
        value=13
    )

    active_series = (
        active.set_index("Week")["Sales"]
        .asfreq("7D")
    )

    moving_average = active_series.rolling(
        window,
        min_periods=window
    ).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=selected_history["Week"],
            y=selected_history["Sales"],
            name="Actual sales",
            mode="lines+markers",
            line=dict(color=BLUE)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=moving_average.index,
            y=moving_average,
            name=f"{window}-week moving average",
            line=dict(color=TEAL, width=3)
        )
    )

    outliers = selected_history[
        selected_history["Sales_Outlier"]
    ]

    fig.add_trace(
        go.Scatter(
            x=outliers["Week"],
            y=outliers["Sales"],
            name="Potential outliers",
            mode="markers",
            marker=dict(
                color=ORANGE,
                size=9
            )
        )
    )

    fig.update_yaxes(
        title="Units sold"
    )

    show_chart(fig)

    st.subheader("Historical prices")

    fig = px.line(
        selected_history,
        x="Week",
        y="Price",
        markers=True,
        labels={"Price": "Price (€)"}
    )

    fig.update_traces(
        line_color=ORANGE,
        connectgaps=False
    )

    show_chart(fig)

    if len(active) >= 26:

        first_13 = active["Sales"].head(13).mean()

        last_13 = active["Sales"].tail(13).mean()

        change = (
            (last_13 / first_13 - 1) * 100
            if first_13 > 0
            else np.nan
        )

        st.subheader("Demand development")

        a, b, c = st.columns(3)

        a.metric(
            "First 13 weeks — mean",
            format_units(first_13)
        )

        b.metric(
            "Latest 13 weeks — mean",
            format_units(last_13)
        )

        c.metric(
            "Change",
            format_percent(change)
        )

    st.subheader("Historical observations")

    display = selected_history[
        [
            "Week",
            "Sales",
            "Price",
            "Revenue",
            "Pre_Launch",
            "Sales_Outlier"
        ]
    ]

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True
    )

    csv_download(
        display,
        "Download selected historical data",
        "selected_demand_history.csv",
        "history_download"
    )

    insights.render_seasonality(
        history,
        selected_series
    )
# ============================================================
# 10. FORECAST MODEL RECOMMENDATION
# ============================================================

elif page in [
    "Forecast accuracy",
    "Demand planning"
]:

    try:

        clean_history = planner.validate_history(
            history
        )

    except Exception as error:

        st.error(
            f"Forecasting unavailable: {error}"
        )

        st.stop()

    group_history = clean_history[
        clean_history["Product_Group"]
        == selected_product
    ]

    available_models = planner.MODELS.copy()

    if set(group_history["Region"]) != {
        "North", "South"
    }:

        available_models.remove(
            "Pooled log price regression"
        )

    rank_key = (
        f"ranking_{dataset_id}_{selected_series}"
    )

    st.title(page)

    st.caption(
        f"{selected_region} · Product group "
        f"{selected_product}"
    )

    with st.expander(
        "Automatic model recommendation",
        expanded=rank_key not in st.session_state
    ):

        st.write(
            "Compare available forecasting models "
            "using three chronological 13-week "
            "validation periods."
        )

        st.caption(
            "The latest 13 historical weeks are "
            "reserved for separate accuracy evaluation."
        )

        if st.button(
            "Compare forecasting models",
            key=f"compare_{dataset_id}_{selected_series}"
        ):

            try:

                with st.spinner(
                    "Fitting and comparing models..."
                ):

                    ranking = cached_recommendation(
                        clean_history,
                        selected_series
                    )

                st.session_state[
                    rank_key
                ] = ranking

            except Exception as error:

                st.error(
                    f"Model comparison failed: {error}"
                )

        ranking = st.session_state.get(
            rank_key
        )

        if ranking is not None:

            valid_ranking = ranking.dropna(
                subset=["MAPE_Pct"]
            )

            if valid_ranking.empty:

                st.warning(
                    "MAPE cannot be calculated "
                    "for this evaluation sample."
                )

            else:

                winner = valid_ranking.iloc[0]

                st.success(
                    f"Recommended model: "
                    f"{winner['Model']}"
                )

                st.caption(
                    f"Development MAPE: "
                    f"{winner['MAPE_Pct']:.2f}%"
                )

                columns = [
                    "Model",
                    "MAPE_Pct",
                    "WAPE_Pct",
                    "MAD",
                    "MSE",
                    "Bias_Pct"
                ]

                st.dataframe(
                    ranking[columns].round(2),
                    hide_index=True,
                    use_container_width=True
                )

                if winner["Bias_Pct"] < -50:

                    st.warning(
                        "The recommended model materially "
                        "underestimated demand during development."
                    )

                csv_download(
                    ranking,
                    "Download model comparison",
                    "model_comparison.csv",
                    "ranking_download"
                )

    ranking = st.session_state.get(
        rank_key
    )

    recommended_model = None

    if ranking is not None:

        valid_ranking = ranking.dropna(
            subset=["MAPE_Pct"]
        )

        if not valid_ranking.empty:

            recommended_model = (
                valid_ranking.iloc[0]["Model"]
            )

    options = available_models.copy()

    if recommended_model is not None:

        options.insert(
            0,
            "Automatic recommendation"
        )

    selected_option = st.selectbox(
        "Forecasting method",
        options,
        key=(
            f"model_{dataset_id}_"
            f"{selected_series}_{page}_"
            f"{recommended_model}"
        )
    )

    model = (
        recommended_model
        if selected_option == "Automatic recommendation"
        else selected_option
    )

    st.caption(
        MODEL_DESCRIPTIONS.get(model, "")
    )

    # ========================================================
    # 11. HISTORICAL FORECAST ACCURACY
    # ========================================================

    if page == "Forecast accuracy":

        try:

            with st.spinner(
                "Evaluating historical forecasts..."
            ):

                result, metrics = (
                    cached_historical_test(
                        clean_history,
                        selected_series,
                        model
                    )
                )

        except Exception as error:

            st.error(
                f"Historical evaluation failed: {error}"
            )

            st.stop()

        st.subheader("Reserved 13-week historical test")

        st.caption(
            f"Forecast origin: "
            f"{result['Forecast_Origin'].iloc[0]:%d %b %Y}"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "MAPE",
            format_percent(
                metrics["MAPE_Pct"]
            )
        )

        c2.metric(
            "WAPE",
            format_percent(
                metrics["WAPE_Pct"]
            )
        )

        c3.metric(
            "MAD",
            format_units(
                metrics["MAD"]
            )
        )

        c4.metric(
            "Forecast bias",
            format_percent(
                metrics["Bias_Pct"]
            )
        )

        st.caption(
            "MAPE excludes observations with zero actual sales. "
            "Negative bias indicates underforecasting."
        )

        if metrics["WAPE_Pct"] > 50:

            st.warning(
                "High historical forecast error. "
                "Review the forecasts before making "
                "purchasing decisions."
            )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=result["Week"],
                y=result["Actual"],
                mode="lines+markers",
                name="Actual sales",
                line=dict(color=BLUE)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=result["Week"],
                y=result["Forecast"],
                mode="lines+markers",
                name="Historical forecast",
                line=dict(
                    color=TEAL,
                    dash="dash"
                )
            )
        )

        fig.update_yaxes(
            title="Units"
        )

        show_chart(fig)

        st.subheader(
            "Cumulative forecast error"
        )

        fig = px.line(
            result,
            x="Week",
            y="Cumulative_Error",
            markers=True
        )

        fig.add_hline(y=0)

        show_chart(fig)

        st.subheader("Detailed forecast errors")

        result["Absolute_Error"] = (
            result["Error"].abs()
        )

        result["APE_Pct"] = (
            result["Absolute_Error"]
            / result["Actual"].replace(0, np.nan)
            * 100
        )

        display = result[
            [
                "Week",
                "Actual",
                "Forecast",
                "Error",
                "Absolute_Error",
                "APE_Pct",
                "Cumulative_Error"
            ]
        ]

        st.dataframe(
            display.round(2),
            hide_index=True,
            use_container_width=True
        )

        csv_download(
            display,
            "Download historical evaluation",
            "historical_forecast_evaluation.csv",
            "historical_test_download"
        )

        st.caption(
            "Changing model selection after examining "
            "these test results makes the evaluation exploratory."
        )

    # ========================================================
    # 12. FUTURE DEMAND PLANNING
    # ========================================================

    else:

        c1, c2 = st.columns(2)

        with c1:

            horizon = st.selectbox(
                "Forecast horizon",
                [13, 26, 52],
                format_func=lambda x: f"{x} weeks"
            )

        uses_price = (
            model in planner.PRICE_MODELS
        )

        assumed_price = None

        default_price = None

        if uses_price:

            try:

                default_price = planner.default_price(
                    clean_history,
                    selected_series
                )

            except Exception as error:

                st.error(str(error))
                st.stop()

            with c2:

                custom_price = st.checkbox(
                    "Customize future price"
                )

                if custom_price:

                    assumed_price = st.number_input(
                        "Assumed selling price (€)",
                        min_value=0.01,
                        value=float(default_price),
                        step=0.10,
                        format="%.4f",
                        key=(
                            f"price_{dataset_id}_"
                            f"{selected_series}"
                        )
                    )

                else:

                    st.metric(
                        "Default assumed price",
                        format_euro(default_price)
                    )

            st.caption(
                "Default future prices use the median "
                "observed price over the preceding 13 weeks, "
                "with a historical fallback if necessary."
            )

            if assumed_price is not None:

                observed = selected_history[
                    "Price"
                ].dropna()

                if (
                    assumed_price < observed.min()
                    or assumed_price > observed.max()
                ):

                    st.warning(
                        "The selected price is outside the "
                        "observed historical price range."
                    )

        else:

            with c2:

                st.info(
                    "This forecasting method does not "
                    "use price information."
                )

        if (
            not future_input.empty
            and future_input["Price"].notna().any()
        ):

            st.warning(
                "The workbook contains future prices. "
                "This version uses a constant historical "
                "or manually specified price assumption "
                "instead of importing weekly future prices."
            )

        try:

            with st.spinner(
                "Generating demand forecast..."
            ):

                forecast = cached_forecast(
                    clean_history,
                    selected_series,
                    model,
                    horizon,
                    assumed_price
                )

        except Exception as error:

            st.error(
                f"Forecast generation failed: {error}"
            )

            st.stop()

        total_forecast = (
            forecast["Forecast_Units"].sum()
        )

        first_13 = forecast.head(13)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            f"{horizon}-week forecast",
            format_units(total_forecast)
        )

        c2.metric(
            "Average weekly units",
            format_units(
                forecast["Forecast_Units"].mean()
            )
        )

        c3.metric(
            "Next 13 weeks",
            format_units(
                first_13["Forecast_Units"].sum()
            )
        )

        if horizon > 13:

            st.warning(
                "This horizon exceeds the validated "
                "13-week forecasting horizon. Treat "
                "extended forecasts as planning scenarios."
            )

        if ranking is not None:

            development_row = ranking[
                ranking["Model"] == model
            ]

            if not development_row.empty:

                bias = development_row[
                    "Bias_Pct"
                ].iloc[0]

                if bias < -50:

                    st.warning(
                        f"Development forecast bias was "
                        f"{bias:.1f}%. Historical demand "
                        "was substantially underestimated."
                    )

        st.subheader(
            "Historical and forecast demand"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=selected_history["Week"],
                y=selected_history["Sales"],
                name="Historical sales",
                line=dict(color=BLUE)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast["Week"],
                y=forecast["Forecast"],
                name="Future forecast",
                mode="lines+markers",
                line=dict(
                    color=TEAL,
                    width=3,
                    dash="dash"
                )
            )
        )

        fig.add_vrect(
            x0=forecast["Week"].min(),
            x1=forecast["Week"].max(),
            fillcolor=TEAL,
            opacity=0.07,
            line_width=0
        )

        fig.update_yaxes(
            title="Weekly units"
        )

        show_chart(fig)

        # ----------------------------------------------------
        # Price scenario comparison
        # ----------------------------------------------------

        if uses_price:

            st.subheader(
                "Price scenario assessment"
            )

            try:

                baseline = cached_forecast(
                    clean_history,
                    selected_series,
                    model,
                    horizon,
                    None
                )

                baseline_units = baseline[
                    "Forecast_Units"
                ].sum()

                scenario_units = forecast[
                    "Forecast_Units"
                ].sum()

                scenario_price = (
                    assumed_price
                    if assumed_price is not None
                    else default_price
                )

                scenario_table = pd.DataFrame({
                    "Scenario": [
                        "Default price",
                        "Selected price"
                    ],

                    "Price_EUR": [
                        default_price,
                        scenario_price
                    ],

                    "Forecast_Units": [
                        baseline_units,
                        scenario_units
                    ]
                })

                scenario_table[
                    "Estimated_Revenue_EUR"
                ] = (
                    scenario_table["Price_EUR"]
                    * scenario_table["Forecast_Units"]
                )

                st.dataframe(
                    scenario_table.style.format({
                        "Price_EUR": "€{:,.2f}",
                        "Forecast_Units": "{:,.0f}",
                        "Estimated_Revenue_EUR": "€{:,.2f}"
                    }),
                    hide_index=True,
                    use_container_width=True
                )

                if assumed_price is not None:

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=baseline["Week"],
                            y=baseline["Forecast"],
                            name="Default price",
                            line=dict(color=BLUE)
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=forecast["Week"],
                            y=forecast["Forecast"],
                            name="Selected price",
                            line=dict(
                                color=TEAL,
                                dash="dash"
                            )
                        )
                    )

                    show_chart(fig)

            except Exception as error:

                st.warning(
                    f"Scenario comparison unavailable: {error}"
                )

            st.warning(
                "Price scenarios reflect historical "
                "statistical associations. They do not "
                "establish causal price effects or "
                "identify the optimal selling price."
            )

        # ----------------------------------------------------
        # Historical reliability indicators
        # ----------------------------------------------------

        st.subheader(
            "Historical forecasting reliability"
        )

        try:

            _, metrics = cached_historical_test(
                clean_history,
                selected_series,
                model
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Historical test MAPE",
                format_percent(
                    metrics["MAPE_Pct"]
                )
            )

            c2.metric(
                "Historical test WAPE",
                format_percent(
                    metrics["WAPE_Pct"]
                )
            )

            c3.metric(
                "Historical test bias",
                format_percent(
                    metrics["Bias_Pct"]
                )
            )

            if metrics["WAPE_Pct"] > 50:

                st.warning(
                    "High historical forecast error. "
                    "Consider alternative models and "
                    "additional commercial information."
                )

            st.caption(
                "These values measure past forecasting "
                "performance, not the realized accuracy "
                "of the future forecast or custom price scenario."
            )

        except Exception as error:

            st.info(
                f"Historical accuracy unavailable: {error}"
            )

        # ----------------------------------------------------
        # Forecast export
        # ----------------------------------------------------

        st.subheader("Forecast quantities")

        display = forecast[
            [
                "Week",
                "Series",
                "Model",
                "Forecast_Units",
                "Assumed_Price"
            ]
        ].copy()

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True
        )

        csv_download(
            display,
            "Download forecast CSV",
            f"forecast_{selected_series}_{horizon}w.csv",
            "forecast_download"
        )

        st.caption(
            "Forecasts represent sales estimates. "
            "Inventory, lead times, safety stock and "
            "supplier constraints must be considered "
            "before calculating purchasing quantities."
        )
# ============================================================
# 13. PROMOTION EFFECTIVENESS
# ============================================================

elif page == "Promotion effectiveness":

    insights.render_promotions(
        history,
        selected_series
    )


# ============================================================
# 14. PORTFOLIO FORECAST
# ============================================================

elif page == "Portfolio forecast":

    if (
        not future_input.empty
        and future_input["Price"].notna().any()
    ):

        st.warning(
            "Supplied future prices are not used. "
            "The portfolio uses historical median "
            "price assumptions."
        )

    insights.render_portfolio(
        history,
        dataset_id
    )