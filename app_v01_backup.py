
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hafen & Home | Demand Planning",
    page_icon="📊",
    layout="wide"
)

st.title("Hafen & Home")
st.subheader("Demand Planning Suite")

st.caption(
    "Version 0.1 | Historical demand exploration"
)

st.write(
    "Explore historical sales and prices to support "
    "demand planning and purchasing decisions."
)

# ============================================================
# 2. DATA IMPORT AND VALIDATION
# ============================================================

def load_workbook(uploaded_file):

    extension = Path(uploaded_file.name).suffix.lower()

    engine = (
        "xlrd" if extension == ".xls"
        else "openpyxl"
    )

    workbook = pd.ExcelFile(
        uploaded_file,
        engine=engine
    )

    required_sheets = {
        "Sales in unit",
        "Prices in Euro"
    }

    if not required_sheets.issubset(
        workbook.sheet_names
    ):
        raise ValueError(
            "The workbook must contain the sheets "
            "'Sales in unit' and 'Prices in Euro'."
        )

    sales = pd.read_excel(
        workbook,
        sheet_name="Sales in unit"
    )

    prices = pd.read_excel(
        workbook,
        sheet_name="Prices in Euro"
    )

    sales.columns = (
        sales.columns.astype(str).str.strip()
    )

    prices.columns = (
        prices.columns.astype(str).str.strip()
    )

    if "Week" not in sales.columns:
        raise ValueError(
            "The sales sheet has no Week column."
        )

    if "Week" not in prices.columns:
        raise ValueError(
            "The price sheet has no Week column."
        )

    if set(sales.columns) != set(prices.columns):
        raise ValueError(
            "Sales and price series do not match."
        )

    sales["Week"] = pd.to_datetime(
        sales["Week"],
        errors="raise"
    )

    prices["Week"] = pd.to_datetime(
        prices["Week"],
        errors="raise"
    )

    if (
        sales["Week"].isna().any()
        or prices["Week"].isna().any()
    ):
        raise ValueError(
            "Missing dates detected."
        )

    if (
        sales["Week"].duplicated().any()
        or prices["Week"].duplicated().any()
    ):
        raise ValueError(
            "Duplicate weeks detected."
        )

    sales_long = sales.melt(
        id_vars=["Week"],
        var_name="Series",
        value_name="Sales"
    )

    prices_long = prices.melt(
        id_vars=["Week"],
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
            "Some sales and price records "
            "could not be matched."
        )

    df = df.drop(columns="_merge")

    # Extract region and product identifier
    # from the series name.

    series_parts = df["Series"].str.extract(
        r"^(.+)\s+(\d+)$"
    )

    if series_parts.isna().any().any():
        raise ValueError(
            "Series names must follow the format "
            "'Region ProductNumber', e.g. North 1."
        )

    df["Region"] = series_parts[0]
    df["Product_Group"] = series_parts[1].astype(int)

    df["Sales"] = pd.to_numeric(
        df["Sales"],
        errors="raise"
    )

    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="raise"
    )

    if df["Sales"].lt(0).any():
        raise ValueError(
            "Negative sales values detected."
        )

    if df["Price"].le(0).any():
        raise ValueError(
            "Non-positive prices detected."
        )

    if not df["Sales"].notna().any():
        raise ValueError(
            "No historical sales were found."
        )

    last_historical_week = df.loc[
        df["Sales"].notna(), "Week"
    ].max()

    df["Period"] = np.where(
        df["Week"] <= last_historical_week,
        "Historical",
        "Forecast"
    )

    df["Revenue"] = df["Sales"] * df["Price"]

    df.loc[
        df["Sales"] == 0, "Revenue"
    ] = 0

    df = df.sort_values(
        ["Series", "Week"]
    ).reset_index(drop=True)

    return df


# ============================================================
# 3. FILE UPLOAD
# ============================================================

st.sidebar.header("Data source")

uploaded_file = st.sidebar.file_uploader(
    "Upload sales and price workbook",
    type=["xls", "xlsx"]
)

if uploaded_file is None:

    st.info(
        "Upload your Excel workbook using "
        "the sidebar to begin."
    )

    st.markdown(
        """
        **Expected workbook structure**

        - Sheet 1: Sales in unit
        - Sheet 2: Prices in Euro
        - One Week column in each sheet
        - Matching product-region columns
        """
    )

    st.stop()

try:

    df = load_workbook(uploaded_file)

except Exception as error:

    st.error(
        f"Unable to import the workbook: {error}"
    )

    st.stop()

historical = df[
    df["Period"] == "Historical"
].copy()

future = df[
    df["Period"] == "Forecast"
].copy()

st.sidebar.success(
    "Workbook imported successfully."
)

# ============================================================
# 4. DATA QUALITY FLAGS
# ============================================================

first_sales = (
    historical.loc[historical["Sales"] > 0]
    .groupby("Series")["Week"]
    .min()
)

historical["First_Positive_Sale"] = (
    historical["Series"].map(first_sales)
)

historical["Pre_Launch"] = (
    historical["Week"] <
    historical["First_Positive_Sale"]
)

historical["Potential_Availability_Issue"] = (
    ~historical["Pre_Launch"]
    & historical["First_Positive_Sale"].notna()
    & historical["Sales"].eq(0)
    & historical["Price"].isna()
)

historical["Active_Demand"] = (
    ~historical["Pre_Launch"]
    & historical["First_Positive_Sale"].notna()
    & historical["Sales"].notna()
)

# ============================================================
# 5. APPLICATION FILTERS
# ============================================================

st.sidebar.divider()

st.sidebar.header("Planning filters")

available_regions = sorted(
    historical["Region"].unique()
)

selected_region = st.sidebar.selectbox(
    "Region",
    available_regions
)

available_products = sorted(
    historical.loc[
        historical["Region"] == selected_region,
        "Product_Group"
    ].unique()
)

selected_product = st.sidebar.selectbox(
    "Product group",
    available_products,
    format_func=lambda x: f"Group {x}"
)

selected = historical[
    (historical["Region"] == selected_region)
    & (
        historical["Product_Group"]
        == selected_product
    )
].copy()

selected = selected.sort_values("Week")

active = selected[
    selected["Active_Demand"]
].copy()

# ============================================================
# 6. APPLICATION NAVIGATION
# ============================================================

overview_tab, quality_tab, history_tab = st.tabs(
    [
        "Business overview",
        "Data quality",
        "Demand history"
    ]
)

# ============================================================
# 7. BUSINESS OVERVIEW
# ============================================================

with overview_tab:

    st.header("Historical business performance")

    total_sales = historical["Sales"].sum()

    revenue = historical["Revenue"].sum(
        min_count=1
    )

    unknown_revenue_units = historical.loc[
        historical["Revenue"].isna(),
        "Sales"
    ].sum()

    coverage = (
        (1 - unknown_revenue_units / total_sales)
        * 100
        if total_sales > 0
        else np.nan
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total units sold",
        f"{total_sales:,.0f}"
    )

    col2.metric(
        "Observed revenue",
        f"€{revenue:,.0f}"
        if pd.notna(revenue)
        else "N/A"
    )

    col3.metric(
        "Historical weeks",
        historical["Week"].nunique()
    )

    col4.metric(
        "Revenue coverage",
        f"{coverage:.2f}%"
        if pd.notna(coverage)
        else "N/A"
    )

    st.caption(
        "Overview metrics include all products and regions. "
        "Observed revenue excludes positive sales "
        "with missing prices."
    )

    sales_summary = historical.groupby(
        ["Region", "Product_Group"]
    )["Sales"].sum().reset_index()

    sales_summary["Product"] = (
        "Group "
        + sales_summary["Product_Group"].astype(str)
    )

    fig = px.bar(
        sales_summary,
        x="Product",
        y="Sales",
        color="Region",
        barmode="group",
        title="Historical sales by product and region",
        template="plotly_white",
        labels={
            "Sales": "Units sold",
            "Product": "Product group"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Product performance")

    product_summary = historical.groupby(
        "Product_Group"
    ).agg(
        Total_Sales=("Sales", "sum"),
        Observed_Revenue=(
            "Revenue",
            lambda x: x.sum(min_count=1)
        )
    ).reset_index()

    product_summary["Sales_Share_Pct"] = (
        product_summary["Total_Sales"]
        / total_sales * 100
        if total_sales > 0
        else np.nan
    )

    st.dataframe(
        product_summary.style.format({
            "Total_Sales": "{:,.0f}",
            "Observed_Revenue": "€{:,.2f}",
            "Sales_Share_Pct": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# 8. DATA QUALITY
# ============================================================

with quality_tab:

    st.header("Data quality assessment")

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Demand series",
        historical["Series"].nunique()
    )

    q2.metric(
        "Missing historical prices",
        int(historical["Price"].isna().sum())
    )

    q3.metric(
        "Zero sales observations",
        int(historical["Sales"].eq(0).sum())
    )

    q4.metric(
        "Missing historical sales",
        int(historical["Sales"].isna().sum())
    )

    expected_weeks = pd.date_range(
        historical["Week"].min(),
        historical["Week"].max(),
        freq="7D"
    )

    quality_records = []

    for series, group in historical.groupby("Series"):

        missing_weeks = expected_weeks.difference(
            pd.DatetimeIndex(group["Week"])
        )

        quality_records.append({
            "Series": series,
            "Weeks": group["Week"].nunique(),
            "Missing_Weeks": len(missing_weeks),
            "Missing_Sales": group["Sales"].isna().sum(),
            "Missing_Prices": group["Price"].isna().sum(),
            "Zero_Sales": group["Sales"].eq(0).sum(),
            "Before_First_Sale": group["Pre_Launch"].sum(),
            "Possible_Availability_Issues":
                group["Potential_Availability_Issue"].sum()
        })

    quality_summary = pd.DataFrame(
        quality_records
    )

    st.dataframe(
        quality_summary,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Missing price observations")

    missing_prices = historical.loc[
        historical["Price"].isna(),
        [
            "Week",
            "Series",
            "Sales",
            "Pre_Launch",
            "Potential_Availability_Issue"
        ]
    ]

    st.dataframe(
        missing_prices,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Missing prices and zero sales are preserved. "
        "Periods before the first recorded sale "
        "are treated as possible pre-launch periods."
    )

# ============================================================
# 9. HISTORICAL DEMAND EXPLORATION
# ============================================================

with history_tab:

    st.header(
        f"{selected_region} | "
        f"Product group {selected_product}"
    )

    if active.empty:

        st.warning(
            "No active demand history is available "
            "for the selected series."
        )

    else:

        mean_sales = active["Sales"].mean()

        cv = (
            active["Sales"].std()
            / mean_sales * 100
            if mean_sales > 0
            else np.nan
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total units sold",
            f"{selected['Sales'].sum():,.0f}"
        )

        c2.metric(
            "Average weekly demand",
            f"{mean_sales:,.1f}"
        )

        c3.metric(
            "Median weekly demand",
            f"{active['Sales'].median():,.1f}"
        )

        c4.metric(
            "Demand variability (CV)",
            f"{cv:.1f}%"
            if pd.notna(cv)
            else "N/A"
        )

        moving_average_window = st.slider(
            "Moving average window (weeks)",
            min_value=4,
            max_value=26,
            value=13
        )

        # Retain the weekly calendar so missing
        # observations do not shift the rolling window.

        active_history = selected.loc[
            selected["Week"] >= active["Week"].min()
        ].copy()

        active_history = (
            active_history.set_index("Week")
            .asfreq("7D")
            .reset_index()
        )

        active_history["Moving_Average"] = (
            active_history["Sales"]
            .rolling(
                moving_average_window,
                min_periods=moving_average_window
            )
            .mean()
        )

        fig_sales = go.Figure()

        fig_sales.add_trace(
            go.Scatter(
                x=selected["Week"],
                y=selected["Sales"],
                name="Actual sales",
                mode="lines",
                line=dict(color="#365C8D")
            )
        )

        fig_sales.add_trace(
            go.Scatter(
                x=active_history["Week"],
                y=active_history["Moving_Average"],
                name=(
                    f"{moving_average_window}-week "
                    "moving average"
                ),
                mode="lines",
                line=dict(
                    color="#C77C35",
                    width=3
                )
            )
        )

        fig_sales.update_layout(
            title="Historical weekly demand",
            xaxis_title="Week",
            yaxis_title="Units sold",
            template="plotly_white"
        )

        st.plotly_chart(
            fig_sales,
            use_container_width=True
        )

        fig_price = px.line(
            selected,
            x="Week",
            y="Price",
            title="Historical selling prices",
            labels={
                "Price": "Price (€)"
            },
            template="plotly_white"
        )

        fig_price.update_traces(
            line_color="#46A094",
            connectgaps=False
        )

        st.plotly_chart(
            fig_price,
            use_container_width=True
        )

        st.caption(
            "Missing prices appear as gaps. "
            "Demand statistics exclude probable "
            "pre-launch observations."
        )

    st.subheader("Historical observations")

    st.dataframe(
        selected[
            [
                "Week",
                "Sales",
                "Price",
                "Revenue",
                "Pre_Launch"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# 10. DATASET INFORMATION
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    f"Historical period: "
    f"{historical['Week'].min().date()} to "
    f"{historical['Week'].max().date()}"
)

st.sidebar.caption(
    f"Future weeks in workbook: "
    f"{future['Week'].nunique()}"
)

st.sidebar.caption(
    "Hafen & Home | Demand Planning Suite v0.1"
)
