import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Amazon Sales Dashboard", layout="wide")

DATA_PATH = "cleaned_amazon_sales.csv"
MODEL_PATH = "amount_pipeline_XGB.joblib"

STATUS_MAP = {
    "Shipped": "Shipped",
    "Shipped - Delivered to Buyer": "Delivered",
    "Shipped - Picked Up": "In_Transit",
    "Pending": "Pending",
    "Pending - Waiting for Pick Up": "Pending",
    "Shipped - Out for Delivery": "In_Transit",
    "Shipped - Rejected by Buyer": "Other",
    "Shipping": "In_Transit",
}

PREDICTION_FEATURES = [
    "Category",
    "Size",
    "SKU_freq",
    "Category_freq",
    "size_freq",
]


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)
    df["SKU_freq"] = pd.to_numeric(df["SKU_freq"], errors="coerce").fillna(0)
    df["B2B"] = df["B2B"].fillna(False).astype(bool)

    text_columns = [
        "Status",
        "Fulfilment",
        "Sales Channel ",
        "ship-service-level",
        "Category",
        "Size",
        "Courier Status",
        "ship-city",
        "ship-state",
    ]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown")

    if "month" not in df.columns:
        df["month"] = df["Date"].dt.month
    if "year" not in df.columns:
        df["year"] = df["Date"].dt.year
    if "day" not in df.columns:
        df["day"] = df["Date"].dt.day
    if "weekday" not in df.columns:
        df["weekday"] = df["Date"].dt.weekday
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    if "Category_freq" not in df.columns:
        df["Category_freq"] = df["Category"].map(df["Category"].value_counts())
    if "size_freq" not in df.columns:
        df["size_freq"] = df["Size"].map(df["Size"].value_counts())

    df["Status_grouped"] = df["Status"].map(STATUS_MAP).fillna("Other")
    df["Month Name"] = df["Date"].dt.strftime("%b")
    df["Order Count"] = 1
    return df


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def build_prediction_frame(category, size, sku_freq, category_freq, size_freq):
    row = {
        "Category": category,
        "Size": size,
        "SKU_freq": int(sku_freq),
        "Category_freq": int(category_freq),
        "size_freq": int(size_freq),
    }
    return pd.DataFrame([row], columns=PREDICTION_FEATURES)


def default_numeric_value(series: pd.Series) -> int:
    positive_values = pd.to_numeric(series, errors="coerce")
    positive_values = positive_values[positive_values > 0]
    if positive_values.empty:
        return 0
    return int(positive_values.median())


def option_frequency(df: pd.DataFrame, column: str, frequency_column: str, value: str) -> int:
    frequency = df.loc[df[column] == value, frequency_column]
    if frequency.empty:
        return 0
    return int(frequency.iloc[0])


df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

st.title("Amazon Sales Dashboard")
st.caption("Explore sales performance and predict order amount with the saved ML pipeline.")

dashboard_tab, predict_tab = st.tabs(["Dashboard", "Predict Amount"])

with dashboard_tab:
    st.sidebar.header("Filters")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    selected_dates = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = min_date
        end_date = max_date

    category_options = sorted(df["Category"].dropna().unique())
    state_options = sorted(df["ship-state"].dropna().unique())
    status_options = sorted(df["Status"].dropna().unique())
    channel_options = sorted(df["Sales Channel "].dropna().unique())

    selected_categories = st.sidebar.multiselect(
        "Category", category_options, default=category_options
    )
    selected_states = st.sidebar.multiselect(
        "State", state_options, default=state_options
    )
    selected_statuses = st.sidebar.multiselect(
        "Status", status_options, default=status_options
    )
    selected_channels = st.sidebar.multiselect(
        "Sales channel", channel_options, default=channel_options
    )

    filtered_df = df[
        (df["Date"].dt.date >= start_date)
        & (df["Date"].dt.date <= end_date)
        & (df["Category"].isin(selected_categories))
        & (df["ship-state"].isin(selected_states))
        & (df["Status"].isin(selected_statuses))
        & (df["Sales Channel "].isin(selected_channels))
    ].copy()

    if filtered_df.empty:
        st.warning("No rows match the selected filters.")
    else:
        total_sales = filtered_df["Amount"].sum()
        total_orders = int(filtered_df["Order Count"].sum())
        total_units = int(filtered_df["Qty"].sum())
        average_order_value = total_sales / total_orders if total_orders else 0

        cancelled_orders = filtered_df["Status"].str.contains(
            "cancel", case=False, na=False
        ).sum()
        cancel_rate = (cancelled_orders / total_orders * 100) if total_orders else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Sales", f"₹{total_sales:,.0f}")
        col2.metric("Orders", f"{total_orders:,}")
        col3.metric("Units Sold", f"{total_units:,}")
        col4.metric("Avg Order Value", f"₹{average_order_value:,.0f}")
        col5.metric("Cancel Rate", f"{cancel_rate:.1f}%")

        sales_trend = (
            filtered_df.groupby("Date", as_index=False)["Amount"]
            .sum()
            .sort_values("Date")
        )
        category_sales = (
            filtered_df.groupby("Category", as_index=False)["Amount"]
            .sum()
            .sort_values("Amount", ascending=False)
        )
        state_sales = (
            filtered_df.groupby("ship-state", as_index=False)["Amount"]
            .sum()
            .sort_values("Amount", ascending=False)
            .head(10)
        )
        status_count = (
            filtered_df.groupby("Status", as_index=False)["Order Count"]
            .sum()
            .sort_values("Order Count", ascending=False)
        )
        courier_sales = (
            filtered_df.groupby("Courier Status", as_index=False)["Amount"]
            .sum()
            .sort_values("Amount", ascending=False)
        )

        left, right = st.columns(2)

        with left:
            st.subheader("Sales Trend")
            fig_sales_trend = px.line(
                sales_trend,
                x="Date",
                y="Amount",
                markers=True,
                color_discrete_sequence=["#0f766e"],
            )
            fig_sales_trend.update_layout(xaxis_title="", yaxis_title="Sales")
            st.plotly_chart(fig_sales_trend, use_container_width=True)

        with right:
            st.subheader("Sales by Category")
            fig_category = px.bar(
                category_sales,
                x="Category",
                y="Amount",
                color="Category",
            )
            fig_category.update_layout(
                showlegend=False, xaxis_title="", yaxis_title="Sales"
            )
            st.plotly_chart(fig_category, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.subheader("Top 10 States by Sales")
            fig_state = px.bar(
                state_sales,
                x="Amount",
                y="ship-state",
                orientation="h",
                color="Amount",
                color_continuous_scale="Tealgrn",
            )
            fig_state.update_layout(yaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig_state, use_container_width=True)

        with right:
            st.subheader("Order Status Distribution")
            fig_status = px.pie(
                status_count,
                names="Status",
                values="Order Count",
                hole=0.45,
            )
            st.plotly_chart(fig_status, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.subheader("Courier Performance")
            fig_courier = px.bar(
                courier_sales,
                x="Courier Status",
                y="Amount",
                color="Courier Status",
            )
            fig_courier.update_layout(
                showlegend=False, xaxis_title="", yaxis_title="Sales"
            )
            st.plotly_chart(fig_courier, use_container_width=True)

        with right:
            st.subheader("Raw Data")
            st.dataframe(
                filtered_df[
                    [
                        "Date",
                        "Status",
                        "Category",
                        "Qty",
                        "Amount",
                        "ship-state",
                        "Courier Status",
                    ]
                ].sort_values("Date", ascending=False),
                use_container_width=True,
                height=420,
            )

with predict_tab:
    st.subheader("Predict Order Amount")
    st.write("Enter the fields used by the saved model to estimate `Amount`.")

    with st.form("amount_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox(
                "Category", sorted(df["Category"].dropna().unique())
            )
            size = st.selectbox("Size", sorted(df["Size"].dropna().unique()))
        with col2:
            sku_freq = st.number_input(
                "SKU Frequency",
                min_value=0,
                value=default_numeric_value(df["SKU_freq"]),
            )
            category_freq = st.number_input(
                "Category Frequency",
                min_value=0,
                value=option_frequency(df, "Category", "Category_freq", category),
            )
            size_freq = st.number_input(
                "Size Frequency",
                min_value=0,
                value=option_frequency(df, "Size", "size_freq", size),
            )
        submit_prediction = st.form_submit_button("Predict Amount")

    if submit_prediction:
        input_df = build_prediction_frame(
            category=category,
            size=size,
            sku_freq=sku_freq,
            category_freq=category_freq,
            size_freq=size_freq,
        )
        predicted_amount = float(model.predict(input_df)[0])

        st.metric("Predicted Amount", f"₹{predicted_amount:,.2f}")
        st.dataframe(input_df, use_container_width=True)
