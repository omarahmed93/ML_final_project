import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Amazon Sales Dashboard", layout="wide")

st.title("Amazon Sales Dashboard")
st.caption("Sales performance, category analysis, shipping, and order insights")

df = pd.read_csv("cleaned_amazon_sales.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(0)

total_sales = df["Amount"].sum()
total_orders = len(df)
total_qty = df["Qty"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"₹{total_sales:,.0f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Total Quantity", f"{total_qty:,.0f}")

sales_trend = df.groupby("Date", as_index=False)["Amount"].sum()
category_sales = df.groupby("Category", as_index=False)["Amount"].sum()
state_sales = (
    df.groupby("ship-state", as_index=False)["Amount"].sum()
    .sort_values("Amount", ascending=False)
    .head(10)
)
size_qty = df.groupby("Size", as_index=False)["Qty"].sum()
status_count = df.groupby("Status", as_index=False).size().rename(columns={"size": "Count"})
courier_count = df.groupby("Courier Status", as_index=False).size().rename(columns={"size": "Count"})

fig1 = px.line(sales_trend, x="Date", y="Amount", title="Sales Trend")
fig2 = px.bar(category_sales, x="Category", y="Amount", title="Sales by Category")
fig3 = px.pie(courier_count, names="Courier Status", values="Count", title="Courier Status Distribution")
fig4 = px.pie(status_count, names="Status", values="Count", title="Order Status Distribution")
fig5 = px.bar(state_sales, x="ship-state", y="Amount", title="Top 10 States by Sales")
fig6 = px.bar(size_qty, x="Size", y="Qty", title="Quantity by Size")

st.subheader("Sales Overview")
row1_col1, row1_col2 = st.columns(2)
row1_col1.plotly_chart(fig1, use_container_width=True)
row1_col2.plotly_chart(fig2, use_container_width=True)

st.subheader("Operations Overview")
row2_col1, row2_col2 = st.columns(2)
row2_col1.plotly_chart(fig3, use_container_width=True)
row2_col2.plotly_chart(fig4, use_container_width=True)

st.subheader("Regional and Product Insights")
row3_col1, row3_col2 = st.columns(2)
row3_col1.plotly_chart(fig5, use_container_width=True)
row3_col2.plotly_chart(fig6, use_container_width=True)
