# Import necessary libraries
import pandas as pd
import streamlit as st
import plotly.express as px

# Load data
@st.cache
def load_data():
    # Replace with your file path if necessary
    file_path = "divfcfe.xls"  # Adjust to your file
    data = pd.read_excel(file_path, sheet_name=0)
    return data

# Load data
data = load_data()

# Clean and preprocess data
def preprocess_data(data):
    data.columns = data.columns.str.strip()  # Remove extra spaces in column names
    numeric_cols = [
        "Dividends", "Net Income", "Payout", "Dividends + Buybacks", 
        "Cash Return as % of Income", "FCFE (before debt cash flows)",
        "FCFE (after debt cash flows)", "Net Cash Returns as % of FCFE", 
        "Net Cash Returns as % of Net Income", "Cash/Firm Value"
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col].str.replace(",", "").str.replace("$", ""), errors="coerce")
    return data

data = preprocess_data(data)

# App title and description
st.title("Industry-Wise Dividend Analysis")
st.markdown("""
This app provides an interactive analysis of dividend metrics across industries. Use the filters and charts below to explore insights on dividend payouts, buybacks, and cash returns.
""")

# Sidebar for user input
st.sidebar.header("User Input Filters")
selected_metric = st.sidebar.selectbox(
    "Select Metric to Analyze:",
    [
        "Dividends", "Net Income", "Payout", "Dividends + Buybacks",
        "Cash Return as % of Income", "FCFE (before debt cash flows)",
        "FCFE (after debt cash flows)", "Net Cash Returns as % of FCFE",
        "Net Cash Returns as % of Net Income", "Cash/Firm Value"
    ]
)
selected_industries = st.sidebar.multiselect(
    "Select Industries to Compare:",
    options=data["Industry name"].unique(),
    default=data["Industry name"].unique()
)

# Filter data based on user input
filtered_data = data[data["Industry name"].isin(selected_industries)]

# Main visualization: Bar chart for selected metric
st.subheader(f"Bar Chart: {selected_metric}")
bar_chart = px.bar(
    filtered_data,
    x="Industry name",
    y=selected_metric,
    title=f"{selected_metric} by Industry",
    labels={"Industry name": "Industry", selected_metric: selected_metric},
    color="Industry name",
)
bar_chart.update_layout(showlegend=False, xaxis_title="Industry", yaxis_title=selected_metric)
st.plotly_chart(bar_chart)

# Additional visualization: Top 5 industries for the selected metric
st.subheader(f"Top 5 Industries by {selected_metric}")
top_5_data = filtered_data.nlargest(5, selected_metric)
top_5_chart = px.bar(
    top_5_data,
    x="Industry name",
    y=selected_metric,
    title=f"Top 5 Industries by {selected_metric}",
    labels={"Industry name": "Industry", selected_metric: selected_metric},
    color="Industry name",
)
top_5_chart.update_layout(showlegend=False, xaxis_title="Industry", yaxis_title=selected_metric)
st.plotly_chart(top_5_chart)

# Additional visualization: Scatter plot of Dividends vs Net Income
st.subheader("Scatter Plot: Dividends vs Net Income")
scatter_chart = px.scatter(
    filtered_data,
    x="Net Income",
    y="Dividends",
    size="Dividends + Buybacks",
    color="Industry name",
    title="Dividends vs Net Income (Bubble Size: Dividends + Buybacks)",
    labels={"Net Income": "Net Income", "Dividends": "Dividends"},
    hover_data=["Industry name"],
)
scatter_chart.update_layout(xaxis_title="Net Income", yaxis_title="Dividends")
st.plotly_chart(scatter_chart)

# Insights and summary
st.subheader("Summary Insights")
if selected_metric == "Payout":
    st.markdown("""
    **Payout Ratio Insights:**
    - The Payout Ratio measures how much of the net income is distributed to shareholders.
    - Industries with a higher Payout Ratio might be more focused on returning cash to shareholders, but could have limited reinvestment.
    """)
elif selected_metric == "FCFE (after debt cash flows)":
    st.markdown("""
    **Free Cash Flow to Equity (FCFE):**
    - FCFE after debt cash flows shows the sustainable cash flow available for distribution to shareholders.
    - Industries with high FCFE are generally in a strong financial position.
    """)
elif selected_metric == "Cash/Firm Value":
    st.markdown("""
    **Cash Efficiency:**
    - Cash/Firm Value is a measure of how much cash is held relative to the overall firm value.
    - Higher values may indicate a conservative approach to cash management.
    """)

# Add a data table for user exploration
st.subheader("Explore the Data")
st.dataframe(filtered_data)

# Download button for the filtered dataset
st.sidebar.subheader("Download Filtered Data")
@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

csv_data = convert_df(filtered_data)
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=csv_data,
    file_name="filtered_data.csv",
    mime="text/csv",
)
