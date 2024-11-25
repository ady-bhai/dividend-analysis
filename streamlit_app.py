import pandas as pd
import streamlit as st
import plotly.express as px

# Load data
@st.cache_data
def load_data():
    # Ensure the file path matches your dataset
    file_path = "divfcfe (1).xls"  # Replace with your actual file name
    data = pd.read_excel(file_path, engine="xlrd")
    return data

# Preprocessing data
def preprocess_data(data):
    # List of numeric columns to preprocess
    numeric_cols = [
        "Dividends", "Net Income", "Payout", "Dividends + Buybacks", 
        "Cash Return as % of Income", "FCFE (before debt cash flows)", 
        "FCFE (after debt cash flows)", "Net Cash Returns as % of FCFE", 
        "Net Cash Returns as % of Net Income", "Cash/ Firm Value"
    ]
    
    for col in numeric_cols:
        if col in data.columns:
            # Remove commas and dollar signs from numeric columns
            data[col] = pd.to_numeric(data[col].astype(str).str.replace(",", "").str.replace("$", ""), errors="coerce")
        else:
            st.warning(f"Column '{col}' not found in the dataset and will be skipped.")
    return data

# Load and preprocess data
data = load_data()
data = preprocess_data(data)

# Debugging: Display dataset columns
st.write("Dataset Columns:", data.columns.tolist())

# App title and description
st.title("Industry-Wise Dividend Analysis")
st.markdown("""
This app provides an interactive analysis of dividend metrics across industries. Use the filters and charts below to explore insights on dividend payouts, buybacks, and cash returns.
""")

# Sidebar for user input
st.sidebar.header("User Input Filters")
# Dropdown for metric selection
available_metrics = [
    "Dividends", "Net Income", "Payout", "Dividends + Buybacks",
    "Cash Return as % of Income", "FCFE (before debt cash flows)",
    "FCFE (after debt cash flows)", "Net Cash Returns as % of FCFE",
    "Net Cash Returns as % of Net Income", "Cash/ Firm Value"
]
selected_metric = st.sidebar.selectbox(
    "Select Metric to Analyze:",
    options=[metric for metric in available_metrics if metric in data.columns]
)

# Multiselect for industries
if "Industry name" in data.columns:
    selected_industries = st.sidebar.multiselect(
        "Select Industries to Compare:",
        options=data["Industry name"].unique(),
        default=data["Industry name"].unique()
    )
else:
    st.error("Column 'Industry name' not found in the dataset. Please check the dataset structure.")
    selected_industries = []

# Filter data based on user input
filtered_data = data[data["Industry name"].isin(selected_industries)] if "Industry name" in data.columns else data

# Main visualization: Bar chart for selected metric
if selected_metric in data.columns:
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
else:
    st.error(f"The selected metric '{selected_metric}' is not available in the dataset.")

# Additional visualization: Top 5 industries for the selected metric
if selected_metric in data.columns:
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
if "Dividends" in data.columns and "Net Income" in data.columns:
    st.subheader("Scatter Plot: Dividends vs Net Income")
    scatter_chart = px.scatter(
        filtered_data,
        x="Net Income",
        y="Dividends",
        size="Dividends + Buybacks" if "Dividends + Buybacks" in data.columns else None,
        color="Industry name",
        title="Dividends vs Net Income (Bubble Size: Dividends + Buybacks)",
        labels={"Net Income": "Net Income", "Dividends": "Dividends"},
        hover_data=["Industry name"],
    )
    scatter_chart.update_layout(xaxis_title="Net Income", yaxis_title="Dividends")
    st.plotly_chart(scatter_chart)
else:
    st.warning("Required columns for scatter plot ('Dividends' and 'Net Income') are not available in the dataset.")

# Add a data table for user exploration
st.subheader("Explore the Data")
st.dataframe(filtered_data)

# Download button for the filtered dataset
st.sidebar.subheader("Download Filtered Data")
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

if not filtered_data.empty:
    csv_data = convert_df(filtered_data)
    st.sidebar.download_button(
        label="Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_data.csv",
        mime="text/csv",
    )
