import pandas as pd
import streamlit as st
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    file_path = "divfcfe (1).xls"  # Ensure this file is present in the working directory
    data = pd.read_excel(
        file_path,
        sheet_name="Industry Averages",  # Specify the correct sheet
        skiprows=7  # Skip to the header row (row 8)
    )
    return data

# Preprocessing data
def preprocess_data(data):
    # Drop irrelevant columns (e.g., URLs or unrelated data)
    data = data.drop(columns=[col for col in data.columns if "http" in col], errors="ignore")
    
    # Clean and normalize column names
    data.columns = data.columns.str.strip()  # Remove leading/trailing spaces
    data.columns = data.columns.str.replace("\n", " ", regex=True)  # Replace newlines with spaces
    
    # List of corrected numeric columns
    numeric_cols = [
        "Dividends", 
        "Net Income", 
        "Payout", 
        "Dividends + Buybacks", 
        "Cash Return as % of Net Income", 
        "FCFE (before debt cash flows)", 
        "FCFE (after debt cash flows)", 
        "Net Cash Returned/FCFE (pre-debt)", 
        "Net Cash Returned/FCFE (post-debt)", 
        "Net Cash Returned/ Net Income", 
        "Cash/ Firm Value"
    ]
    
    # Process numeric columns
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(",", "").str.replace("$", ""), errors="coerce"
            )
    
    # Drop extra rows with all NaN values
    data = data.dropna(how="all")
    return data

# Load and preprocess data
data = load_data()
data = preprocess_data(data)

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
    "Cash Return as % of Net Income", "FCFE (before debt cash flows)",
    "FCFE (after debt cash flows)", "Net Cash Returned/FCFE (pre-debt)",
    "Net Cash Returned/FCFE (post-debt)", "Net Cash Returned/ Net Income", "Cash/ Firm Value"
]
selected_metric = st.sidebar.selectbox(
    "Select Metric to Analyze:",
    options=[metric for metric in available_metrics if metric in data.columns]
)

# Multiselect for industries
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
