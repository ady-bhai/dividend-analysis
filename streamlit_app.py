import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
if "Industry name" in data.columns:
    selected_industries = st.sidebar.multiselect(
        "Select Industries to Compare:",
        options=data["Industry name"].unique(),
        default=[]  # No industries are selected by default
    )
else:
    st.error("Column 'Industry name' not found in the dataset. Please check the dataset structure.")
    selected_industries = []

# Filter data based on user input
if selected_industries:
    filtered_data = data[data["Industry name"].isin(selected_industries)]
else:
    filtered_data = pd.DataFrame()  # Empty dataframe when no industry is selected

# Main visualization: Bar chart for selected metric
if not filtered_data.empty and selected_metric in filtered_data.columns:
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
    if selected_metric not in data.columns:
        st.warning(f"The selected metric '{selected_metric}' is not available in the dataset.")
    elif filtered_data.empty:
        st.warning("No industries selected. Please choose industries from the sidebar to view the visualization.")

# Additional visualization: Top 5 industries for the selected metric
if not filtered_data.empty and selected_metric in filtered_data.columns:
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
else:
    if filtered_data.empty:
        st.warning("No industries selected. Unable to generate the Top 5 Industries chart.")

# Advanced Analysis Section
st.header("Advanced Statistical Analysis")

# 1. Descriptive Statistics
st.subheader("Descriptive Statistics")
desc_stats = filtered_data.describe()
st.dataframe(desc_stats)

# Correlation Heatmap Section
st.subheader("Correlation Heatmap")

try:
    # Filter numeric columns
    numeric_data = filtered_data.select_dtypes(include=['float64', 'int64'])

    # Check if there are any numeric columns to analyze
    if not numeric_data.empty:
        # Calculate the correlation matrix
        corr_matrix = numeric_data.corr()

        # Create the heatmap
        st.markdown("This heatmap shows the correlation between different numeric metrics.")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=True)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for correlation analysis.")
except Exception as e:
    st.error(f"An error occurred while generating the correlation heatmap: {e}")

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


# 3. Regression Analysis
st.subheader("Regression Analysis")
if "Dividends" in data.columns and "Net Income" in data.columns:
    # Prepare data for regression
    X = filtered_data[["Net Income"]].dropna()
    y = filtered_data["Dividends"].dropna()
    X, y = X.align(y, join='inner', axis=0)  # Align data

    if len(X) > 1:  # Ensure enough data points for regression
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)

        # Display regression metrics
        st.markdown(f"**Regression Equation**: Dividends = {reg.coef_[0]:.2f} * Net Income + {reg.intercept_:.2f}")
        st.markdown(f"**R-squared**: {r2_score(y, y_pred):.2f}")
        st.markdown(f"**Mean Squared Error**: {mean_squared_error(y, y_pred):.2f}")

        # Plot regression
        regression_fig = px.scatter(
            x=X["Net Income"], y=y,
            labels={"x": "Net Income", "y": "Dividends"},
            title="Regression: Dividends vs Net Income"
        )
        regression_fig.add_scatter(x=X["Net Income"], y=y_pred, mode="lines", name="Regression Line")
        st.plotly_chart(regression_fig)
    else:
        st.warning("Not enough data for regression analysis.")
else:
    st.warning("Required columns for regression ('Dividends' and 'Net Income') are not available in the dataset.")

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
