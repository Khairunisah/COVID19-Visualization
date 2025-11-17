# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

st.set_page_config(layout="wide", page_title="Malaysia COVID-19 Cases Visualization")

@st.cache_data
def load_data(path):
    # read raw CSV (don't parse dates here so we can show original strings if parsing fails)
    return pd.read_csv(path)

file_path = "COVID-19 Malaysia Dataset.csv"
df_raw = load_data(file_path)

# Try parsing dates robustly: prefer day-first (e.g. 13/3/2020), infer formats, coerce errors
df = df_raw.copy()
df["Date_parsed"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    infer_datetime_format=True,
    errors="coerce"
)

# If there are parsing failures, show a small warning and sample
n_fail = df["Date_parsed"].isna().sum()
if n_fail > 0:
    st.warning(f"{n_fail} row(s) had unparseable dates. Showing examples below.")
    st.dataframe(df.loc[df["Date_parsed"].isna(), ["Date"]].drop_duplicates().head(20))

# After parsing and dropping NaT rows
df = df.dropna(subset=["Date_parsed"]).copy()

# Overwrite the original Date safely (no duplicate column names)
df["Date"] = df["Date_parsed"]
df.drop(columns=["Date_parsed"], inplace=True)

# Now Date is datetime dtype (if Date_parsed was datetime)
df = df.sort_values("Date").reset_index(drop=True)

st.title("Malaysia COVID-19 Cases Visualization")

# Sidebar selection
metric = st.sidebar.selectbox("Select a Metric to Display", [
    "Daily New Cases",
    "Active Cases",
    "Cumulative Total Cases",
    "Daily New Deaths"
])

chart_type = st.sidebar.selectbox("Select Chart Type", [
    "Line Chart",
    "Bar Chart",
    "Area Chart",
    "Histogram",
    "Box Plot",
    "Dual Axis Chart (Cases & Deaths)",
    "Heatmap"
])

# Mapping (use .get to avoid KeyError if your CSV column names are slightly different)
metric_mapping = {
    "Daily New Cases": "Daily_New_Cases",
    "Active Cases": "Active_Cases",
    "Cumulative Total Cases": "Cumulative_Total_Cases",
    "Daily New Deaths": "Daily_New_Death",
}
selected_column = metric_mapping.get(metric)

# If selected column missing, show available columns and stop
if selected_column not in df.columns:
    st.error(
        f"Column '{selected_column}' not found in data. Available columns: {', '.join(df.columns)}"
    )
    st.stop()

# Visualization logic
if chart_type == "Line Chart":
    fig = px.line(df, x="Date", y=selected_column, title=f"Malaysia - {metric} Over Time")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Bar Chart":
    fig = px.bar(df, x="Date", y=selected_column, title=f"Malaysia - {metric} (Bar Chart)")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Area Chart":
    fig = px.area(df, x="Date", y=selected_column, title=f"Malaysia - {metric} (Area Chart)")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Histogram":
    fig = px.histogram(df, x=selected_column, title=f"Distribution of {metric}")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Box Plot":
    fig = px.box(df, y=selected_column, title=f"Malaysia - {metric} (Box Plot)")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Dual Axis Chart (Cases & Deaths)":
    # Make sure both columns exist
    if ("Daily_New_Cases" not in df.columns) or ("Daily_New_Death" not in df.columns):
        st.error("Dual axis requires 'Daily_New_Cases' and 'Daily_New_Death' columns.")
    else:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()

        # Optionally downsample for plotting markers (avoid huge marker clutter)
        # Here we plot the full series as lines (no markers) for performance
        ax1.plot(df["Date"], df["Daily_New_Cases"], label="Daily New Cases")
        ax2.plot(df["Date"], df["Daily_New_Death"], label="Daily New Death", linestyle="dashed")

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Daily New Cases")
        ax2.set_ylabel("Daily New Deaths")

        # combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        fig.autofmt_xdate()
        st.pyplot(fig)

elif chart_type == "Heatmap":
    # Create a pivot: rows = Month-Year, cols = day-of-month, values = sum/mean of metric
    df["MonthYear"] = df["Date"].dt.strftime("%Y-%m")
    df["Day"] = df["Date"].dt.day
    pivot_df = df.pivot_table(index="MonthYear", columns="Day", values=selected_column, aggfunc="sum", fill_value=0)

    # sort rows chronologically (MonthYear strings sort okay with YYYY-MM)
    pivot_df = pivot_df.sort_index()

    fig, ax = plt.subplots(figsize=(14, max(6, 0.4 * len(pivot_df))))
    sns.heatmap(pivot_df, cmap="coolwarm", linewidths=0.3, ax=ax)
    ax.set_title(f"Heatmap of {metric} Over Time")
    ax.set_xlabel("Day of Month")
    ax.set_ylabel("Month-Year")
    st.pyplot(fig)

# Optional: show small data preview and dtypes for debugging (toggle)
with st.expander("Data preview & types"):
    st.dataframe(df.head(20))
    st.write(df.dtypes)
