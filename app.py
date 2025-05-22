import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os

# Load API key from environment variable
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY") or ""

def ask_together_for_filter(prompt):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "max_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.95,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def filter_dataframe_with_ai(df, user_comment):
    columns = ", ".join(df.columns)
    prompt = f"""
You are a Python data analyst.
You have a pandas dataframe called `df` with columns: {columns}

User wants to filter the data with this instruction:
\"\"\"{user_comment}\"\"\"

Write a single line of valid Python code assigning the filtered dataframe to variable `filtered_df`.

Example:
filtered_df = df[df["column"] > 1000]

Do NOT include explanations or markdown. Only output the Python code.
"""

    code = ask_together_for_filter(prompt).strip()
    # Remove any markdown code blocks if present
    code = code.replace("```python", "").replace("```", "").strip()

    if "filtered_df" not in code:
        st.error("AI did not return a valid filter code. Try simplifying your instruction.")
        return df

    st.code(code, language="python")

    try:
        local_vars = {"df": df}
        exec(code, {}, local_vars)
        filtered_df = local_vars.get("filtered_df")
        if filtered_df is None or filtered_df.empty:
            st.warning("The filter returned no rows. Please try a different filter.")
            return df
        return filtered_df
    except Exception as e:
        st.error(f"Error executing AI-generated code: {e}")
        return df

def plot_chart(df, chart_type, x_col, y_col):
    plt.clf()
    if chart_type == "Line":
        plt.plot(df[x_col], df[y_col])
    elif chart_type == "Bar":
        plt.bar(df[x_col], df[y_col])
    elif chart_type == "Scatter":
        plt.scatter(df[x_col], df[y_col])
    elif chart_type == "Histogram":
        plt.hist(df[y_col], bins=20)
    else:
        st.warning("Invalid chart type selected.")
        return
    plt.title(f"{chart_type} Chart")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    st.pyplot(plt)

st.set_page_config(page_title="CSV Visualizer with Meta Llama 3.3 70B Instruct Turbo Free")
st.title("📊 CSV Visualizer with AI Filtering")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if "df" not in st.session_state:
    st.session_state.df = None
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None

if uploaded_file:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()

    st.write("### Data Preview")
    st.dataframe(st.session_state.df.head())

    st.markdown("**Examples of filters:** `sales > 1000`, `year == 2023 and region == 'West'`")
    user_filter = st.text_area("Enter your filter instruction")

    if st.button("Apply Filter"):
        if user_filter.strip():
            filtered = filter_dataframe_with_ai(st.session_state.df, user_filter)
            st.session_state.filtered_df = filtered
        else:
            st.warning("Please enter a filter instruction.")

if st.session_state.filtered_df is not None:
    st.write("### Filtered Data Preview")
    st.dataframe(st.session_state.filtered_df.head())

    chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Scatter", "Histogram"])
    x_axis = st.selectbox("Select X-axis column", st.session_state.filtered_df.columns)
    numeric_cols = st.session_state.filtered_df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        st.warning("No numeric columns available for Y-axis.")
    else:
        y_axis = st.selectbox("Select Y-axis column", numeric_cols)

        if st.button("Generate Chart"):
            plot_chart(st.session_state.filtered_df, chart_type, x_axis, y_axis)
