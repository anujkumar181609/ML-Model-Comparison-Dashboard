import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocess import preprocess_data
from src.model import train_models

# Page config
st.set_page_config(page_title="ML Dashboard", layout="wide")

# Title
st.markdown("""
# 🚀 ML Model Comparison Dashboard
Compare multiple ML models on your dataset with ease
""")

# Sidebar
st.sidebar.header("⚙️ Controls")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # Tabs for better structure
    tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "🧹 Cleaned Data", "📈 Results"])

    # ------------------ TAB 1 ------------------
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # KPI cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", df.isnull().sum().sum())

        st.markdown("### Columns")
        st.write(list(df.columns))

        # 🎯 Target selection ONLY here
        target = st.selectbox("🎯 Select Target Column", df.columns)

        if target:
            st.success(f"Target Selected: {target}")


        # Preprocess once
        df_clean = preprocess_data(df)

        # ------------------ TAB 2 ------------------
        with tab2:
            st.subheader("Cleaned Data")
            st.dataframe(df_clean.head(), use_container_width=True)

        # Prepare data
        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        # Detect problem type
        if y.nunique() < 20:
            problem_type = "Classification"
        else:
            problem_type = "Regression"

        st.info(f"Detected Problem Type: {problem_type}")

        # ------------------ TAB 3 ------------------
        with tab3:
            if st.button("🚀 Run Models"):
                with st.spinner("Training models..."):
                    results = train_models(X, y, problem_type)

                results_df = pd.DataFrame(results).T

                st.subheader("Model Performance Table")
                st.dataframe(results_df, use_container_width=True)

                # Best model highlight
                best_model = results_df["score"].idxmax()
                st.success(f"🏆 Best Model: {best_model}")

                # KPI for best score
                st.metric("Best Score", round(results_df["score"].max(), 4))

                # Smaller chart
                st.subheader("Performance Comparison")
                fig, ax = plt.subplots(figsize=(5,3))
                results_df["score"].plot(kind="bar", ax=ax)
                ax.set_ylabel("Score")
                plt.xticks(rotation=30)

                st.pyplot(fig)

else:
    st.markdown("""
    ### 👈 Start Here
    Upload a CSV file from the sidebar to begin analysis.
    """)