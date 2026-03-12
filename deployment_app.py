
# 1. Import Required Libraries

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# 2. Load Dataset and Trained Model

df = pd.read_csv("Final_Bank_Data_With_KPIs.csv")
model = joblib.load("churn_model.pkl")


# 3. Streamlit Page Configuration

st.set_page_config(
    page_title="Bank Retention Hub",
    layout="wide"
)

st.title("European Bank Retention Intelligence")

st.markdown(
"""
This dashboard analyzes **customer churn behavior** using  
Behavioral Segmentation and the **Relationship Strength Index (RSI)**.
"""
)


# 4. KPI Metrics Section

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Churn Rate",
        value="20.37%",
        delta="High Risk",
        delta_color="inverse"
    )

with col2:
    st.metric(
        label="Inactivity Risk (ERR)",
        value="1.88x"
    )

with col3:
    avg_rsi = df["RSI"].mean()

    st.metric(
        label="Average Relationship Strength (RSI)",
        value=f"{avg_rsi:.2f}"
    )


# 5. Analytics Visualization Section

left_col, right_col = st.columns(2)


# Sunburst Chart (Behavior Analysis)
with left_col:

    st.subheader("Churn by Geography and Customer Segment")

    fig = px.sunburst(
        df,
        path=["Geography", "Behavior_Profile"],
        values="Exited",
        title="Customer Churn Segmentation"
    )

    st.plotly_chart(fig, use_container_width=True)


# 6. Churn Prediction Tool

with right_col:

    st.subheader("Predict At-Risk Premium Customers")

    #  User Inputs
    age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=40
    )

    balance = st.number_input(
        "Account Balance ($)",
        min_value=0,
        max_value=300000,
        value=50000
    )

    products = st.slider(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=1
    )

    active_member = st.selectbox(
        "Is Active Member?",
        [0, 1]
    )

    geography = st.selectbox(
        "Customer Geography",
        ["France", "Germany", "Spain"]
    )

    # 7. Prepare Input for Model Prediction
    
    # Calculate RSI score
    rsi_value = (
        0.4 * active_member +
        0.4 * (products / 4) +
        0.2 * 1
    )

    # Geography Encoding
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    # Model input (must match training features)
    input_data = [[
        650,            # Credit Score (default)
        1,              # Gender encoded
        age,
        5,              # Tenure
        balance,
        products,
        1,              # Has credit card
        active_member,
        100000,         # Estimated Salary
        rsi_value,
        geo_germany,
        geo_spain
    ]]


    # 8. Predict Churn Risk
    
    if st.button("Calculate Churn Risk"):

        probability = model.predict_proba(input_data)[0][1]

        if probability > 0.5:

            st.error(
                f"High Risk Customer: {probability:.1%} probability of churn."
            )

        else:

            st.success(
                f"Loyal Customer: {probability:.1%} churn risk."
            )