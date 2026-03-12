# European Bank Customer Retention Analytics

## Project Overview
This project analyzes customer behavior in a European bank dataset to identify churn drivers and improve customer retention strategies.

The project focuses on behavioral analytics rather than just demographics by examining engagement levels, product adoption, and relationship strength.

## Objectives
- Analyze customer engagement and churn patterns
- Evaluate the impact of product usage on retention
- Identify high-value customers at risk of churn
- Build a predictive model to estimate churn probability

## Dataset
The dataset contains **10,000 bank customers** with the following features:

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited (Churn indicator)

## Key Business KPIs

### Engagement Retention Ratio (ERR)
Measures churn risk between inactive and active customers.

ERR = Churn Rate of Inactive Customers / Churn Rate of Active Customers

Result: **1.88**

Inactive customers are **1.88 times more likely to churn**.

### Product Depth Index (PDI)

Average number of products held by retained customers.

Result: **1.54**

### Relationship Strength Index (RSI)

A custom metric measuring customer loyalty using:

RSI = (0.4 × Active Status) + (0.4 × Product Usage) + (0.2 × Credit Card Ownership)

## Machine Learning Model

Model Used: **Random Forest Classifier**

Model Performance:

AUC Score: **0.85**

Key churn predictors:

- Age
- Number of Products
- Geography
- Customer Activity

## Streamlit Dashboard

An interactive dashboard built using Streamlit provides:

- Customer churn analytics
- Behavioral segmentation
- Real-time churn prediction
- KPI monitoring

Run the dashboard:

## Project Structure

Dataset/
European_Bank.csv

analytics_and_modeling.py
deployment_app.py
EDA.ipynb
Final_Bank_Data_With_KPIs.csv
churn_model.pkl
research_paper.pdf


## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Author

Avinash Pandey  
B.Tech CSE (Data Science)
