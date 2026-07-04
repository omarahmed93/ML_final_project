# Amazon Sales Amount Prediction

This project is an end-to-end Machine Learning and Data Analytics application built to analyze Amazon sales data and predict order amount using a saved XGBoost regression pipeline.

The project includes data preprocessing, feature engineering, model prediction, and an interactive Streamlit dashboard for exploring sales performance and generating amount predictions.

---

## Project Objective

The main objective of this project is to predict the order `Amount` based on product and sales-related features such as:

- Product Category
- Product Size
- SKU frequency
- Category frequency
- Size frequency

The project also provides a dashboard to help analyze sales trends, order performance, courier status, and state-level sales distribution.

---

## Features

- Cleaned Amazon sales dataset
- Interactive Streamlit dashboard
- Sales trend analysis
- Sales by category
- Top states by sales
- Order status distribution
- Courier performance analysis
- Amount prediction using a trained XGBoost model
- Saved ML pipeline using Joblib

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Joblib

---

## Project Structure

```text
ML_final_project/
│
├── cleaned_amazon_sales.csv        # Cleaned dataset used in the dashboard
├── amount_pipeline_XGB.joblib      # Saved XGBoost prediction pipeline
├── streamlit_dashboard.py          # Streamlit dashboard application
├── requirements.txt                # Required Python packages
├── runtime.txt                     # Python runtime version
└── .gitignore