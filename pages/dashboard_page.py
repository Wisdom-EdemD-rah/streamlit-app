import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dashboard_page():
    st.header("Data Dashboard")
    st.write("Interactive dashboard for exploring and analyzing the dataset.")

    # Load data from CSV
    data=pd.read_csv('data/final_test_set_churn.csv')

    # Preprocess data
    data = preprocess_data(data)

    # EDA Dashboard
    st.write("### Exploratory Data Analysis (EDA) Dashboard")

    # Correlation Matrix
    st.write("#### Correlation Matrix:")
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    corr = data[numerical_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot()

    # Distribution of Numerical Features
    st.write("#### Distribution of Numerical Features:")
    for feature in numerical_features:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        st.pyplot()

    # Distribution of Categorical Features
    st.write("#### Distribution of Categorical Features:")
    categorical_features = data.select_dtypes(include=['object', 'category']).columns
    for feature in categorical_features:
        plt.figure(figsize=(8, 6))
        sns.countplot(data[feature])
        plt.title(f'Distribution of {feature}')
        plt.xticks(rotation=45)
        st.pyplot()

    # KPIs Dashboard
    st.write("### Key Performance Indicators (KPIs) Dashboard")

    # Summary Statistics
    st.write("#### Summary Statistics:")
    st.write(data.describe())

    # Churn Rate (if 'Churn' column exists)
    if 'Churn' in data.columns:
        st.write("#### Churn Rate:")
        churn_rate = data['Churn'].mean() * 100
        st.write(f"Churn Rate: {churn_rate:.2f}%")

    # Average Monthly Charges (if 'MonthlyCharges' column exists)
    if 'MonthlyCharges' in data.columns:
        st.write("#### Average Monthly Charges:")
        avg_monthly_charges = data['MonthlyCharges'].mean()
        st.write(f"Average Monthly Charges: ${avg_monthly_charges:.2f}")

    # Average Tenure (if 'tenure' column exists)
    if 'tenure' in data.columns:
        st.write("#### Average Tenure:")
        avg_tenure = data['tenure'].mean()
        st.write(f"Average Tenure: {avg_tenure:.2f} months")

def preprocess_data(data):
    # Encode categorical features
    categorical_features = ['SeniorCitizen', 'Partner', 'Dependents', 'InternetService', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                            'Contract', 'PaperlessBilling', 'PaymentMethod']
    for feature in categorical_features:
        data[feature] = data[feature].astype('category').cat.codes

    # Convert 'TotalCharges' to numeric, handling non-numeric values
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # Fill NaN values in 'TotalCharges' with the mean of the column
    data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

    return data

if __name__ == "__main__":
    dashboard_page()
