import streamlit as st
import pandas as pd

def data_page():
    st.header("Data Overview")
    # data = pd.read_excel('data/Telco-churn-last-2000.xlsx')
    data=pd.read_csv('data/final_test_set_churn.csv')
    st.write(data)

    if st.checkbox("Show Numeric Features"):
        st.write(data.select_dtypes(include='number'))

    if st.checkbox("Show Categorical Features"):
        st.write(data.select_dtypes(include='object'))