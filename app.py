import streamlit as st
import os

# Import page functions
from pages import dashboard_page as dashboard
from pages import data_page as data_page_module
from pages import history_page as history
from pages import home_page as home
from pages import predict_page as predict

# Main app
def main():
    st.sidebar.title("Navigation")
    menu = ["Home", "View Data", "Dashboard", "Predict", "History"]
    page = st.sidebar.selectbox("Go to", menu)

    if page == "Home":
        home.home_page()
    elif page == "View Data":
        data_page_module.data_page()
    elif page == "Dashboard":
        dashboard.dashboard_page()
    elif page == "Predict":
        predict.predict_page()
    elif page == "History":
        history.history_page()

if __name__ == "__main__":
    main()