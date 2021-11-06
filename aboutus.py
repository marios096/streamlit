import streamlit as st
import existing_data
import aboutus

def streamlit_app():
    st.title('🤑 Price Prediction 💰')
    st.info('This app displays the prices distributed by location and date')

    st.text("")
    st.text("")

    with st.spinner(text='Loading Data! Please wait...'):
        price_df = load_data()

    st.text("")