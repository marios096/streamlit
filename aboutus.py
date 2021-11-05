import streamlit as st
import existing_data
import aboutus

def streamlit_app():
    st.title('🤑 Price Prediction 💰')
    st.info('This app displays the prices distributed by location and date')

    st.text("")
    st.text("")

    st.sidebar.text("")
    st.sidebar.text("")

    st.sidebar.title("🔗 Sources")
    st.sidebar.info(
        '[Data Science Class](https://elearning.cut.ac.cy/course/view.php?id=693)')

    st.sidebar.title("🛈 About")
    st.sidebar.info('Created and maintained by:' + '\r' + '[andreas christoforou](xristofo@gmail.com)')

    with st.spinner(text='Loading Data! Please wait...'):
        price_df = load_data()

    st.text("")