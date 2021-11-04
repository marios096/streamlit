import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import datetime
from datetime import date


def streamlit_app():
    
    st.title('🤑 Price Prediction 💰')
    st.info('This app displays the prices distributed by location and date')

    st.text("")
    st.text("")

    st.sidebar.text("")
    st.sidebar.text("")

    st.sidebar.title("🔗 Sources")
    st.sidebar.info(
        '[Data Science Class](https://elearning.cut.ac.cy/course/view.php?id=693)' )

    st.sidebar.title("🛈 About")
    st.sidebar.info('Created and maintained by:  \n[Giorgos Savva]  \n[Marios Charalambous]  \n[Antonis Savvidis]')
    with st.spinner(text='Loading Data! Please wait...'):
        clean_data = load_data_vac()

    st.text("")

    with pd.option_context('display.precision', 2):
        st.dataframe(clean_data)
    
   # print(clean_data)


@st.cache(ttl=60 * 60 * 1, allow_output_mutation=True)
def load_data_vac():
    df = pd.read_csv('https://github.com/marios096/streamlit/blob/main/data.csv?raw=true',float_format='%.0f', index=False, header=False)
    df = df.drop_duplicates(subset=['Suburb', 'Address', 'Date', 'Price'], keep='last')
    df = df.dropna(subset=['Price'])
    df[["day", "month", "year"]] = df["Date"].str.split("/", expand=True)
    df['Dates'] = df[df.columns[16:12:-1]].apply(
        lambda x: '-'.join(x.dropna().astype(str)),
        axis=1)
    #pd.set_option('precision', 2)
 
    #df['Price'] = df['Price'].apply(lambda x: float("{:.2f}".format(x)))
    #df.round(2)
    #pd.set_option('display.float_format', '{:.2f}'.format)

    df = df.drop(columns=['Date', 'day', 'month', 'year'])
  #  airbnb.head()
    return df



