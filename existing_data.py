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
    pd.set_option('precision', 2)

    with st.spinner(text='Loading Data! Please wait...'):
        clean_data = load_data_vac()

    st.text("")

   st.dataframe(clean_data)
      #  plot_date(plot_df, multiselection, colors_dict, yaxistype)

  #  st.subheader(
   #     'Rapid test units for ' + date.today().strftime('%d-%m-%Y') + ' (by [@lolol20](https://twitter.com/lolol20))')

   # components.iframe("https://covidmap.cy/", height=480, scrolling=False)





@st.cache(ttl=60 * 60 * 1, allow_output_mutation=True)
def load_data_vac():
    df = pd.read_csv('https://github.com/marios096/streamlit/blob/main/data.csv?raw=true')
  #  df = data_cleaning(df.loc[df['location'] == 'Cyprus'])
   # df = df.drop_duplicates(subset=['Suburb', 'Address', 'Date', 'Price'], keep='last')
    #df = df.dropna(subset=['Price'])
   # df[["day", "month", "year"]] = df["Date"].str.split("/", expand=True)
   # df['Dates'] = df[df.columns[16:12:-1]].apply(
   #     lambda x: '-'.join(x.dropna().astype(str)),
    #    axis=1)
    #df['Price'] = df['Price'].apply(lambda x: float("{:.2f}".format(x)))
   # df = df.drop(columns=['Date', 'day', 'month', 'year'])
  #  airbnb.head()
    return df


def plot_date(df, selection, colors_dict, yaxistype):
#    # st.line_chart(df[selection],use_container_width=True)
   plot = figure(title='', plot_width=700, plot_height=450, x_axis_type="datetime", y_axis_type=yaxistype)

    for selected_column in selection:
        linecolor = colors_dict[selected_column]
       plot.line(df['Dates'], df[selected_column], legend_label=selected_column, line_width=2, alpha=0.5,
                  color=linecolor)

   plot.legend.location = "top_left"

st.bokeh_chart(plot, use_container_width=True)


def data_cleaning(df):
    for column in df:
        df[column].replace(["NaN", ":"], 0, inplace=True)
        df[column] = df[column].fillna(0)

    df['Dates'] = pd.to_datetime(df['Dates'], exact=False, dayfirst=True)

    return df
