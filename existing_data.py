import streamlit as st
import pandas as pd
pd.set_option('precision', 2)
import numpy as np
import streamlit.components.v1 as components
import datetime
from datetime import date
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error as MSE
from xgboost import XGBRegressor
# import xgboost as xg

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
    st.sidebar.info('Created and maintained by:' + '\r' + '[george savva](georgesavva@windowslive.com)'+ ', '+'[george savva](georgesavva@windowslive.com)'
                    +', '+ '[george savva](georgesavva@windowslive.com)')

    with st.spinner(text='Loading Data! Please wait...'):
        price_df = load_data()
        pPrice = predict_price(price_df)

    st.text("")

#    features = ['daily new cases', 'daily deaths', 'Hospitalised Cases', 'Cases In ICUs', 'total_daily tests performed',
 #               'Positively Rate']
  #  colors_dict = {'daily new cases': '#1f77b4', 'daily deaths': '#2ca02c', 'Hospitalised Cases': '#9467bd',
   #                'Cases In ICUs': '#e377c2', 'total_daily tests performed': '#bcbd22', 'Positively Rate': '#bc7722'}

    #features = ["new_cases","new_deaths","icu_patients","hosp_patients","new_tests","people_vaccinated","people_fully_vaccinated"]
    #col1, col2, col3, col4 = st.beta_columns(4)
    #col1, col2, col3 = st.beta_columns(3)

   # with col1:
     #   st.warning('Confirmed cases: ' + str(int(cyprus_df['total cases'].iloc[-1])))

   # with col2:
    #    st.success('Total tests: ' + str(int(cyprus_df['total tests'].iloc[-1])))

   # with col3:
    #    st.error('Deaths: ' + str(int(cyprus_df['total deaths'].iloc[-1])))

   # with col4:
     #   st.info('Population Fully Vaccinated: ' + str(
     #       '{0:.2f}'.format(int(cyprus_vac_df['people_fully_vaccinated'].max(skipna=True)) * 100 / 875899)) + "%")

#    with col1:
 #       st.subheader("Dates")
  #      from_date = st.date_input("From Date:", datetime.date(2020, 9, 1))
   #     to_date = st.date_input("To Date:", datetime.date.today())
    #    filtered_df = cyprus_vac_df[cyprus_vac_df["Dates"].isin(pd.date_range(from_date, to_date))]

    #with col2:
     #   st.subheader("Options")
      #  if st.checkbox('Logarithmic scale'):
     #       yaxistype = "log"
    #    else:
    #        yaxistype = "linear"

       # if st.checkbox('5 Days Moving Average'):
      #      plot_df = filtered_df.rolling(5).sum()
    #    else:
     #       plot_df = filtered_df

  #  with col3:
       # st.subheader("Features")
      #  multiselection = st.multiselect("", features, default=features)

        #plot_df['Dates'] = filtered_df["Dates"]

   # if len(multiselection) > 0:
     #   with st.beta_expander("Raw data", expanded=False):
       # st.dataframe(plot_df[["Dates"]])
    st.title("Make a Radioo Button")
    st.title("Make a Radioo Button")

    page_names = ['Checkbox', 'Button']

    page = st.radio('Navigation', page_names, index=1)
    st.write("**The variable 'page' returns:**", page)

    if page == 'Checkbox':
        st.subheader('Welcome to the Checkbox page!')
        st.write("Nice to see you! :wave:")
    else:
        st.subheader("Welcome to the Button page!")
        st.write(":thumbsup:")

    st.dataframe(price_df)
    #st.bar_chart(pPrice)
    #print(pPrice)
    #st.write(pPrice)
      #  plot_date(plot_df, multiselection, colors_dict, yaxistype)

  #  st.subheader(
   #     'Rapid test units for ' + date.today().strftime('%d-%m-%Y') + ' (by [@lolol20](https://twitter.com/lolol20))')

   # components.iframe("https://covidmap.cy/", height=480, scrolling=False)





@st.cache(ttl=60 * 60 * 1, allow_output_mutation=True)
def load_data():
    df = pd.read_csv('https://github.com/marios096/streamlit/blob/main/data.csv?raw=true')
    #  df = data_cleaning(df.loc[df['location'] == 'Cyprus'])
    df = df.drop_duplicates(subset=['Suburb', 'Address', 'Date', 'Price'], keep='last')
    df = df.dropna(subset=['Price'])
    df[["day", "month", "year"]] = df["Date"].str.split("/", expand=True)
    df['Dates'] = df[df.columns[16:12:-1]].apply(
        lambda x: '-'.join(x.dropna().astype(str)),
        axis=1)
    df['Price'] = df['Price'].astype('int')

    df = df.drop(columns=['Date', 'day', 'month', 'year','CouncilArea','Postcode'])
  #  airbnb.head()
    return df

def predict_price(df):

    X = df.iloc[:,[0,2,3,5,7,9,10]].values
    y = df.iloc[:, 4].values

    X[:, 6] = pd.DatetimeIndex(X[:, 6]).year

    le_X_0 = LabelEncoder()
    le_X_2 = LabelEncoder()
    le_X_3 = LabelEncoder()
    le_X_4 = LabelEncoder()
    #le_X_6 = LabelEncoder()

    X[:, 0] = le_X_0.fit_transform(X[:, 0])
    X[:, 2] = le_X_2.fit_transform(X[:, 2])
    X[:, 3] = le_X_3.fit_transform(X[:, 3])
    X[:, 4] = le_X_4.fit_transform(X[:, 4])
   # X[:, 6] = le_X_6.fit_transform(X[:, 6])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = XGBRegressor()
    regressor.fit(X_train, y_train)
    Y_pred_train = regressor.predict(X_train)
    y_pred = regressor.predict(X_test)
    st.write("Accuracy attained on Training Set = ", rmsle(Y_pred_train, y_train))
    st.write("Accuracy attained on Test Set = ", rmsle(y_pred, y_test))

    # train_dmatrix = xg.DMatrix(data=X_train, label=y_train)
    # test_dmatrix = xg.DMatrix(data=X_test, label=y_test)
    # param = {"booster": "gblinear", "objective": "reg:linear"}
    # xgb_r = xg.train(params=param, dtrain=train_dmatrix, num_boost_round=10)
    # pred = xgb_r.predict(test_dmatrix)
    # rmse = np.sqrt(MSE(y_test, pred))
    # print("RMSE : % f" % (rmse))

    return X

def rmsle(y_pred,y_test) :
    error = np.square(np.log10(y_pred +1) - np.log10(y_test +1)).mean() ** 0.5
    Acc = 1 - error
    return Acc

def plot_date(df, selection, colors_dict, yaxistype):
    # st.line_chart(df[selection],use_container_width=True)
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


