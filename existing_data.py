from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import datetime
import xgboost as xg
from datetime import date
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance

matplotlib.style.use('ggplot')
import altair as alt
from datetime import datetime
from bokeh.plotting import figure, output_file, show
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import EMPTY_LAYOUT, MISSING_RENDERERS

silence(EMPTY_LAYOUT, True)
from bokeh.models import ColumnDataSource, ranges, LabelSet
from pandas.core.frame import DataFrame
from bokeh.models.glyphs import Text
from sklearn.tree import DecisionTreeRegressor

def streamlit_app():
    st.title('Welcome to our project')
    st.info('This project is about predicting prices based on some information that we have on our dataset')

    st.text("")
    st.text("")

    st.sidebar.text("")
    st.sidebar.text("")

    st.sidebar.title("🔗 Sources")
    st.sidebar.info(
        '[Data Science Class](https://elearning.cut.ac.cy/course/view.php?id=693)')

    st.sidebar.title("🛈 About")
    st.sidebar.info(
        'Created and maintained by:' + '\r' + '[george savva](georgesavva@windowslive.com)' + ', ' + '[george savva](georgesavva@windowslive.com)'
        + ', ' + '[george savva](georgesavva@windowslive.com)')

    st.text("")

    # creating 3 columns
    col1, col2, col3 = st.columns(3)
    # load data
    df = pd.read_csv('https://github.com/marios096/streamlit/blob/main/data.csv?raw=true')

    with col1:
        # we choose a specific suburb to predict the price of it
        st.subheader("Choosing Suburb")
        # we wanted a list with the suburbs and the first choice to be 'All'
        suburbs = df['Suburb']
        suburbs.loc[-1] = 'All'  # adding a row
        suburbs.index = suburbs.index + 1  # shifting index

        suburbs.sort_index(inplace=True)
        suburbs = suburbs.drop_duplicates()  # we dropped duplicated suburbs

        # display list
        option = st.selectbox("Choose suburb to predict price", list(suburbs.items()), 0, format_func=lambda o: o[1])

    # second col
    with col2:

        st.subheader("Handling empty Prices")
        # choose how to deal with empty cell of prices

        status = st.selectbox(
            'How would you like handle empty cells of Prices?',
            ('Delete Rows', 'Median for Price', 'Mean for Price', 'Mode for Price'))

        if (status == 'Delete Rows'):
            price_df = load_data(df, 1)
        if (status == 'Median for Price'):
            price_df = load_data(df, 2)
        if (status == 'Mean for Price'):
            price_df = load_data(df, 3)
        if (status == 'Mode for Price'):
            price_df = load_data(df, 4)
    # third col-> the different machine learning algorithms that we used
    with col3:
        st.subheader("Choosing ML method")

        status = st.selectbox(
            'Select a method to perform',
            ('XGBOOST', 'Desicion Tree Regressor', 'Knnclassification'))

    if (status == 'XGBOOST'):
        source = predict_price_for_graph(price_df, option[1])
    if (status == 'Desicion Tree Regressor'):
        source = price_predict_desicion(price_df, option[1])
    if (status == 'Knnclassification'):
        source = knnclassification(price_df, option[1])

    make_a_graph(source)
    distributed(price_df)


@st.cache(ttl=60 * 60 * 1, allow_output_mutation=True)
def load_data(df, n):
    # drop duplicates
    df = df.drop_duplicates(subset=['Suburb', 'Address', 'Date', 'Price'], keep='last')

    if n == 2:
        # find median price by grouping suburbs
        df['Price'] = df.groupby(['Suburb'])['Price'].apply(lambda x: x.fillna(x.median()))
    if n == 3:
        # find mean price by grouping suburbs
        df['Price'] = df.groupby(['Suburb'])['Price'].apply(lambda x: x.fillna(x.mean()))
    if n == 4:
        # find mode price by grouping suburbs
        df['Price'] = df.groupby(['Suburb'])['Price'].apply(lambda x: x.fillna(x.mode()))

    # the n == 1 wil come here
    df = df.dropna(subset=['Price'])

    # change the day/month/year to year-month-date
    # I created 3 different columns to store each value (day, month, year) and then i merged them
    df[["day", "month", "year"]] = df["Date"].str.split("/", expand=True)
    df['Dates'] = df[df.columns[16:12:-1]].apply(
        lambda x: '-'.join(x.dropna().astype(str)),
        axis=1)
    # i dropped the extra 3 coli,ms that i created
    # we also dropped councilArea and Postcode because they were unnecessary
    df = df.drop(columns=['Date', 'day', 'month', 'year', 'CouncilArea', 'Postcode'])

    # print(df.isna().sum())
    return df

#desicion tree
def price_predict_desicion(dataset, sub):
    # we created a copy to not interfere with our original dataset
    df = dataset.copy()

    # removes spaces from sub(suburb)
    if sub.strip():
        # get the index of those suburbs that are not equal with the suburb that we want
        indexNames = df[~(df['Suburb'] == sub)].index
        # then we drop the useless suburbs
        df.drop(indexNames, inplace=True)
    if sub == 'All':
        df = dataset.copy()
    # the :, means get the columns, and we get the values of the specific columns
    X = df.iloc[:, [0, 2, 3, 5, 7, 9, 10]].values
    # we get the 4th column which is the price
    y = df.iloc[:, 4].values
    # we get the year only
    X[:, 6] = pd.DatetimeIndex(X[:, 6]).year
    # print our dataset
    st.dataframe(df)
    # we encode each column, with other words we change them into numbers 0 to n_classes-1
    le_X_0 = LabelEncoder()
    le_X_2 = LabelEncoder()
    le_X_3 = LabelEncoder()
    le_X_4 = LabelEncoder()
    #we scale the training data, scale correlates the data so it learns that each column represents different meaning of its value

    X[:, 0] = le_X_0.fit_transform(X[:, 0])
    X[:, 2] = le_X_2.fit_transform(X[:, 2])
    X[:, 3] = le_X_3.fit_transform(X[:, 3])
    X[:, 4] = le_X_4.fit_transform(X[:, 4])

    # split, test set -> 20%, train set -> 80%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #set the maximum depth of the tree
    max_depth = np.arange(1, 25)
    #puts zero in the length of max_depth 1x25 with zeros in it
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    #execute loop 25 times
    #each time it trains the algorithm and predicts it with the test set
    for i, d in enumerate(max_depth):
        # make the tree
        regressor = DecisionTreeRegressor(max_depth=d)
        #train algorithm
        regressor.fit(X_train, y_train)
        #store the accuracy that it gets on each iterration
        train_err[i] = rmsle(y_train, regressor.predict(X_train))
        test_err[i] = rmsle(y_test, regressor.predict(X_test))

    #after the ML is trained we predict the price with the test set
    y_pred = regressor.predict(X_test)

    #create 2 columns to show the train and test accuracy
    col5, col4 = st.columns(2)

    with col5:
        st.write("Accuracy attained on Training Set = ", train_err)
    with col4:
        st.write("Accuracy attained on Test Set = ", test_err)
        # store data into source to make a graph

    source = DataFrame(
        dict(
            x_values=X_test[:, 6],
            y_values=y_pred
        ))

    return source



def knnclassification(dataset, sub):
    # we created a copy to not interfere with our original dataset
    df = dataset.copy()

    # removes spaces from sub(suburb)
    if sub.strip():
        # get the index of those suburbs that are not equal with the suburb that we want
        indexNames = df[~(df['Suburb'] == sub)].index
        # then we drop the useless suburbs
        df.drop(indexNames, inplace=True)
        #if its ALl suburbs then just copy the dataset
    if sub == 'All':
        df = dataset.copy()

        # the :, means get the columns, and we get the values of the specific columns
        # i means integer, loc means location, simple words it gets the index
    X = df.iloc[:, [0, 1, 2, 3, 5, 7, 9, 10]].values
    y = df.iloc[:, 4].values
    #get only the year
    X[:, 7] = pd.DatetimeIndex(X[:, 7]).year
    st.dataframe(df)
    # Encode target labels with value between 0 and n_classes-1.
    le_X_0 = LabelEncoder()
    le_X_1 = LabelEncoder()
    le_X_3 = LabelEncoder()
    le_X_4 = LabelEncoder()
    le_X_5 = LabelEncoder()

    #we scale the training data, scale correlates the data so it learns that each column represents different meaning of its value
    X[:, 0] = le_X_0.fit_transform(X[:, 0])
    X[:, 1] = le_X_1.fit_transform(X[:, 1])
    X[:, 3] = le_X_3.fit_transform(X[:, 3])
    X[:, 4] = le_X_4.fit_transform(X[:, 4])
    X[:, 5] = le_X_5.fit_transform(X[:, 5])

    #test set -> 20%, training set -> 80%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #call the standardscaler
    scaler = StandardScaler()
    # Variables that are measured at different scales do not contribute equally to the model fitting & model learned function and might end up creating a bias.
    # Thus, to deal with this potential problem feature-wise standardized (μ=0, σ=1) is usually used prior to model fitting.
    scaler.fit(X_train)

    #transform applys the scaler and returns different dataset
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #call the knn having 5 neighbors which means each data point it will be compared with its 5 neighboors in order to be labelled
    classifier = KNeighborsClassifier(n_neighbors=5)

    #train the algorithm
    classifier.fit(X_train, y_train)
    #predict price with train set
    y_pred_train = classifier.predict(X_train)
    #predict price with the test set
    y_pred = classifier.predict(X_test)
    #accuracy of predictions
    st.write("Accuracy attained on Training Set = ", rmsle(y_pred_train, y_train))
    st.write("Accuracy attained on Test Set = ", rmsle(y_pred, y_test))
    #show our dataset
  #  st.dataframe(X_test)
    #store data into source to make a graph
    source = DataFrame(
        dict(
            x_values=X_test[:, 6],
            y_values=y_pred
        ))

    return source

#xgboost
def predict_price_for_graph(dataset, sub):
    # we created a copy to not interfere with our original dataset
    df = dataset.copy()

    # removes spaces from sub(suburb)
    if sub.strip():
        # get the index of those suburbs that are not equal with the suburb that we want
        indexNames = df[~(df['Suburb'] == sub)].index
        # then we drop the useless suburbs
        df.drop(indexNames, inplace=True)
        # if its ALl suburbs then just copy the dataset
    if sub == 'All':
        df = dataset.copy()

    # the :, means get the columns, and we get the values of the specific columns
    #i means integer, loc means location, simple words it gets the index
    X = df.iloc[:, [0, 1, 2, 3, 5, 7, 9, 10]].values
    y = df.iloc[:, 4].values

    X[:, 7] = pd.DatetimeIndex(X[:, 7]).year
    st.dataframe(df)
    # Encode target labels with value between 0 and n_classes-1.
    le_X_0 = LabelEncoder()
    le_X_1 = LabelEncoder()
    le_X_3 = LabelEncoder()
    le_X_4 = LabelEncoder()
    le_X_5 = LabelEncoder()
    #we scale the training data, scale correlates the data so it learns that each column represents different meaning of its value
    X[:, 0] = le_X_0.fit_transform(X[:, 0])
    X[:, 1] = le_X_1.fit_transform(X[:, 1])
    X[:, 3] = le_X_3.fit_transform(X[:, 3])
    X[:, 4] = le_X_4.fit_transform(X[:, 4])
    X[:, 5] = le_X_5.fit_transform(X[:, 5])

    #20% for test, 80% for train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #calls the ML method
    regressor = XGBRegressor()
    #train ML with train data
    regressor.fit(X_train, y_train)
    #predict price with train set
    Y_pred_train = regressor.predict(X_train)
    #predict price with test set
    y_pred = regressor.predict(X_test)

    #find accuracy
    st.write("Accuracy attained on Training Set = ", rmsle(Y_pred_train, y_train))
    st.write("Accuracy attained on Test Set = ", rmsle(y_pred, y_test))
    #store data into source to make a graph

    source = DataFrame(
        dict(
            x_values=X_test[:, 7],
            y_values=y_pred
        ))

    return source


def distributed(dataset):
    #simple graph on distribution of the price
    sns.distplot(dataset['Price'], bins=20)
    plt.title('Distribution of listing ratings')
    st.pyplot(plt)


def make_a_graph(source):
    #graph on years with prediction
    gp = source.groupby('x_values').mean()

    #st.write(gp.values)
    p = figure()
    p.xaxis.ticker = gp.index.values
    p.vbar(x=gp.index.values, top=gp['y_values'], width=0.9)
    p.xaxis.axis_label = 'Years'
    p.yaxis.axis_label = 'Prediction'
    st.bokeh_chart(p)

#accuracy of our algorithm
def rmsle(y_pred, y_test):
    error = np.square(np.log10(y_pred + 1) - np.log10(y_test + 1)).mean() ** 0.5
    Acc = 1 - error
    return Acc

