import streamlit as st
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

import numpy as np

def streamlit_app():

    #sidebar

    #main page
    st.title('Welcome to our project')
    st.info('This project is about predicting prices based on some information that we have on our dataset')
    st.text("Using the available data and parameters that we are provided with, we have to create a model"
            " that can predict " + '\n' + "the house price  ")


    st.header('Task 1')
    st.subheader('Cleaning Data')
    df = pd.read_csv('https://github.com/marios096/streamlit/blob/main/data.csv?raw=true')

    st.text('First we need to read our data from the github ' + '\n' +
            'df = pd.read_csv(https://github.com/marios096/streamlit/blob/main/data.csv?raw=true)')
    st.text('After we read the data we made a hypothesis that there might be duplicate records so we wanted to remove '
    + '\n' + 'the duplicate records based on the Suburb, Address, Date and the Price because it is not possible that '
            + '\n' + 'the same house is sold 2 times the same date for the same price' +'\n'+
            'df = df.drop_duplicates(subset=[Suburb, Address, Date, Price], keep=last)')

    msno.matrix(df)
    st.pyplot(plt)
    df = df.drop_duplicates(subset=['Suburb', 'Address', 'Date', 'Price'], keep='last')
    st.text('Before removing duplicated we had 63023 records and after we removed the duplicate records we can see that our records' + '\n'
            +'have declined so our hypothesis was correct')
    msno.matrix(df)
    st.pyplot(plt)
    st.text("We checked the types of our columns with df.dtypes to see if the numeric columns are numeric and not objects")

    st.text(
        'Now we need to see what parameters have empty cells in them.'
        + '\n' + 'To do that we wrote the command ' + '\n' +
        'df.isna().sum()' + '\n' + 'which returns the columns with how many empty lines they contain')
    st.write(df.isna().sum())
    st.text('\nWe can see the distribution of each of our numeric columns')
    df.hist(figsize=(20, 20), xrot=-45)
    st.pyplot(plt)
    st.text("Then we saw that we have empty cells on the Price column ")
    st.text('When we saw that we have empty Prices thought 3 different cases in order to fix the problem'
            + '\n' + '- The first case is to delete the rows that contain empty Prices.'
            + '\n' + '- The second case is to set median/mean/mode/(median/mean with group by of one of our parameters) in the empty Prices'
            + '\n' + '- The third case is to do group the Dataset By the Suburb and find their mean Price and set it to the other empty prices')

    st.header('Task 2')
    st.subheader('ML methods')
    st.subheader('XGBOOST')
    st.text('Now we have a clean dataset and we have to apply the Machine Learning methods. ' + '\n' +
            'We have created a dropdown menu that gives the user the choice to select either XGBOOST or Desicion Tree Regressor'
            + '\n' + 'to train and test the dataset. We implemented the project with 2 different methods to see which one is the best '
            + '\n' + 'because there are plenty of algorithms that can do this task.'
            + '\n' + 'The first method is the XGBOOST and it is implemented with the library from xgboost import XGBRegressor'
            + '\n' + ' def predict_price(df):' + '\n' + '  X = df.iloc[:,[0,2,3,5,7,9,10]].values' + '\n' + '  y = df.iloc[:, 4].values'
            + '\n' + '  X[:, 6] = pd.DatetimeIndex(X[:, 6]).year' + '\n' + '  le_X_0 = LabelEncoder()' + '\n' + '  le_X_2 = LabelEncoder()'
            + '\n' + '  le_X_3 = LabelEncoder()' + '\n' + '  le_X_4 = LabelEncoder()' + '\n' + '  X[:, 0] = le_X_0.fit_transform(X[:, 0])'
            + '\n' + '  X[:, 2] = le_X_2.fit_transform(X[:, 2])' + '\n' + '  X[:, 3] = le_X_3.fit_transform(X[:, 3])'
            + '\n' + '  X[:, 4] = le_X_4.fit_transform(X[:, 4])'
            + '\n' + '  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)'
            + '\n' + '  regressor = XGBRegressor()' + '\n' + '  regressor.fit(X_train, y_train)'
            + '\n' + '  Y_pred_train = regressor.predict(X_train)' + '\n' + '  y_pred = regressor.predict(X_test)'
            + '\n' + '  st.write("Accuracy attained on Training Set = ", rmsle(Y_pred_train, y_train))'
            + '\n' + '  st.write("Accuracy attained on Test Set = ", rmsle(y_pred, y_test))'
            + '\n' + '  return X'
            + '\n' + 'Firstly we turn string into value so the machine can understand the data, then we seperate the data'
            + '\n' + 'to train set which is the 80% of our data and the test set which is the 20% of our dataset'
            + '\n' + 'Then we apply the method and then print the accuracy of the training and test set')
    st.text("")
    st.subheader('Desicion Tree Regressor')
    st.text('The second method that the user can select is the Desicion Tree Regressor and it is implemented as shown below'
    + '\n' + 'def price_predict_desicion(df):'
            + '\n' + '    X = df.iloc[:, [0, 2, 3, 5, 7, 9, 10]].values'
            + '\n' + '    y = df.iloc[:, 4].values'

            + '\n' + '    X[:, 6] = pd.DatetimeIndex(X[:, 6]).year'

            + '\n' + '    le_X_0 = LabelEncoder()'
            + '\n' + '    le_X_2 = LabelEncoder()'
            + '\n' + '    le_X_3 = LabelEncoder()'
                 + '\n' + '    le_X_4 = LabelEncoder()'

                 + '\n' + '    X[:, 0] = le_X_0.fit_transform(X[:, 0])'
                  + '\n' + '    X[:, 2] = le_X_2.fit_transform(X[:, 2])'
                  + '\n' + '    X[:, 3] = le_X_3.fit_transform(X[:, 3])'
                  + '\n' + '    X[:, 4] = le_X_4.fit_transform(X[:, 4])'
            + '\n' + '    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)'

            + '\n' + '    max_depth = np.arange(1, 25)'
            + '\n' + '    train_err = np.zeros(len(max_depth))'
            + '\n' + '    test_err = np.zeros(len(max_depth))'
            + '\n' + '    for i, d in enumerate(max_depth):'
            + '\n' + '        regressor = DecisionTreeRegressor(max_depth=d)'

            + '\n' + '        regressor.fit(X_train, y_train)'

            + '\n' + '        train_err[i] = rmsle(y_train, regressor.predict(X_train))'

            + '\n' + '        test_err[i] = rmsle(y_test, regressor.predict(X_test))'

            + '\n' + '    st.write("Accuracy attained on Training Set = ", train_err)'
            + '\n' + '    st.write("Accuracy attained on Test Set = ", test_err)'
            + '\n' + '    return X'
            + '\n' + 'As before we trasnform our dataset in order to be understandable by the machine and then we split'
            + '\n' + 'the dataset to 80% training set and 20% testing set'
            + '\n' + 'Then we define our depth that the tree will reach and we execute the method until we reach the max depth'
            + '\n' + 'and then we print the accuracy of each set and in this case we will have an array that will be the accuracy'
                     + '\n' + 'of each iteration until it reaches the max depth')

