import streamlit as st
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

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
    st.text('Before removing duplicated we had 63023 records and after we removed the duplicate records we can see that the number of records' + '\n'
            +'have been reduced so our hypothesis was correct')
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
    st.text('When we saw that we have empty Prices thought 2 different cases in order to fix the problem'
            + '\n' + '- The first case is to delete the rows that contain empty Prices.'
            + '\n' + '- The second case is to set median/mean/mode with group by of Suburbs in the empty Prices')

    st.header('Task 2')
    st.subheader('ML methods')
    st.subheader('XGBOOST')
    st.text('Now we have a clean dataset and we have to apply the Machine Learning methods. ' + '\n' +
            'We have created a dropdown menu that gives the user the choice to select either XGBOOST or Desicion Tree Regressor or K nearest neighboor'
            + '\n' + 'to train and test the dataset. We implemented the project with 3 different methods to see which one is the best '
            + '\n' + 'because there are plenty of algorithms that can do this task.'
            + '\n' + 'The first method is the XGBOOST and it is implemented with the library from xgboost import XGBRegressor'
            + '\n' + ' def predict_price_for_graph(dataset, sub):' + '\n' + '  df = dataset.copy()'+ '\n' + '  if sub.strip():'
            + '\n' + '    indexNames = df[~(df[Suburb] == sub)].index'+ '\n' + '    df.drop(indexNames, inplace=True)'
            + '\n' + '  if sub == All:'+ '\n' + '    df = dataset.copy()'
            + '\n' + '  X = df.iloc[:,[0,1,2,3,5,7,9,10]].values' + '\n' + '  y = df.iloc[:, 4].values'
            + '\n' + '  X[:, 7] = pd.DatetimeIndex(X[:, 7]).year' + '\n' + '  le_X_0 = LabelEncoder()' + '\n' + '  le_X_2 = LabelEncoder()'
            + '\n' + '  le_X_3 = LabelEncoder()' + '\n' + '  le_X_4 = LabelEncoder()' + '\n' + '  X[:, 0] = le_X_0.fit_transform(X[:, 0])'
            + '\n' + '  X[:, 1] = le_X_1.fit_transform(X[:, 1])' + '\n' + '  X[:, 3] = le_X_3.fit_transform(X[:, 3])'
            + '\n' + '  X[:, 4] = le_X_4.fit_transform(X[:, 4])' + '\n' + '  X[:, 5] = le_X_5.fit_transform(X[:, 5])'
            + '\n' + '  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)'
            + '\n' + '  regressor = XGBRegressor()' + '\n' + '  regressor.fit(X_train, y_train)'
            + '\n' + '  Y_pred_train = regressor.predict(X_train)' + '\n' + '  y_pred = regressor.predict(X_test)'
            + '\n' + '  st.write("Accuracy attained on Training Set = ", rmsle(Y_pred_train, y_train))'
            + '\n' + '  st.write("Accuracy attained on Test Set = ", rmsle(y_pred, y_test))'
            + '\n' + 'XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) '
            + '\n' + 'machine learning library. It provides parallel tree boosting and is the leading machine learning library for regression,'
            + '\n' + 'classification, and ranking problems.A Gradient Boosting Decision Trees (GBDT) is a decision tree ensemble learning algorithm similar '
            + '\n' + 'to random forest, for classification and regression. Ensemble learning algorithms combine multiple machine learning '
            + '\n' + 'algorithms to obtain a better model.')
    st.text("")
    st.subheader('Desicion Tree Regressor')
    st.text('The second method that the user can select is the Desicion Tree Regressor and it is implemented as shown below'
    + '\n' + 'def price_predict_desicion(dataset, sub):'
            + '\n' + '  df = dataset.copy()' + '\n' + '    if sub.strip():'
            + '\n' + '      indexNames = df[~(df[Suburb] == sub)].index' + '\n' + '      df.drop(indexNames, inplace=True)'
            + '\n' + '    if sub == All:' + '\n' + '      df = dataset.copy()'
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
            + '\n' + 'Decision tree builds regression or classification models in the form of a tree structure. '
            + '\n' + 'It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. '
            + '\n' + 'The final result is a tree with decision nodes and leaf nodes.')

    st.subheader('knnclassification')
    st.text('The third method that the user can select is the K nearest Neighboor'
    + '\n' + 'def knnclassification(dataset, sub):'
            + '\n' + 'def price_predict_desicion(dataset, sub):'
            + '\n' + '  df = dataset.copy()' + '\n' + '    if sub.strip():'
            + '\n' + '      indexNames = df[~(df[Suburb] == sub)].index' + '\n' + '      df.drop(indexNames, inplace=True)'
            + '\n' + '    if sub == All:' + '\n' + '      df = dataset.copy()'
            + '\n' + '    X = df.iloc[:, [0, 1, 2, 3, 5, 7, 9, 10]].values'
            + '\n' + '    y = df.iloc[:, 4].values'

            + '\n' + '    X[:, 7] = pd.DatetimeIndex(X[:, 7]).year'

            + '\n' + '    le_X_0 = LabelEncoder()'
            + '\n' + '    le_X_1 = LabelEncoder()'
            + '\n' + '    le_X_3 = LabelEncoder()'
            + '\n' + '    le_X_4 = LabelEncoder()'
            + '\n' + '    le_X_5 = LabelEncoder()'
            + '\n' + '    X[:, 0] = le_X_0.fit_transform(X[:, 0])'
            + '\n' + '    X[:, 1] = le_X_2.fit_transform(X[:, 1])'
            + '\n' + '    X[:, 3] = le_X_3.fit_transform(X[:, 3])'
            + '\n' + '    X[:, 4] = le_X_4.fit_transform(X[:, 4])'
            + '\n' + '    X[:, 5] = le_X_4.fit_transform(X[:, 5])'
            + '\n' + '    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)'
            + '\n' + '    scaler = StandardScaler()'
            + '\n' + '    scaler.fit(X_train)'
            + '\n' + '    X_train = scaler.transform(X_train)'
            + '\n' + '    X_test = scaler.transform(X_test)'
            + '\n' + '    classifier = KNeighborsClassifier(n_neighbors=5)'
            + '\n' + '    classifier.fit(X_train, y_train)'
            + '\n' + '    y_pred_train = classifier.predict(X_train)'
            + '\n' + '    y_pred = classifier.predict(X_test)'
            + '\n' + '    st.write("Accuracy attained on Training Set = ", rmsle(y_pred_train, y_train))'
            + '\n' + '    st.write("Accuracy attained on Test Set = ", rmsle(y_pred, y_test))'
            + '\n' + 'K nearest neighboor takes as a parameter a value K that represents how many neighboors we want to '
                     'compare our data with.')
    st.subheader('Correlation Matrix')
    corr_matrix = df.corr()
    fig, ax = plt.subplots()
    ax = sns.heatmap(corr_matrix,
                     annot=True,
                     linewidths=0.5,
                     fmt=".2f",
                     cmap="YlGnBu")

    bottom, top = ax.get_ylim()
    st.pyplot(fig)
