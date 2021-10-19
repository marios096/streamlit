# simple_streamlit_app.py
"""
A simple streamlit app
run the app by installing streamlit with pip and typing
> streamlit run simple_streamlit_app.py
"""

import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
#import missingno as msno
import datetime as dt

# Read in the dataset
airbnb = pd.read_csv('https://github.com/marios096/streamlit/blob/main/data.csv?raw=true')

#st.title('Simple Streamlit App')

#st.text('Type a number in the box below')

#n = st.number_input('Number', step=1)

#st.write(f'{n} + 1 = {n+1}')
#
#s = st.text_input('Type a name in the box below')
newdf1 = df1.groupby(['Suburb', 'Address'])
  
# print new dataframe
print(newdf1)
#st.write(f'Hello {airbnb.head()}')
