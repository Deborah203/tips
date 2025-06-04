import sklearn
print(sklearn.__version__)


import joblib
import streamlit as st
import numpy as np

model=joblib.load('tips.pkl')

total_bill=st.number_input('enter total bill')
tips = st.number_input('enter the amount you tipped ')
size=st.number_input('enter size')

if st.button('classify'):
    input_features=np.array([[total_bill, tips, size]])
    pred=model.predict(input_features)
    st.success(pred)
