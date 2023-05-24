import pickle
import streamlit as st
import numpy as np
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))

saves = st.number_input("Saves")
comments = st.number_input("Comments")
shares = st.number_input("Shares")
likes = st.number_input("Likes")
profile_visits = st.number_input("Profile Visits")
follows = st.number_input("Follows")

if st.button("Predict"):
	test = np.array([[saves, comments, shares, likes, profile_visits, follows]])
	res = model.predict(test)
	print(res)
	st.success("Prediction: " + str(res[0]))
