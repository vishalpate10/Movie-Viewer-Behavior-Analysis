import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import date

# Load trained CatBoost model
with open("catboost_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title and style
st.set_page_config(page_title="🎬 Movie Like Prediction", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>🎬 Movie Platform - Like Prediction</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e2f;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader("Enter User & Movie Details for Prediction")

# Input form
with st.form("user_form"):
    watch_date = st.date_input("Watch Date").toordinal()
    watch_duration = st.slider("Watch Duration (mins)", 10, 300, 90)
    watch_time_slot = st.selectbox("Watch Time Slot", [0, 1, 2, 3])  # already label encoded
    rating_given = st.slider("Rating Given by User", 0.0, 5.0, 3.5, step=0.1)
    completed = st.selectbox("Completed Watching?", [0, 1])
    age = st.slider("User Age", 10, 80, 30)
    gender = st.selectbox("Gender", [0, 1])  # encoded: 0=Male, 1=Female
    membership_type = st.selectbox("Membership Type", [0, 1, 2])  # encoded
    preferred_genre = st.selectbox("Preferred Genre", [0, 1, 2, 3])
    genre = st.selectbox("Movie Genre", [0, 1, 2, 3])
    release_year = st.slider("Release Year", 1990, 2025, 2020)
    duration_mins = st.slider("Movie Duration", 30, 240, 120)
    language = st.selectbox("Language", [0, 1, 2])  # encoded
    age_rating = st.selectbox("Age Rating", [0, 1, 2, 3])
    avg_rating = st.slider("Average Rating", 0.0, 5.0, 3.0, step=0.1)
    device_type = st.selectbox("Device Type", [0, 1, 2])
    os = st.selectbox("Operating System", [0, 1, 2])
    screen_size_inch = st.slider("Screen Size (inches)", 4.0, 80.0, 15.6)
    supports_hd = st.selectbox("Supports HD", [0, 1])

    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame([[
        watch_date, watch_duration, watch_time_slot, rating_given, completed,
        age, gender, membership_type, preferred_genre, genre, release_year,
        duration_mins, language, age_rating, avg_rating, device_type,
        os, screen_size_inch, supports_hd
    ]], columns=['watch_date', 'watch_duration', 'watch_time_slot',
                 'rating_given', 'completed', 'age', 'gender', 'membership_type',
                 'preferred_genre', 'genre', 'release_year', 'duration_mins', 'language',
                 'age_rating', 'avg_rating', 'device_type', 'os', 'screen_size_inch',
                 'supports_hd'])

    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Prediction: {'USER LIKED MOVIE✅' if prediction == 1 else 'USER NOT LIKED MOVIE❌'}")
