import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import date

# ---------- Load CatBoost Model ----------
with open("catboost_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------- Page Configuration ----------
st.set_page_config(page_title="🎬 Movie Like Predictor", layout="centered")

# ---------- Custom CSS ----------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://images.unsplash.com/photo-1598899134739-6e7e94b9f2a6?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}
    .main-container {{
        background-color: rgba(0, 0, 0, 0.7);
        padding: 2rem;
        border-radius: 15px;
    }}
    .logo {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 120px;
    }}
    h1 {{
        text-align: center;
        color: white;
        font-size: 36px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Logo & Title ----------
st.markdown('<img src="https://cdn-icons-png.flaticon.com/512/744/744922.png" class="logo">', unsafe_allow_html=True)
st.markdown("<h1>🎬 Movie Like Prediction System</h1>", unsafe_allow_html=True)

# ---------- Label Encoders Used in Training ----------
label_maps = {
    'watch_time_slot': {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3},
    'gender': {'Male': 0, 'Female': 1},
    'membership_type': {'Free': 0, 'Basic': 1, 'Premium': 2},
    'preferred_genre': {'Action': 0, 'Drama': 1, 'Comedy': 2, 'Thriller': 3},
    'genre': {'Action': 0, 'Drama': 1, 'Comedy': 2, 'Thriller': 3},
    'language': {'English': 0, 'Hindi': 1, 'Marathi': 2},
    'age_rating': {'G': 0, 'PG': 1, 'PG-13': 2, 'R': 3},
    'device_type': {'Mobile': 0, 'Tablet': 1, 'Laptop': 2},
    'os': {'Android': 0, 'iOS': 1, 'Windows': 2},
    'supports_hd': {'No': 0, 'Yes': 1}
}

# ---------- Form Input ----------
with st.form("input_form"):
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    watch_date = st.date_input("Watch Date").toordinal()
    watch_duration = st.slider("Watch Duration (mins)", 10, 300, 90)
    watch_time_slot = st.selectbox("Watch Time Slot", list(label_maps['watch_time_slot'].keys()))
    rating_given = st.slider("User Rating", 0.0, 5.0, 3.5, step=0.1)
    completed = st.selectbox("Completed Watching?", [0, 1])
    age = st.slider("Viewer Age", 10, 100, 30)
    gender = st.selectbox("Gender", list(label_maps['gender'].keys()))
    membership_type = st.selectbox("Membership Type", list(label_maps['membership_type'].keys()))
    preferred_genre = st.selectbox("Preferred Genre", list(label_maps['preferred_genre'].keys()))
    genre = st.selectbox("Genre Watched", list(label_maps['genre'].keys()))
    release_year = st.slider("Release Year", 1990, 2025, 2020)
    duration_mins = st.slider("Movie Duration", 30, 300, 120)
    language = st.selectbox("Language", list(label_maps['language'].keys()))
    age_rating = st.selectbox("Age Rating", list(label_maps['age_rating'].keys()))
    avg_rating = st.slider("Average Movie Rating", 0.0, 5.0, 3.0, step=0.1)
    device_type = st.selectbox("Device Type", list(label_maps['device_type'].keys()))
    os = st.selectbox("Operating System", list(label_maps['os'].keys()))
    screen_size_inch = st.slider("Screen Size (inch)", 4.0, 80.0, 15.6)
    supports_hd = st.selectbox("Supports HD?", list(label_maps['supports_hd'].keys()))

    submit = st.form_submit_button("Predict")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Prediction ----------
if submit:
    input_data = pd.DataFrame([[
        watch_date,
        watch_duration,
        label_maps['watch_time_slot'][watch_time_slot],
        rating_given,
        completed,
        age,
        label_maps['gender'][gender],
        label_maps['membership_type'][membership_type],
        label_maps['preferred_genre'][preferred_genre],
        label_maps['genre'][genre],
        release_year,
        duration_mins,
        label_maps['language'][language],
        label_maps['age_rating'][age_rating],
        avg_rating,
        label_maps['device_type'][device_type],
        label_maps['os'][os],
        screen_size_inch,
        label_maps['supports_hd'][supports_hd],
    ]], columns=[
        'watch_date', 'watch_duration', 'watch_time_slot', 'rating_given', 'completed',
        'age', 'gender', 'membership_type', 'preferred_genre', 'genre',
        'release_year', 'duration_mins', 'language', 'age_rating', 'avg_rating',
        'device_type', 'os', 'screen_size_inch', 'supports_hd'
    ])

    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Prediction: {'LIKED ✅' if prediction == 1 else 'NOT LIKED ❌'}")
