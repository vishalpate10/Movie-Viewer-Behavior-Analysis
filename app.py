import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# ---------- Page config ----------
st.set_page_config(page_title="🎬 Movie Like Prediction", layout="wide")

# ---------- Custom Background Style ----------
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(to right top, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- App Title ----------
st.markdown('<div class="title">🎬 Movie Platform - User Like Prediction</div>', unsafe_allow_html=True)
st.markdown("### Upload your movie behavior dataset (CSV format)")

# ---------- Upload File ----------
uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Sample Data")
    st.dataframe(df.head(), use_container_width=True)

    # ---------- Handle Missing Values ----------
    df['age'].fillna(df['age'].mean(), inplace=True)

    # ---------- Encode Categorical Columns ----------
    cat_cols = ['watch_time_slot', 'gender', 'membership_type',
                'preferred_genre', 'genre', 'language', 'age_rating',
                'device_type', 'os', 'supports_hd']
    
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # ---------- Feature & Target Selection ----------
    x = df[['watch_date', 'watch_duration', 'watch_time_slot',
            'rating_given', 'completed', 'age', 'gender', 'membership_type',
            'preferred_genre', 'genre', 'release_year', 'duration_mins', 'language',
            'age_rating', 'avg_rating', 'device_type', 'os', 'screen_size_inch',
            'supports_hd']]
    
    y = df['liked']

    # ---------- Convert date ----------
    x['watch_date'] = pd.to_datetime(x['watch_date'], errors='coerce')
    x['watch_date'] = x['watch_date'].map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
    x['watch_date'].fillna(x['watch_date'].median(), inplace=True)

    # ---------- Train-Test Split ----------
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # ---------- Model ----------
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # ---------- Evaluation ----------
    y_pred = model.predict(x_test)
    st.subheader("📊 Model Evaluation")
    st.code(classification_report(y_test, y_pred))

    # ---------- Make Prediction ----------
    st.markdown("---")
    st.subheader("🎯 Predict if a user will like the movie")

    with st.form("prediction_form"):
        user_input = {}
        user_input['watch_date'] = st.date_input("Watch Date").toordinal()
        user_input['watch_duration'] = st.number_input("Watch Duration (mins)", 1, 300, 60)
        user_input['watch_time_slot'] = st.selectbox("Watch Time Slot", df['watch_time_slot'].unique())
        user_input['rating_given'] = st.slider("Rating Given", 0.0, 5.0, 3.0, step=0.1)
        user_input['completed'] = st.selectbox("Completed Watching?", [0, 1])
        user_input['age'] = st.number_input("Age", 5, 100, 30)
        user_input['gender'] = st.selectbox("Gender", df['gender'].unique())
        user_input['membership_type'] = st.selectbox("Membership Type", df['membership_type'].unique())
        user_input['preferred_genre'] = st.selectbox("Preferred Genre", df['preferred_genre'].unique())
        user_input['genre'] = st.selectbox("Movie Genre", df['genre'].unique())
        user_input['release_year'] = st.number_input("Release Year", 1980, 2025, 2020)
        user_input['duration_mins'] = st.number_input("Movie Duration (mins)", 30, 240, 120)
        user_input['language'] = st.selectbox("Language", df['language'].unique())
        user_input['age_rating'] = st.selectbox("Age Rating", df['age_rating'].unique())
        user_input['avg_rating'] = st.slider("Average Rating", 0.0, 5.0, 3.5, step=0.1)
        user_input['device_type'] = st.selectbox("Device Type", df['device_type'].unique())
        user_input['os'] = st.selectbox("Operating System", df['os'].unique())
        user_input['screen_size_inch'] = st.slider("Screen Size (inch)", 4.0, 80.0, 15.6)
        user_input['supports_hd'] = st.selectbox("Supports HD", df['supports_hd'].unique())

        submit = st.form_submit_button("Predict")

    if submit:
        for col in cat_cols:
            le = label_encoders[col]
            user_input[col] = le.transform([user_input[col]])[0]

        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Prediction: {'USER LIKED MOVIE✅' if prediction == 1 else 'USER NOT LIKED MOVIE❌'}")
