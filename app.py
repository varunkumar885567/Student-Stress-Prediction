#  Student Stress Prediction Dashboard (Streamlit)
# ==========================================
# FUTURISTIC AI STUDENT STRESS DASHBOARD
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Stress Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.main {
    background: linear-gradient(to right, #0f172a, #111827);
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 15px;
    height: 3em;
    background: linear-gradient(90deg, #06b6d4, #8b5cf6);
    color: white;
    border: none;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #8b5cf6, #06b6d4);
    color: white;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 0px 15px rgba(0,255,255,0.2);
    backdrop-filter: blur(12px);
}

.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 0px 12px rgba(0,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD FILES
# ==========================================

model = joblib.load("student_stress_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
encoder = joblib.load("label_encoder.pkl")

# ==========================================
# HEADER
# ==========================================

st.markdown(
    """
    <h1 style='text-align:center;'>🧠 StressSense AI DASHBOARD</h1>
    <p style='text-align:center; font-size:20px; color:#cbd5e1;'>
    Real-Time Mental Wellness & Lifestyle Analysis System
    </p>
    """,
    unsafe_allow_html=True
)

st.write("---")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Student Lifestyle Inputs")

study_hours = st.sidebar.number_input(
    "📚 Study Hours Per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

sleep_hours = st.sidebar.number_input(
    "😴 Sleep Hours Per Day",
    min_value=0.0,
    max_value=24.0,
    value=7.0,
    step=0.5
)

social_media = st.sidebar.number_input(
    "📱 Social Media Usage (Hours)",
    min_value=0.0,
    max_value=24.0,
    value=2.0,
    step=0.5
)

physical_activity = st.sidebar.number_input(
    "🏃 Physical Activity (Hours)",
    min_value=0.0,
    max_value=24.0,
    value=1.0,
    step=0.5
)
gpa = st.sidebar.number_input(
    "🎓 GPA / Academic Performance",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

st.sidebar.write("---")
st.sidebar.subheader("🧠 Mental Health Assessment")

# ==========================================
# QUESTIONNAIRE
# ==========================================

q1 = st.sidebar.selectbox(
    "Do you feel nervous frequently?",
    ["Never", "Sometimes", "Often"]
)

q2 = st.sidebar.selectbox(
    "Do you overthink about studies?",
    ["Never", "Sometimes", "Often"]
)

q3 = st.sidebar.selectbox(
    "Do exams make you stressed?",
    ["Never", "Sometimes", "Often"]
)

q4 = st.sidebar.selectbox(
    "Do you feel emotionally low?",
    ["Never", "Sometimes", "Often"]
)

q5 = st.sidebar.selectbox(
    "Do you lack motivation?",
    ["Never", "Sometimes", "Often"]
)

q6 = st.sidebar.selectbox(
    "Do you feel tired most of the time?",
    ["Never", "Sometimes", "Often"]
)

# ==========================================
# SCORE CONVERSION
# ==========================================

score_map = {
    "Never": 1,
    "Sometimes": 2,
    "Often": 3
}

anxiety_score = (
    score_map[q1] +
    score_map[q2] +
    score_map[q3]
)

depression_score = (
    score_map[q4] +
    score_map[q5] +
    score_map[q6]
)

anxiety_level = round((anxiety_score / 9) * 10)
depression = round((depression_score / 9) * 10)

# ==========================================
# METRIC CARDS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>📚 Study</h3>
        <h2>{study_hours} hrs</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>😴 Sleep</h3>
        <h2>{sleep_hours} hrs</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>😟 Anxiety</h3>
        <h2>{anxiety_level}/10</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>😔 Depression</h3>
        <h2>{depression}/10</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# ==========================================
# PREPARE INPUT
# ==========================================
input_dict = {
    "student_id": 1,

    "study_hours_per_day": study_hours,

    "extracurricular_hours_per_day": 2,

    "sleep_hours_per_day": sleep_hours,

    "social_hours_per_day": social_media,

    "physical_activity_hours_per_day": physical_activity,

    "gpa": gpa,
    "anxiety_level": anxiety_level,
    "depression": depression
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=features, fill_value=0)

input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)
result = encoder.inverse_transform(prediction)[0]

# ==========================================
# STRESS DISPLAY
# ==========================================

st.subheader("🧠 AI Stress Prediction")

if result.lower() == "high":
    st.error(f"🔴 High Stress Detected")
    stress_score = 90

elif result.lower() == "medium":
    st.warning(f"🟡 Moderate Stress Detected")
    stress_score = 60

else:
    st.success(f"🟢 Low Stress Detected")
    stress_score = 25

# ==========================================
# STRESS METER
# ==========================================

st.progress(stress_score)

st.write(f"### Stress Score: {stress_score}%")

# ==========================================
# CHARTS
# ==========================================

colA, colB = st.columns(2)

with colA:

    st.markdown("### 📊 Lifestyle Analytics")

    labels = [
        'Study',
        'Sleep',
        'Social Media',
        'Physical Activity'
    ]

    values = [
        study_hours,
        sleep_hours,
        social_media,
        physical_activity
    ]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(labels, values)
    ax.set_ylabel("Hours")
    ax.set_title("Daily Lifestyle Distribution")

    st.pyplot(fig)

with colB:

    st.markdown("### 🧠 Mental Health Analysis")

    mental_labels = ['Anxiety', 'Depression']
    mental_values = [anxiety_level, depression]

    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.pie(
        mental_values,
        labels=mental_labels,
        autopct='%1.1f%%'
    )

    ax2.set_title("Mental Health Indicators")

    st.pyplot(fig2)

# ==========================================
# AI RECOMMENDATIONS
# ==========================================

st.write("---")
st.subheader("🤖 AI Wellness Recommendations")

if result.lower() == "high":

    st.markdown("""
    <div class='card'>

    🔴 <b>High Stress Detected</b><br><br>

    ✅ Improve sleep schedule<br>
    ✅ Reduce excessive screen time<br>
    ✅ Practice meditation or yoga<br>
    ✅ Take regular study breaks<br>
    ✅ Increase physical activity<br>

    </div>
    """, unsafe_allow_html=True)

elif result.lower() == "medium":

    st.markdown("""
    <div class='card'>

    🟡 <b>Moderate Stress Detected</b><br><br>

    ✅ Maintain balanced routine<br>
    ✅ Stay socially active<br>
    ✅ Exercise regularly<br>
    ✅ Maintain healthy sleep habits<br>

    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div class='card'>

    🟢 <b>Low Stress Level</b><br><br>

    ✅ Great mental wellness<br>
    ✅ Continue healthy lifestyle habits<br>
    ✅ Maintain work-life balance<br>

    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================

st.write("---")

st.markdown(
    """
    <center>
    <p style='color:gray;'>
    Developed using Machine Learning & Streamlit 🚀
    </p>
    </center>
    """,
    unsafe_allow_html=True
)
