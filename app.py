"""Streamlit app. Run:  streamlit run app.py   (after running train.py)"""
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict whether the student is likely to **pass** or **fail**.")

bundle = joblib.load("model.joblib")
model, features = bundle["model"], bundle["features"]

c1, c2 = st.columns(2)
with c1:
    studytime = st.slider("Weekly study time (1:<2h, 2:2-5h, 3:5-10h, 4:>10h)", 1, 4, 2)
    failures = st.slider("Past class failures", 0, 3, 0)
    absences = st.slider("Absences", 0, 60, 4)
    age = st.slider("Age", 15, 22, 17)
    Medu = st.slider("Mother's education (0-4)", 0, 4, 2)
    Fedu = st.slider("Father's education (0-4)", 0, 4, 2)
    traveltime = st.slider("Travel time (1-4)", 1, 4, 1)
    freetime = st.slider("Free time after school (1-5)", 1, 5, 3)
    goout = st.slider("Going out with friends (1-5)", 1, 5, 3)
    health = st.slider("Health status (1-5)", 1, 5, 3)
with c2:
    Dalc = st.slider("Weekday alcohol use (1-5)", 1, 5, 1)
    Walc = st.slider("Weekend alcohol use (1-5)", 1, 5, 1)
    famrel = st.slider("Family relationship quality (1-5)", 1, 5, 4)
    schoolsup = st.checkbox("Extra school support")
    famsup = st.checkbox("Family educational support", True)
    paid = st.checkbox("Paid extra classes")
    activities = st.checkbox("Extra-curricular activities")
    internet = st.checkbox("Internet at home", True)
    higher = st.checkbox("Wants higher education", True)
    romantic = st.checkbox("In a relationship")

if st.button("Predict"):
    row = dict(studytime=studytime, failures=failures, absences=absences, age=age,
               Medu=Medu, Fedu=Fedu, traveltime=traveltime, freetime=freetime,
               goout=goout, Dalc=Dalc, Walc=Walc, health=health, famrel=famrel,
               schoolsup=int(schoolsup), famsup=int(famsup), paid=int(paid),
               activities=int(activities), internet=int(internet),
               higher=int(higher), romantic=int(romantic))
    X = pd.DataFrame([row])[features]
    prob = model.predict_proba(X)[0][1]
    if prob >= 0.5:
        st.success(f"✅ Likely to PASS (confidence {prob:.0%})")
    else:
        st.error(f"⚠️ At risk of FAILING (pass probability {prob:.0%})")
    tips = []
    if absences > 10: tips.append("Reduce absences - attend more classes.")
    if studytime < 3: tips.append("Increase weekly study time.")
    if failures > 0: tips.append("Get extra help for subjects failed before.")
    if goout > 3: tips.append("Balance social time with study time.")
    if tips:
        st.subheader("Suggestions")
        for t in tips: st.write("- " + t)
