import streamlit as st
import joblib

st.write("🚀 App started")

try:
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
    st.write("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model load error: {e}")

st.title("Fake News Detector")

inputn = st.text_area("Enter News:")

if st.button("Check"):
    if inputn:
        try:
            data = vectorizer.transform([inputn])
            result = model.predict(data)

            if result[0] == 1:
                st.success("Real News")
            else:
                st.error("Fake News")
        except Exception as e:
            st.error(f"Prediction error: {e}")
