import streamlit as st
import joblib

st.title("Fake News Detector")

@st.cache_resource
def load_model():
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
    return vectorizer, model

vectorizer, model = load_model()

inputn = st.text_area("Enter News:")

if st.button("Check"):
    if inputn:
        data = vectorizer.transform([inputn])

        # 🔥 use probability instead of direct predict
        prob = model.predict_proba(data)[0]

        real_prob = prob[1]
        fake_prob = prob[0]

        st.write(f"Confidence → Real: {real_prob:.2f}, Fake: {fake_prob:.2f}")

        if real_prob > fake_prob:
            st.success("Real News")
        else:
            st.error("Fake News")
