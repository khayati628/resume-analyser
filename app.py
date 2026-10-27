from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from utils import clean_resume, extract_text

MODEL_DIR = Path("models")

st.set_page_config(page_title="Resume Analyser", page_icon="📄")


@st.cache_resource
def load_models():
    try:
        tfidf = joblib.load(MODEL_DIR / "tfidf.pkl")
        model = joblib.load(MODEL_DIR / "classifier.pkl")
    except FileNotFoundError:
        st.error("Model files not found. Run `python train.py` first.")
        st.stop()
    return tfidf, model


def predict(text, tfidf, model, top_n=3):
    vector = tfidf.transform([clean_resume(text)])
    probabilities = model.predict_proba(vector)[0]
    top = probabilities.argsort()[::-1][:top_n]
    return [(model.classes_[i], probabilities[i] * 100) for i in top]


tfidf, model = load_models()

st.title("📄 Resume Analyser")
st.write("Upload a PDF or DOCX resume and predict its job category.")

uploaded_file = st.file_uploader("Choose a resume", type=["pdf", "docx"])

if uploaded_file is not None:
    with st.spinner("Reading resume..."):
        try:
            resume_text = extract_text(uploaded_file, uploaded_file.name)
        except Exception as error:
            st.error(f"Could not read the file: {error}")
            st.stop()

    if not resume_text.strip():
        st.error("No readable text found. Scanned or image-only resumes are not supported.")
        st.stop()

    results = predict(resume_text, tfidf, model)
    best_category, best_confidence = results[0]

    st.subheader("Prediction")
    col1, col2 = st.columns(2)
    col1.metric("Predicted category", best_category)
    col2.metric("Confidence", f"{best_confidence:.1f}%")

    st.subheader("Top matches")
    chart_data = pd.DataFrame(results, columns=["Category", "Confidence (%)"]).set_index("Category")
    st.bar_chart(chart_data)

    with st.expander("Extracted text"):
        st.text_area("Text", resume_text, height=300, label_visibility="collapsed")
