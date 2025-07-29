import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from PIL import Image

st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")

model = joblib.load("model.pkl")

logo = Image.open("logo.png")
st.image(logo, width=120)

st.markdown("<h1 style='text-align: center;'>Credit Card Fraud Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect fraudulent credit card transactions using a trained ML model.</p>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("###  Upload your <code>creditcard.csv</code>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop file here", type=["csv"])

if uploaded_file is not None:
    st.success("File uploaded successfully! ")

    with st.spinner(" Predicting fraud transactions... Please wait"):
        time.sleep(1)  
        df = pd.read_csv(uploaded_file)

        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)

        predictions = model.predict(df)
        df['Prediction'] = ['Fraud' if x == 1 else 'Normal' for x in predictions]

        fraud_count = df['Prediction'].value_counts().get('Fraud', 0)
        total_count = len(df)

    st.success(" Prediction complete!")
    st.subheader(" Prediction Results")
    st.dataframe(df, use_container_width=True)

    st.markdown(
        f"<div style='background-color:#ffc107; padding:10px; border-radius:8px; text-align:center;'>"
        f" <strong>{fraud_count}</strong> Fraudulent transactions detected out of <strong>{total_count}</strong>."
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("### Download Results")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇Download Predictions as CSV",
        data=csv_data,
        file_name='fraud_predictions.csv',
        mime='text/csv',
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made  by <strong>Narendra Singh Rajput</strong></p>",
    unsafe_allow_html=True
)
