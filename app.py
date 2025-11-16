import streamlit as st
import requests
API_URL = "https://spam-detection-and-profanity-filter-fkdw.onrender.com/predict/"

st.title("Comment Moderation System")
st.write("Detect **Spam**, **Profanity**, or **Clean** comments.")
user_input = st.text_area("Enter a comment to analyze:")
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        payload = {"comment": user_input}
        try:
            response = requests.post(API_URL, json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                st.subheader("Analysis Result")
                st.write(f"Prediction: {result['predicted_label']}")
                st.write(f"Spam Probability: {result['spam_probability']}")
                st.write(f"Profanity Probability: {result['profanity_probability']}")
            else:
                st.error("API returned an error. Check your backend logs.")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
