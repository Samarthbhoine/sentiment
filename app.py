import streamlit as st
from sentiment_model import analyze_sentiment
from sentiment_model import analyze_with_genai

st.title("Sentiment Analysis App")

st.write("Enter text to check if sentiment is Positive or Negative or Neutral")

user_input = st.text_area("Enter Text")

if st.button("Analyze"):

    if user_input != "":

        label, score = analyze_sentiment(user_input)
        explain = analyze_with_genai(user_input, label)

        st.write("### Result")
        st.write("Sentiment:", label)
        st.write(f"Confidence: {round(score*100, 2)}%")
        st.write("### Explanation")
        st.write(explain)

    else:
        st.warning("Please enter some text")
