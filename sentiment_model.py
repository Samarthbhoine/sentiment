from transformers import pipeline
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):

    result = sentiment_pipeline(text)

    label = result[0]["label"]
    score = result[0]["score"]

    # Convert to Neutral if confidence is low
    if score < 0.6:
        label = "Neutral 😐"
    elif label == "POSITIVE":
        label = "Positive 😊"
    elif label == "NEGATIVE":
        label = "Negative 😞"

    return label


def analyze_with_genai(text, label):
    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(
        f"""
        You are a sentiment analysis assistant.

        The sentiment of the following text is already determined as: {label}

        Text: "{text}"

        Give ONLY a short explanation (1-2 lines) explaining why this sentiment is {label}.
        Do NOT change the sentiment.
        """
    )

    return response.text
