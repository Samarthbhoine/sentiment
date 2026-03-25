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
    # score = result[0]["score"]

    return label


def analyze_with_genai(text):
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        f"""
        Analyze the sentiment of this review and explain briefly:
        Review: {text}
        
        Give:
        - Sentiment (positive/negative)
        - Short explanation
        """
    )
    return response.text