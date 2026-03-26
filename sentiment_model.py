from transformers import pipeline
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Load sentiment analysis model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    return_all_scores=True
)

def analyze_sentiment(text):
    results = sentiment_pipeline(text)[0]

    # Convert to dictionary
    scores = {res['label']: res['score'] for res in results}

    # Labels mapping
    label_map = {
        "LABEL_0": "Negative 😞",
        "LABEL_1": "Neutral 😐",
        "LABEL_2": "Positive 😊"
    }

    # Get highest score label
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    return f"{label_map[best_label]} (Confidence: {best_score:.2f})"


def analyze_with_genai(text, best_label):
    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(
        f"""
        You are a sentiment analysis assistant.

        The sentiment of the following text is already determined as: {best_label}

        Text: "{text}"

        Give ONLY a short explanation (1-2 lines) explaining why this sentiment is {label}.
        Do NOT change the sentiment.
        """
    )

    return response.text
