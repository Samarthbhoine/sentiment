from transformers import pipeline
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Load sentiment analysis model
sentiment_pipeline = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None 
)

def analyze_sentiment(text):
    results = sentiment_pipeline(text)[0]  

    scores = {res['label']: res['score'] for res in results}

    label_map = {
        "LABEL_0": "Negative 😞",
        "LABEL_1": "Neutral 😐",
        "LABEL_2": "Positive 😊"
    }

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    return label_map[best_label], round(best_score, 2)


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
