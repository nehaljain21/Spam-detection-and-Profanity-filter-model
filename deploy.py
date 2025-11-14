from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback
import joblib

app = FastAPI(title="E-Commerce Recommendation API")

spam_model = joblib.load("spam_model.joblib")
spam_vectorizer = joblib.load("spam_vectorizer.joblib")
profanity_model = joblib.load("profanity_model.joblib")
profanity_vectorizer = joblib.load("profanity_vectorizer.joblib")

class CommentInput(BaseModel):
    comment: str

import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.post("/predict/")
def predict_comment(data: CommentInput):
    try:
        text = data.comment
        cleaned = clean_text(text)

        spam_vec = spam_vectorizer.transform([cleaned])
        spam_prob = spam_model.predict_proba(spam_vec)[0][1]
        is_spam = spam_prob >= 0.7

        prof_vec = profanity_vectorizer.transform([cleaned])
        prof_prob = profanity_model.predict_proba(prof_vec)[0][1]
        is_profane = prof_prob >= 0.7

        if is_spam and is_profane:
            label = "Spam & Profane"
        elif is_spam:
            label = "Spam"
        elif is_profane:
            label = "Profane"
        else:
            label = "Clean"

        return {
        "input_comment": text,
        "cleaned_comment": cleaned,
        "predicted_label": label,
        "spam_probability": round(float(spam_prob), 3),
        "profanity_probability": round(float(prof_prob), 3)
        }
    
    except Exception as e:
        print("Error Traceback:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def home():
    return {"message": "Comment Moderation API is live and running!"}
