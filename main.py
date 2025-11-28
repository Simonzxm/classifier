from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import csv
import os

app = FastAPI(title="Notification Classifier")

MODEL_DIR = "./model"
LOG_FILE = "logs.csv"

def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    sanitized_text = re.sub(url_pattern, '[URL_PLACEHOLDER]', text)
    email_pattern = r"(?:mailto:)?\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    sanitized_text = re.sub(email_pattern, '[EMAIL_PLACEHOLDER]', sanitized_text)
    return sanitized_text


class PredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    labels: List[str]
    scores: List[float]


# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, fix_mistral_regex=True)
except TypeError:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

label_map = {0: '非通知', 1: '通知'}


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    texts = [sanitize_text(t) for t in req.texts]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    for k, v in enc.items():
        enc[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        scores, preds = torch.max(probs, dim=1)
        preds = preds.cpu().tolist()
        scores = scores.cpu().tolist()

    # Log predictions
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for text, label in zip(req.texts, preds):
            writer.writerow([text, label])

    labels = [label_map[p] for p in preds]
    return PredictResponse(labels=labels, scores=scores)


@app.get('/', response_class=HTMLResponse)
def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get('/status')
def status():
    return {"status": "ok", "model_dir": MODEL_DIR}
