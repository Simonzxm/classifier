from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

app = FastAPI(title="Notification Classifier")

MODEL_DIR = "./model"


def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '[URL_PLACEHOLDER]', text)


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

    labels = [label_map[p] for p in preds]
    return PredictResponse(labels=labels, scores=scores)


@app.get('/')
def root():
    return {"status": "ok", "model_dir": MODEL_DIR}
