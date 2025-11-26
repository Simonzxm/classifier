import pandas as pd
import re
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

DATA_FILE = "data.csv"
MODEL_DIR = "./model"


def sanitize_text(text):
    if not isinstance(text, str):
        return ""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '[URL_PLACEHOLDER]', text)


def load_data():
    df = pd.read_csv(DATA_FILE)
    df['text'] = df['text'].apply(sanitize_text)
    return df


def prepare_eval(df):
    X = df['text']
    y = df['label']
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    eval_df = pd.DataFrame({'text': X_eval, 'label': y_eval}).reset_index(drop=True)
    return eval_df


def predict_batch(tokenizer, model, texts, device, batch_size=32):
    all_preds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i:i+batch_size])
            enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
            for k, v in enc.items():
                enc[k] = v.to(device)
            outputs = model(**enc)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
    return np.array(all_preds)


def main():
    print("Loading data...")
    df = load_data()
    eval_df = prepare_eval(df)
    print(f"Eval samples: {len(eval_df)}")

    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    texts = eval_df['text'].tolist()
    labels = eval_df['label'].astype(int).values

    print("Running predictions on eval set (batched)...")
    preds = predict_batch(tokenizer, model, texts, device, batch_size=16)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')
    cm = confusion_matrix(labels, preds)

    print("\nEvaluation results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (binary): {f1:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=['非通知','通知']))


if __name__ == '__main__':
    main()
