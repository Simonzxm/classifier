import csv
import json
import requests

LOG_FILE = "data.csv"
OUTPUT_FILE = "out.csv"

with open(LOG_FILE, mode='r', newline='') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    texts = [row[0] for row in reader]

result = requests.post("http://127.0.0.1:8001/predict", json={"texts": texts})
labels = result.json()["labels"]
scores = result.json()["scores"]

with open(OUTPUT_FILE, mode='w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(["text", "label", "score"])
    for text, label, score in zip(texts, labels, scores):
        if label == "通知" or score < 0.9:
            writer.writerow([text, label, score])