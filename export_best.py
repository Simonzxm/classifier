from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

ckpt = "./results/checkpoint-154"
out = "./model"

print(f"Loading model from {ckpt} ...")
model = AutoModelForSequenceClassification.from_pretrained(ckpt, ignore_mismatched_sizes=True)
try:
    # Try loading tokenizer with fix for known mistral regex warning
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
except Exception:
    tokenizer = None

print(f"Saving model to {out} ...")
model.save_pretrained(out)
if tokenizer:
    tokenizer.save_pretrained(out)

print("Export complete.")
