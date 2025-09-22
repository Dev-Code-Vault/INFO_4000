"""
MP1 Part 1 - Text Classification with Hugging Face

Trains a simple transformer classifier on your CSV of symptoms → disease labels.

Usage:
    python MP1_part1_text.py path/to/diseases.csv
"""

import sys
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
)
import evaluate
import numpy as np
import torch

# ========== Load CSV ==========
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "Medical_data.csv"
df = pd.read_csv(CSV_PATH)

print("CSV head:\n", df.head())

# normalize columns
df.columns = [c.strip().lower() for c in df.columns]

if "text" not in df.columns:
    raise ValueError("CSV must contain a 'text' column with symptom descriptions.")

if "label" not in df.columns:
    raise ValueError("CSV must contain a 'label' column (disease name or numeric id).")

# create string labels
df["label_text"] = df["label"].astype(str)

# label mapping
unique_labels = sorted(df["label_text"].unique().tolist())
label2id = {l: i for i, l in enumerate(unique_labels)}
id2label = {i: l for l, i in label2id.items()}

df["label_id"] = df["label_text"].map(label2id)

print("Unique labels:", unique_labels)

# train/eval split
train_df = df.sample(frac=0.85, random_state=42)
eval_df = df.drop(train_df.index)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
eval_ds = Dataset.from_pandas(eval_df.reset_index(drop=True))

print("Train size:", len(train_ds), "Eval size:", len(eval_ds))

# ========== Tokenizer + Model ==========
MODEL_NAME = "distilbert-base-uncased"  # bigger than tiny, still light
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(preprocess, batched=True)
eval_ds = eval_ds.map(preprocess, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])
eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id,
)

# ========== Training setup ==========
data_collator = DefaultDataCollator()
metric = evaluate.load("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)

args = TrainingArguments(
    output_dir="results_text",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    logging_dir="./logs_text",
    logging_steps=50,
    load_best_model_at_end=True,
    dataloader_pin_memory=False,
    # OLD-STYLE evaluation control:
    do_eval=True
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ========== Train ==========
print("🚀 Starting training...")
trainer.train()

# ========== Save ==========
model.save_pretrained("results_text")
tokenizer.save_pretrained("results_text")

print("✅ Model + tokenizer saved in 'results_text/'")

# ========== Evaluate ==========
results = trainer.evaluate()
print("📊 Test Accuracy:", results["eval_accuracy"])

# ========== Quick demo ==========
from transformers import pipeline
device = 0 if torch.cuda.is_available() else -1
clf = pipeline("text-classification", model="results_text", tokenizer="results_text", device=device)

example = "I have itchy, scaly skin patches on my arms and elbows."
print("Example prediction:", clf(example))
