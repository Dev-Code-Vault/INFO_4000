import torch
from datasets import load_dataset, DatasetDict
from transformers import Trainer, TrainingArguments, AutoImageProcessor, AutoModelForImageClassification
import evaluate
import numpy as np


# Use raw strings to avoid path issues
train_path = r"C:\Users\siona\Desktop\Data\Info_4000\train-20240112T210350Z-001\train"
test_path = r"C:\Users\siona\Desktop\Data\Info_4000\test-20240112T210346Z-001\test"


#test test debug 
from pathlib import Path
# Debug: show some filenames
print("Sample training files:")
print(list(Path(train_path).rglob("*.jpg"))[:5])



# Load datasets
train_dataset = load_dataset("imagefolder", data_dir=train_path)["train"]
test_dataset = load_dataset("imagefolder", data_dir=test_path)["train"]


# Combine into one dataset dictionary
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# DEBUG: Inspect label key
print("Dataset features:", dataset["train"].features)

# Assuming label key is "label" (change to "class" if needed)
label_key = "label"

# Define labels
labels = dataset["train"].features[label_key].names

# Load pretrained processor and model
model_ckpt = "microsoft/resnet-18"
processor = AutoImageProcessor.from_pretrained(model_ckpt)

# Transform function
def transform(example):
    img = example["image"].convert("RGB")
    inputs = processor(img, return_tensors="pt")
    inputs["label"] = example[label_key]
    return inputs

# Apply transform
dataset = dataset.with_transform(transform)

# Data collator
def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"].squeeze(0) for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# Load model with correct label mapping
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-18",
    num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)},
    label2id={l: i for i, l in enumerate(labels)},
    ignore_mismatched_sizes=True  # <--- add this line!
)


# Accuracy metric
metric = evaluate.load("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)

#debug
print("TrainingArguments source:", TrainingArguments.__module__)


# Training arguments
args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    num_train_epochs=2,
    save_strategy="epoch",
    push_to_hub=False,  # CHANGE: Only do this if you want to push!
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model & processor locally
model.save_pretrained("results")
processor.save_pretrained("results")

print("✅ Model trained and saved in 'results/' folder.")
