# train_hf.py

import torch
from torchvision import transforms
from datasets import load_dataset, DatasetDict
from transformers import Trainer, TrainingArguments, AutoImageProcessor, AutoModelForImageClassification
import evaluate
import numpy as np

# Use raw strings (r"") to avoid Windows path issues
train_path = r"C:\Users\siona\Desktop\Data\Info_4000\train-20240112T210350Z-001\train"
test_path = r"C:\Users\siona\Desktop\Data\Info_4000\test-20240112T210346Z-001\test"

# Load datasets
train_dataset = load_dataset("imagefolder", data_dir=train_path, split="train")
test_dataset = load_dataset("imagefolder", data_dir=test_path, split="train")

# Combine into one dataset dictionary
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# 2. Define labels
labels = dataset["train"].features["label"].names

# 3. Preprocessor
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

def transform(example):
    img = example["image"].convert("RGB")  # ensure 3 channels
    inputs = processor(img, return_tensors="pt")
    inputs["label"] = example["label"]
    return inputs

dataset = dataset.with_transform(transform)

# 4. Data collator
def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"].squeeze() for x in batch])
    labels = torch.tensor([x["label"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# 5. Load model
model = AutoModelForImageClassification.from_pretrained(
    model_ckpt,
    num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)},
    label2id={l: i for i, l in enumerate(labels)}
)

# 6. Metrics
metric = evaluate.load("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)

# 7. Training
args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    num_train_epochs=2,   # only 1–2 epochs needed
    save_strategy="epoch",
    push_to_hub=True,
    hub_model_id="your-username/covid-ct-scan-classifier"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

trainer.train()

# Save model and processor locally
model.save_pretrained("results")
processor.save_pretrained("results")

# 8. Push to Hugging Face Hub
trainer.push_to_hub()
