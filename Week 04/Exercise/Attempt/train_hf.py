import torch
from torchvision import transforms
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoImageProcessor, AutoModelForImageClassification
import evaluate
import numpy as np

# 1. Load dataset
data_dir = "./CovidCT_Scan"
dataset = load_dataset("imagefolder", data_dir=data_dir)

# 2. Define labels
labels = dataset["train"].features["label"].names

# 3. Preprocessor
model_ckpt = "google/vit-base-patch16-224-in21k"  # Vision Transformer pretrained
processor = AutoImageProcessor.from_pretrained(model_ckpt)

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

# 8. Push to Hugging Face Hub
trainer.push_to_hub()
