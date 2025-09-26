# train_hf.py
import torch
from datasets import load_dataset, DatasetDict
from transformers import Trainer, TrainingArguments, AutoImageProcessor, AutoModelForImageClassification
from transformers import DefaultDataCollator
import evaluate
import numpy as np
from PIL import Image

# use raw strings
train_path = r"C:\Users\siona\Desktop\Data\Info_4000\train-20240112T210350Z-001\train"
test_path = r"C:\Users\siona\Desktop\Data\Info_4000\test-20240112T210346Z-001\test"

#load datasets
train_dataset = load_dataset("imagefolder", data_dir=train_path)["train"]
test_dataset = load_dataset("imagefolder", data_dir=test_path)["train"]

print("Dataset features:", train_dataset.features)
print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))

#check first example
print("First example keys:", train_dataset[0].keys())

#define labels
labels = train_dataset.features["label"].names
print("Labels:", labels)

#load pretrained processor and model
model_ckpt = "microsoft/resnet-18"
processor = AutoImageProcessor.from_pretrained(model_ckpt)

#preprocessing function
def preprocess_examples(examples):
    #load images
    images = examples["image"]
    labels = examples["label"]

    if not isinstance(images, list):
        images = [images]
        labels = [labels]

    #convert to RGB
    images = [img.convert("RGB") for img in images]
    
    #process with the processor
    processed = processor(images, return_tensors="pt")
    
    #add labels
    processed["labels"] = labels
    
    return processed

#apply preprocessing to datasets
train_dataset = train_dataset.map(
    preprocess_examples,
    batched=True,
    batch_size=100,
    remove_columns=train_dataset.column_names
)

test_dataset = test_dataset.map(
    preprocess_examples,
    batched=True,
    batch_size=100,
    remove_columns=test_dataset.column_names
)

#set format for PyTorch
train_dataset.set_format("torch")
test_dataset.set_format("torch")

print("After preprocessing - train features:", train_dataset.features)

#load model with correct label mapping
model = AutoModelForImageClassification.from_pretrained(
    model_ckpt,
    num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)},
    label2id={l: i for i, l in enumerate(labels)},
    ignore_mismatched_sizes=True
)

#use default data collator
data_collator = DefaultDataCollator()

#accuracy metric
metric = evaluate.load("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)

#training arguments
args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    num_train_epochs=2,
    save_strategy="epoch",
    push_to_hub=False,
    logging_dir="./logs",
    logging_steps=100,
    dataloader_pin_memory=False,  # Disable pin_memory to avoid warning
)

#trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

#train
print("Starting training...")
trainer.train()

#save model & processor locally
model.save_pretrained("results")
processor.save_pretrained("results")

print("✅ Model trained and saved in 'results/' folder.")

#evaluateeeee, for debug
results = trainer.evaluate()
print("Test Accuracy:", results["eval_accuracy"])


"""
Questions:
1. What specific preprocessing did you do on the data?

    I loaded all the images, converted them to RGB, and used a pretrained 
    image processor to resize and normalize them so they fit the model's 
    input format. I also mapped the class labels properly for training.


2. Which model did you use and what, if any, were the modifications you made to the
pretrained model and why?

    I used a pretrained ResNet-18 model from Microsoft and
    I replaced its final layer to match the number of classes
    in my dataset and set up the label mappings so predictions return the correct class names. 
    No other major changes were made.

3. What were the performance metrics of the model and how many predictions did you
get correct?

    The model reached 97.2 percent accuracy on the test set and got 243 correct predictions.


"""