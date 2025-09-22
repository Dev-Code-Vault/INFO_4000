"""
MP1 Part 1 - Simplified Medical Text Classification
Trains a transformer classifier to diagnose diseases based on symptoms.

Usage: python mp1_part1.py
"""

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    pipeline
)
from sklearn.metrics import accuracy_score
import torch

def load_and_prepare_data(csv_path="Medical_data.csv"):
    """Load and prepare the medical data"""
    print("📄 Loading data...")
    df = pd.read_csv(csv_path)
    
    # Clean column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Check required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Unique labels: {df['label'].nunique()}")
    
    return df

def create_label_mappings(df):
    """Create label mappings"""
    unique_labels = sorted(df['label'].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    df['label_id'] = df['label'].map(label2id)
    
    print(f"Labels: {unique_labels}")
    return df, label2id, id2label

def prepare_datasets(df):
    """Split data into train/validation sets"""
    # Simple train-validation split
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    return train_dataset, val_dataset

def tokenize_data(dataset, tokenizer):
    """Tokenize the text data"""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=128
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def compute_metrics(eval_pred):
    """Compute accuracy metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

def main():
    # Load and prepare data
    df = load_and_prepare_data()
    df, label2id, id2label = create_label_mappings(df)
    
    # Create datasets
    train_dataset, val_dataset = prepare_datasets(df)
    
    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    print(f"🤖 Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    # Tokenize datasets
    train_dataset = tokenize_data(train_dataset, tokenizer)
    val_dataset = tokenize_data(val_dataset, tokenizer)
    
    # Set format for PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label_id'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label_id'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("🚀 Starting training...")
    trainer.train()
    
    # Save the model
    model.save_pretrained('./medical_model')
    tokenizer.save_pretrained('./medical_model')
    print("✅ Model saved to './medical_model'")
    
    # Test with example predictions
    test_examples = [
        "I'm experiencing a rapid heartbeat, shortness of breath, and chest pain.",
        "I've been feeling feverish and tired, with a persistent dry cough and difficulty breathing.",
        "I have a cough, runny nose, and a sore throat",
        "I have been experiencing a skin rash on my arms, legs. It is red, itchy, and dry.",
        "I have a fever.",
        "I have diarrhea"
    ]
    
    print("\n🔍 Testing predictions:")
    classifier = pipeline("text-classification", 
                         model="./medical_model", 
                         tokenizer="./medical_model")
    
    for example in test_examples:
        result = classifier(example)
        print(f"Symptoms: {example}")
        print(f"Predicted: {result[0]['label']} (confidence: {result[0]['score']:.3f})")
        print("-" * 50)

if __name__ == "__main__":
    main()