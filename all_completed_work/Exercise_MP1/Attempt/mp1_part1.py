# mp1_part1.py
# part1 training


# import libraries
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
import os

# main
def main():
    print("📄 Loading data...")
    
    #loading data
    df = pd.read_csv("Medical_data.csv")
    df.columns = [col.strip().lower() for col in df.columns]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Unique labels: {df['label'].nunique()}")
    
    #create label mappings
    unique_labels = sorted(df['label'].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    df['labels'] = df['label'].map(label2id)
    print(f"Labels: {unique_labels}")
    
    #split data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.85 * len(df_shuffled))
    train_df = df_shuffled.iloc[:train_size]
    val_df = df_shuffled.iloc[train_size:]
    
    #create datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'labels']])
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    #load tokenizer and model
    model_name = "distilbert-base-uncased"
    print(f"🤖 Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    #tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length',
            max_length=128
        )
    
    #tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    #simple metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy}
    
    #initialize training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=16,
        warmup_steps=50,
        logging_steps=50,
        save_steps=1000,  
        dataloader_pin_memory=False,
    )
    
    #initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )
    
    #start training
    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training completed!")
    
    #evaluate
    print("📊 Evaluating model...")
    eval_results = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    
    #save model
    os.makedirs('./medical_model', exist_ok=True)
    model.save_pretrained('./medical_model')
    tokenizer.save_pretrained('./medical_model')
    print("✅ Model saved to './medical_model'")
    
    #test
    print("\n🔍 Testing predictions:")
    
    #loading model
    classifier = pipeline(
        "text-classification", 
        model="./medical_model", 
        tokenizer="./medical_model"
    )
    
    #test cases
    test_examples = [
        "I'm experiencing a rapid heartbeat, shortness of breath, and chest pain.",
        "I've been feeling feverish and tired, with a persistent dry cough and difficulty breathing.",
        "I have a cough, runny nose, and a sore throat",
        "I have been experiencing a skin rash on my arms, legs. It is red, itchy, and dry.",
        "I have a fever.",
        "I have diarrhea"
    ]
    
    #print results
    print("Results for assignment test cases:")
    for i, example in enumerate(test_examples, 1):
        result = classifier(example)
        print(f"{i}. Symptoms: {example}")
        print(f"   Predicted: {result[0]['label']} (confidence: {result[0]['score']:.3f})")
        print()
    
    #manual validation
    print("🔍 Manual validation check on sample data:")
    val_sample = val_df.head(3)
    correct = 0
    for idx, row in val_sample.iterrows():
        prediction = classifier(row['text'])
        predicted_label = prediction[0]['label']
        true_label = row['label']
        is_correct = predicted_label == true_label
        correct += is_correct
        
        print(f"Text: {row['text'][:50]}...")
        print(f"True: {true_label} | Predicted: {predicted_label} {'✓' if is_correct else '✗'}")
        print("-" * 50)
    
    manual_acc = correct / len(val_sample)
    print(f"Manual accuracy on sample: {manual_acc:.3f}")
    
    #conclusion
    print("\n" + "="*60)
    print("✅ All done! Model trained successfully!")
    print("📁 Model saved in './medical_model/' directory")
    print("🎯 You can now use the model for new predictions!")

if __name__ == "__main__":
    main()