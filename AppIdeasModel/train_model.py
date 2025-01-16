import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Load dataset using pandas
DATASET_PATH = "ideas_dataset.json"
df = pd.read_json(DATASET_PATH)

# Inspect the dataset
print("Dataset preview:")
print(df.head())

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize_row(row):
    combined_text = f"{row['idea']} {row['explanation']}"
    return tokenizer(
        combined_text,
        truncation=True,      # Truncate sequences longer than max length
        max_length=128,       # Set a max token length
        padding="max_length"  # Pad sequences to max length
    )

# Apply tokenization to the dataset
print("Tokenizing the dataset...")
tokenized_data = df.apply(tokenize_row, axis=1)

# Convert tokenized data into lists for Hugging Face Dataset
input_ids = [x["input_ids"] for x in tokenized_data]
attention_mask = [x["attention_mask"] for x in tokenized_data]
labels = df["label"].tolist()

# Create Hugging Face Dataset
hf_dataset = Dataset.from_dict({
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "label": labels,
})

# Split dataset into train and test sets
split_data = hf_dataset.train_test_split(test_size=0.2)
train_dataset = split_data["train"]
test_dataset = split_data["test"]

print("Sample from train dataset:")
print(train_dataset[0])

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting training...")
trainer.train()
print("Training complete!")

# Save the model and tokenizer
model.save_pretrained("./app_ideas_model")
tokenizer.save_pretrained("./app_ideas_model")
print("Model and tokenizer saved to './app_ideas_model'.")
