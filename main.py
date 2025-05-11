import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch

# Load your CSV
df = pd.read_csv("DataFrame.csv")

# Convert polarity to sentiment labels
df['Sentiment'] = df['Polarity'].apply(lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral")
# Encode string labels to integers
le = LabelEncoder()
df["label"] = le.fit_transform(df["Sentiment"])  # Sentiment -> label column (0, 1, 2...)

# Rename the text column if necessary
df = df.rename(columns={"Summary": "text"})

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

# Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the text
def preprocess(example):
    return tokenizer(example["text"], padding=True, truncation=True)

tokenized_dataset = dataset.map(preprocess, batched=True)

# Load pre-trained model
num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",      # Enable evaluation
    save_strategy="epoch",            # Save model every epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1,
    load_best_model_at_end=True,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Evaluate
print(trainer.evaluate())

# Save model and tokenizer
model.save_pretrained("multiclass-sentiment-model")
tokenizer.save_pretrained("multiclass-sentiment-model")