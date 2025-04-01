import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset
print("Loading dataset...")
df = pd.read_csv("mustard++_text.csv")
df = df.dropna(subset=["SENTENCE", "Sarcasm"])  # Ensure clean data
texts = df["SENTENCE"].astype(str).values
labels = df["Sarcasm"].fillna(0).astype(int).values

# Split into train and test first
texts_train, texts_test, labels_train, labels_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# Tokenizer and model name
model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Metric function
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Dataset wrapper
class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

# Cross-validation setup on training data only
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(texts_train, labels_train)):
    print(f"\n Fold {fold+1}/5")

    X_train, X_val = texts_train[train_idx], texts_train[val_idx]
    y_train, y_val = labels_train[train_idx], labels_train[val_idx]

    # Tokenization
    train_enc = tokenizer(list(X_train), truncation=True, padding=True, max_length=128, return_tensors="pt")
    val_enc = tokenizer(list(X_val), truncation=True, padding=True, max_length=128, return_tensors="pt")

    train_enc["labels"] = torch.tensor(y_train)
    val_enc["labels"] = torch.tensor(y_val)

    train_dataset = SarcasmDataset(train_enc)
    val_dataset = SarcasmDataset(val_enc)

    # Initialize model from scratch each fold
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"./fold_{fold}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"./logs/fold_{fold}",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print(f" Finished fold {fold+1}")

# Optional: Save test set for future use
test_df = pd.DataFrame({"text": texts_test, "label": labels_test})
test_df.to_csv("mustard_test_holdout.csv", index=False)
print(" Saved holdout test set to mustard_test_holdout.csv")