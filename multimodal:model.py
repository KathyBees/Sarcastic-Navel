import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import librosa
import torch.nn as nn

# === CONFIGURATION ===
audio_folder = "wav_outputs"  # Folder where .wav files are located
csv_file = "sarcasm_dataset.csv"  # Your CSV with text, label, and audio filenames

# === Load CSV and prepare dataset ===
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()  # Strip whitespace from column names
df = df[df["label"].isin([0, 1])]  # Keep only rows with valid labels

# === Audio Feature Extraction ===
def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc.mean(axis=1)
    except FileNotFoundError:
        print(f"Warning: Missing file {file_path}, skipping this sample.")
        return None  # Return None if the file is missing

# === Dataset class ===
class SarcasmDataset(Dataset):
    def __init__(self, df, audio_folder, tokenizer):
        self.df = df
        self.audio_folder = audio_folder
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_folder, row["filename"])
        
        # Extract audio features
        audio_feats = extract_audio_features(audio_path)
        if audio_feats is None:
            return None  # Skip the sample if audio is missing
        
        # Tokenize text
        text_inputs = self.tokenizer(
            row["transcript"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )

        # Return a dict with text and audio features
        item = {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "audio_features": torch.tensor(audio_feats, dtype=torch.float32),
            "label": torch.tensor(row["label"], dtype=torch.long)
        }
        return item

    def __len__(self):
        return len(self.df)

# === Model class ===
class MultimodalModel(nn.Module):
    def __init__(self, hidden_size=768, audio_size=13):
        super().__init__()
        # Use BERT for text processing
        self.text_model = AutoModel.from_pretrained("bert-base-uncased")
        
        # Project audio features to match BERT hidden size (768)
        self.audio_proj = nn.Linear(audio_size, hidden_size)
        
        # Classifier combining both text and audio features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),  # 768 (text) + 768 (audio) = 1536
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, audio_features):
        # Process text through BERT and get the hidden states
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_out = outputs.last_hidden_state[:, 0, :]  # Use the hidden state of the [CLS] token
        
        # Process audio features
        audio_out = self.audio_proj(audio_features)
        
        # Concatenate text and audio features (make sure they are the same size)
        combined = torch.cat((text_out, audio_out), dim=1)  # This should give a shape of (batch_size, 1536)
        
        return self.classifier(combined)

# === Train Model with Cross-Validation ===
def train_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()  # Strip whitespace from column names
    
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    n_splits = kf.get_n_splits(df)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, df["label"])):
        print(f"🔁 Fold {fold+1}/{n_splits}")
        
        # Prepare train and validation data
        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]
        
        train_dataset = SarcasmDataset(train_data, audio_folder, tokenizer)
        val_dataset = SarcasmDataset(val_data, audio_folder, tokenizer)

        # Filter out None samples (those with missing audio files)
        train_dataset = [sample for sample in train_dataset if sample is not None]
        val_dataset = [sample for sample in val_dataset if sample is not None]

        print(len(train_dataset))
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        # Model, optimizer, loss function
        model = MultimodalModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(5):  # Change to desired number of epochs
            model.train()
            all_preds = []
            all_labels = []
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                audio_features = batch["audio_features"]
                labels = batch["label"]
                
                outputs = model(input_ids, attention_mask, audio_features)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                # Collect predictions and true labels
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)

            print(f"Epoch {epoch+1}: Loss = {loss.item()}")
            print(f"Epoch {epoch+1}: Accuracy = {accuracy:.4f}")
            print(f"Epoch {epoch+1}: Precision = {precision:.4f}")
            print(f"Epoch {epoch+1}: Recall = {recall:.4f}")
            print(f"Epoch {epoch+1}: F1 Score = {f1:.4f}")
        
        # Save model for each fold
        torch.save(model.state_dict(), f"saved_models/model_fold{fold+1}.pt")
        print(f"Model for fold {fold+1} saved!")

# === Run the training function ===
if __name__ == "__main__":
    train_model()
