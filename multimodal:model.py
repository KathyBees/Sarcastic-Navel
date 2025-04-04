import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import librosa
from datetime import datetime
import torch.nn as nn
import os
import csv

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
class MultimodalModel(nn.Module):
    def __init__(self, hidden_size=768, audio_size=13):
        super().__init__()
        self.text_model = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.2)

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, audio_features):
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_out = self.dropout(text_out)

        audio_out = self.audio_encoder(audio_features)
        audio_out = self.dropout(audio_out)

        combined = torch.cat((text_out, audio_out), dim=1)
        gate = self.fusion_gate(combined)
        fused = gate * text_out + (1 - gate) * audio_out

        final_input = torch.cat((fused, combined), dim=1)
        return self.classifier(final_input)

# === Train Model with Cross-Validation ===
def train_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()  # Strip whitespace from column names
    
    kf = StratifiedKFold(n_splits=7, shuffle=True)
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

                # File to save logs for this fold
        start_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/metrics_fold{fold+1}_{start_date}.txt"
        
        with open(log_file, "w") as f_log:
            f_log.write("Epoch\tLoss\tAccuracy\tPrecision\tRecall\tF1\n")

        for epoch in range(5):
            # === Training ===
            model.train()
            all_preds, all_labels = [], []
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(batch["input_ids"], batch["attention_mask"], batch["audio_features"])
                loss = loss_fn(outputs, batch["label"])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["label"].cpu().numpy())
            avg_loss = epoch_loss / len(train_loader)
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            #train_losses.append(avg_loss)

            # === Validation ===
            model.eval()
            val_preds, val_labels = [], []
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(batch["input_ids"], batch["attention_mask"], batch["audio_features"])
                    loss = loss_fn(outputs, batch["label"])
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(batch["label"].cpu().numpy())
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds)
            val_recall = recall_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)
            #val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            print(f"Train Acc={accuracy:.4f}, Val Acc={val_accuracy:.4f}")

            with open(log_file, "a", newline="") as f_log:
                writer = csv.writer(f_log, delimiter='\t')
                writer.writerow([epoch+1, avg_loss, accuracy, precision, recall, f1,
                                 avg_val_loss, val_accuracy, val_precision, val_recall, val_f1])


        
        # Save model for each fold
        torch.save(model.state_dict(), f"saved_models/model_fold{fold+1}.pt")
        print(f"Model for fold {fold+1} saved!")

# === Run the training function ===
if __name__ == "__main__":
    train_model()
