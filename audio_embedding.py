# === Configuration ===
audio_folder = "wav_outputs"
csv_file = "sarcasm_dataset.csv"

import torch
import torch.nn as nn
import torchaudio
import os
import csv
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
    BertTokenizer,
    BertModel
)
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

class SarcasmDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.fusion = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )

    def forward(self, text, audio_waveform):
        tokens = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_output = self.text_model(**tokens).pooler_output

        if audio_waveform.ndim == 1:
            audio_waveform = audio_waveform.unsqueeze(0)  # [1, length]
        elif audio_waveform.ndim == 2 and audio_waveform.shape[0] > 1:
            audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)  # Convert stereo to mono

        # Ensure correct input shape for Wav2Vec2Model: [batch_size, seq_len]
        if audio_waveform.ndim == 2:
            input_values = audio_waveform
        elif audio_waveform.ndim == 1:
            input_values = audio_waveform.unsqueeze(0)

        audio_output = self.wav_model(input_values).last_hidden_state.mean(dim=1)

        combined = torch.cat((text_output, audio_output), dim=1)
        logits = self.fusion(combined)
        return logits
    
def load_audio(file_path, target_sr=16000):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0)

def predict_sarcasm(text, audio_path, model):
    waveform = load_audio(audio_path)
    model.eval()
    with torch.no_grad():
        logits = model(text, waveform)
        probs = torch.softmax(logits, dim=1)
        label = torch.argmax(probs).item()
        confidence = probs[0, label].item()
    return label, confidence

class SarcasmDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, audio_path, label = self.samples[idx]
        waveform = load_audio(audio_path)
        return text, waveform, label

def train_crossfold(model_class, samples, k=5, epochs=20, batch_size=1, lr=1e-4, patience=3):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(samples)):
        print(f"Fold {fold + 1}")
        train_subset = [samples[i] for i in train_idx]
        val_subset = [samples[i] for i in val_idx]
        train_loader = DataLoader(SarcasmDataset(train_subset), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(SarcasmDataset(val_subset), batch_size=1)
        model = model_class()

        # Freeze BERT and Wav2Vec2 to save memory
        model.text_model.eval()
        model.wav_model.eval()
        for param in model.text_model.parameters():
            param.requires_grad = False
        for param in model.wav_model.parameters():
            param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        best_val_acc = 0.0
        epochs_no_improve = 0
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for text, waveform, label in train_loader:
                logits = model(text[0], waveform.squeeze(0))
                loss = criterion(logits, torch.tensor([label]).long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1}: Avg Train Loss: {avg_train_loss:.4f}")
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for text, waveform, label in val_loader:
                    logits = model(text[0], waveform)
                    pred = torch.argmax(logits, dim=1).item()
                    y_pred.append(pred)
                    y_true.append(label)
            val_acc = accuracy_score(y_true, y_pred)
            val_precision = precision_score(y_true, y_pred, zero_division=0)
            val_recall = recall_score(y_true, y_pred, zero_division=0)
            val_f1 = f1_score(y_true, y_pred, zero_division=0)
            print(f"Validation Accuracy: {val_acc:.4f}")
            print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                print("Validation accuracy improved. Saving model.")
                torch.save(model.state_dict(), f"sarcasm_fold{fold + 1}_best.pt")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
        print(f"Best Validation Accuracy for Fold {fold + 1}: {best_val_acc:.4f}\n")

if __name__ == "__main__":
    samples = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text') or row.get('utterance') or row.get('transcript')
            audio_file = os.path.join(audio_folder, row['filename'])
            label = int(float(row['label']))
            if not os.path.isfile(audio_file):
                print(f"Skipping missing audio file: {audio_file}")
                continue
            samples.append((text, audio_file, label))
    train_crossfold(SarcasmDetector, samples)

