input_folder = "mp4_inputs"       # your folder with MUStARD++ .mp4 files
output_folder = "wav_outputs"     # where .wav files will go

import os, subprocess
os.makedirs(output_folder, exist_ok=True)

#for file in os.listdir(input_folder):
#    if file.endswith(".mp4"):
#        in_path = os.path.join(input_folder, file)
#        out_path = os.path.join(output_folder, file.replace(".mp4", ".wav"))
#        subprocess.call(["ffmpeg", "-i", in_path, "-ar", "16000", "-ac", "1", out_path])

import pandas as pd

df = pd.read_csv("mustard++_text.csv", sep=",")  # or ',' depending on format
print(df.columns.tolist())

df = df[df["Sarcasm"].isin([0, 1])]        # keep only labeled lines
df["filename"] = df["KEY"] + ".wav"
df = df[["filename", "SENTENCE", "Sarcasm"]]  # rename to match our pipeline
df.columns = ["filename", "transcript", "label"]
df.to_csv("sarcasm_dataset.csv", index=False)
