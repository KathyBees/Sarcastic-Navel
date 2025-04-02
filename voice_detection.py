import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import speech_recognition as sr
import pyttsx3

# === CONFIGURATION ===
sarcasm_model_path = "./fold_4/checkpoint-291"
llm_model = "gpt2"

# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained("roberta-large")  # load base tokenizer
sarcasm_model = AutoModelForSequenceClassification.from_pretrained(sarcasm_model_path)

# Load LLM
llm = pipeline("text-generation", model=llm_model)
#llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")


# Text-to-speech
tts = pyttsx3.init()

# === Microphone input ===
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(" Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f" You said: {text}")
        return text
    except sr.UnknownValueError:
        print(" Could not understand the audio.")
        return None

# === Sarcasm detection ===
def detect_sarcasm(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        logits = sarcasm_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        sarcasm_score = probs[0][1].item()  # 1 = sarcastic
        return sarcasm_score > 0.5, sarcasm_score

# === LLM response generation ===
def generate_response(text, sarcastic):
    prompt = (
        f"User: {text}\n"
        f"Sarcasm Detected: {sarcastic}\n"
        f"Assistant:"
    )

    response = llm(prompt, max_length=100, do_sample=True, top_k=50, top_p=0.95)[0]["generated_text"]

    # Clean extraction logic
    if "Assistant:" in response:
        return response.split("Assistant:")[-1].strip()
    else:
        return response.strip()

# === Speak the result ===
def speak(text):
    tts.setProperty('rate', 100)
    tts.say(text)
    tts.runAndWait()

# === Main ===
if __name__ == "__main__":
    while True:
        user_input = listen()
        if user_input:
            sarcastic, score = detect_sarcasm(user_input)
            print(f" Sarcasm detected: {sarcastic} (confidence: {score:.2f})")
            response = generate_response(user_input, sarcastic)
            print(f" Response: {response}")
            speak(response)
            break  # Exit after a successful run

        else:
            print(" Trying again...\n")