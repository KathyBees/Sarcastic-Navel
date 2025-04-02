import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import speech_recognition as sr
import pyttsx3

# === CONFIGURATION ===
sarcasm_model_path = "./fold_4/checkpoint-291"
#llm_model_name = "microsoft/Phi-3-mini-4k-instruct"

# Load sarcasm detection model and tokenizer
sarcasm_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
sarcasm_model = AutoModelForSequenceClassification.from_pretrained(sarcasm_model_path)

llm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
pipe = pipeline("text-generation", model=llm_model_name)


# Text-to-speech
tts = pyttsx3.init()

# === Microphone input ===
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None

# === Sarcasm detection ===
def detect_sarcasm(text):
    inputs = sarcasm_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        logits = sarcasm_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        sarcasm_score = probs[0][1].item()  # class 1 = sarcastic
        return sarcasm_score > 0.5, sarcasm_score

def generate_response(text, sarcastic):
    prompt = (
        f"User: {text}\n"
        f"Sarcasm Detected: {sarcastic}\n"
        f"Assistant:"
    )

    response = pipe(
        prompt, 
        max_new_tokens=100, 
        do_sample=True,         # Enable sampling explicitly
        temperature=0.7, 
        top_p=0.9
    )[0]["generated_text"]

    # Extract assistant reply cleanly
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1]
    return response.strip()


# === Speak the result ===
def speak(text):
    tts.setProperty('rate', 120)
    tts.say(text)
    tts.runAndWait()

# === Main Loop ===
if __name__ == "__main__":
    while True:
        user_input = listen()
        if user_input:
            sarcastic, score = detect_sarcasm(user_input)
            print(f"Sarcasm detected: {sarcastic} (confidence: {score:.2f})")
            response = generate_response(user_input, sarcastic)
            print(f"Assistant: {response}")
            speak(response)
        else:
            print("Trying again...\n")