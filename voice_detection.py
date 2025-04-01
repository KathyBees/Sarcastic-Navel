import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import speech_recognition as sr
import pyttsx3

# Load sarcasm detection model
model_path = "./fold_0"  # Or the best performing fold directory
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load LLM for generating responses
llm = pipeline("text-generation", model="gpt2")

# Text-to-speech engine
tts = pyttsx3.init()

def listen_to_microphone():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(" Speak now...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(f" You said: {text}")
        return text
    except sr.UnknownValueError:
        print(" Could not understand audio.")
        return None

def detect_sarcasm(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        sarcasm_score = probs[0][1].item()  # Index 1 = sarcastic
        return sarcasm_score > 0.5, sarcasm_score

def respond(text, sarcastic):
    prompt = prompt = f"""You are an emotionally intelligent AI. Interpret the user's mood. If sarcasm is detected, offer a thoughtful or witty response. User: {text} Sarcasm: {sarcastic} Reply:"""
    output = llm(prompt, max_length=60, do_sample=True)[0]["generated_text"]
    return output.split('Response:')[-1].strip()

# Main loop
user_input = listen_to_microphone()
if user_input:
    sarcastic, score = detect_sarcasm(user_input)
    print(f" Sarcasm detected: {sarcastic} (score={score:.2f})")
    response = respond(user_input, sarcastic)
    print(f" LLM response: {response}")
    tts.say(response)
    tts.runAndWait()