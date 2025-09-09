import os
import json
import threading
from transformers import pipeline
import pyttsx3
from patterns import detect_fact
from trainer_utils import fine_tune

MEMORY_FILE = "memory.txt"
KB_FILE = "knowledge.json"
CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "train_interval": 20,
        "max_train_steps": 500,
        "speech_rate": 170,
        "model_dir": "fine-tuned",
        "base_model": "gpt2-medium"
    }

config = load_config()

# Init speech
engine = pyttsx3.init()
engine.setProperty("rate", config.get("speech_rate", 170))

def speak(text):
    print(f"Learny: {text}")
    engine.say(text)
    engine.runAndWait()

def background_train():
    thread = threading.Thread(
        target=fine_tune,
        args=(MEMORY_FILE, config["model_dir"], config["max_train_steps"])
    )
    thread.daemon = True
    thread.start()

def find_fact_in_kb(user_input, kb):
    lowered = user_input.lower()
    for key, value in kb.items():
        if key in lowered:
            return f"Your {key} is {value}."
    return None

def load_kb():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_kb(kb):
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

def main():
    print("🚀 Starting Learny...")
    model_path = config["model_dir"] if os.path.exists(config["model_dir"]) else config["base_model"]
    generator = pipeline("text-generation", model=model_path)
    print(f"✅ Model loaded: {model_path}")

    kb = load_kb()
    speak("Hello! How are you doing today?")

    convo_count = 0
    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            speak("Goodbye. Saving memory and knowledge.")
            save_kb(kb)
            break

        history.append(f"You: {user_input}")
        convo_count += 1

        # Detect new fact
        fact = detect_fact(user_input)
        if fact:
            key, value = fact
            kb[key] = value
            save_kb(kb)
            reply = f"Got it! I’ll remember your {key} is {value}."
            speak(reply)
            history.append(f"Learny: {reply}")
        else:
            # Try KB first
            fact_reply = find_fact_in_kb(user_input, kb)
            if fact_reply:
                reply = fact_reply
            else:
                # GPT-2 generation
                prompt = "\n".join(history[-10:]) + "\nLearny:"
                out = generator(
                    prompt,
                    max_new_tokens=150,
                    num_return_sequences=1,
                    pad_token_id=50256,
                    truncation=True
                )
                reply = out[0]["generated_text"][len(prompt):].split("\n")[0].strip()
                if not reply:
                    reply = "Hmm, can you say that differently?"

            speak(reply)
            history.append(f"Learny: {reply}")

        # Save conversation
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(history[-2:]) + "\n")

        # Background training
        if convo_count % config["train_interval"] == 0:
            speak("Let me learn a bit from our chats...")
            background_train()

if __name__ == "__main__":
    main()
