import os
import json
import threading
import random
import sys
import select
# import time
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
            # Read lines and filter out JSON comments for robust parsing
            lines = f.readlines()
            valid_json = "".join(line for line in lines if not line.strip().startswith("//"))
            return json.loads(valid_json)
    return {
        "train_interval": 20,
        "max_train_steps": 500,
        "speech_rate": 170,
        "model_dir": "fine-tuned",
        "base_model": "gpt2-medium"
    }

config = load_config()

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", config.get("speech_rate", 170))

def speak(text):
    """Prints the text and speaks it out loud."""
    print(f"Learny: {text}")
    engine.say(text)
    engine.runAndWait()

def background_train():
    """Starts the fine-tuning process in a background thread."""
    thread = threading.Thread(
        target=fine_tune,
        args=(MEMORY_FILE, config["model_dir"], config["max_train_steps"])
    )
    thread.daemon = True
    thread.start()

def load_kb():
    """Loads the knowledge base from a JSON file."""
    if os.path.exists(KB_FILE) and os.path.getsize(KB_FILE) > 0:
        with open(KB_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {} # Return empty dict if file is malformed
    return {}

def save_kb(kb):
    """Saves the knowledge base to a JSON file."""
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

def find_fact_in_kb(user_input, kb):
    """Checks if the user input mentions a known fact."""
    lowered = user_input.lower()
    for key, value in kb.items():
        if key in lowered:
            return f"I remember you told me your {key} is {value}."
    return None

def ask_idle_question(kb, history):
    """Asks a question to re-engage an idle user."""
    reengage_questions = []
    if "name" not in kb:
        reengage_questions.append("I don't think I know your name yet. What is it?")
    if "hobby" not in kb:
        reengage_questions.append("Do you have any hobbies?")
    reengage_questions.append("What's on your mind?")
    reengage_questions.append("Is everything okay?")
    
    reply = random.choice(reengage_questions)
    speak(reply)
    history.append(f"Learny: {reply}")

def main():
    print("🚀 Starting Learny...")
    model_path = config["model_dir"] if os.path.exists(config["model_dir"]) else config["base_model"]
    generator = pipeline("text-generation", model=model_path)
    print(f"✅ Model loaded: {model_path}")

    kb = load_kb()
    speak("Hi! I’m Learny. I don’t know much yet, but I’d love to learn about you!")

    convo_count = 0
    history = []

    while True:
        # Wait for user input for 10 seconds
        print("You: ", end="", flush=True)
        rlist, _, _ = select.select([sys.stdin], [], [], 10)
        
        should_save_turn = True # Flag to control saving to memory.txt

        if not rlist:
            # Timeout occurred, user was idle. Ask a question.
            ask_idle_question(kb, history)
            should_save_turn = False # Do not save this turn to memory
            continue

        # User provided input
        user_input = sys.stdin.readline().strip()

        if user_input.lower() in ["exit", "quit"]:
            speak("Goodbye! It was great chatting with you. Let’s talk again soon!")
            save_kb(kb)
            break

        history.append(f"You: {user_input}")
        convo_count += 1

        # 1. Explicitly handle when asked for its name
        if "what is your name" in user_input.lower() or "who are you" in user_input.lower():
            reply = "My name is Learny, and I am your friendly bot!"
            speak(reply)
            history.append(f"Learny: {reply}")
            continue

        # 2. Detect new fact
        fact = detect_fact(user_input)
        if fact:
            key, value = fact
            kb[key] = value
            save_kb(kb)
            
            # Special response for when it learns the name
            if key == "name":
                reply = f"Nice to meet you, {value}!"
            else:
                reply = f"Got it! I’ll remember your {key} is {value}. Thanks for sharing!"
            
            speak(reply)
            history.append(f"Learny: {reply}")
        else:
            # 3. Try KB first
            fact_reply = find_fact_in_kb(user_input, kb)
            if fact_reply:
                reply = fact_reply
            else:
                # 4. GPT-2 generation
                prompt = "\n".join(history[-10:]) + "\nLearny:"
                out = generator(
                    prompt, max_new_tokens=150, num_return_sequences=1, pad_token_id=50256, truncation=True
                )
                reply = out[0]["generated_text"][len(prompt):].split("\n")[0].strip()
                if not reply:
                    reply = "Hmm, can you say that differently?"

            speak(reply)
            history.append(f"Learny: {reply}")

        # Save conversation only if it was a standard interaction
        if should_save_turn:
            with open(MEMORY_FILE, "a", encoding="utf-8") as f:
                f.write("\n".join(history[-2:]) + "\n")

        # Background training
        if convo_count % config["train_interval"] == 0:
            speak("Let me think about our chat for a moment to learn more...")
            background_train()

if __name__ == "__main__":
    main()