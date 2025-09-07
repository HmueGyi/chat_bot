# chat_bot.py
from models_utils import load_policy_model, generate_candidates
import pyttsx3

# ---------- Chat history functions ----------
def save_history(user_input, bot_response, file_path="chat_history.txt"):
    with open(file_path, "a") as f:
        f.write(f"You: {user_input}\nLearny: {bot_response}\n\n")

def load_last_n_lines(file_path="chat_history.txt", n=6):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        return "".join(lines[-n*2:])  # 2 lines per turn: user + bot
    except FileNotFoundError:
        return ""

# ---------- Load fine-tuned GPT-2 model ----------
tokenizer, model = load_policy_model("./learny_model")

# ---------- Initialize TTS ----------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # female voice

print("Chat with Learny! Type 'exit' to quit.\n")

# ---------- Main chat loop ----------
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Load last N lines for context
    context = load_last_n_lines()
    input_text = context + f"You: {user_input}\nLearny:"

    # Generate GPT-2 response
    response = generate_candidates(tokenizer, model, input_text)

    # Print and speak response
    print("Learny:", response)
    engine.say(response)
    engine.runAndWait()

    # Save conversation to chat history
    save_history(user_input, response)
