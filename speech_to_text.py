import random

greetings = ["Hi", "Hello", "Hey", "Good morning", "Good evening"]
questions = [
    "How are you?", "What are you doing?", "How's your day?", "Are you busy today?", "What's up?"
]
activities = [
    "reading a book", "watching TV", "cooking lunch", "going for a walk",
    "working on my project", "listening to music", "playing games", "shopping"
]
replies = [
    "That sounds fun!", "Cool!", "Nice!", "Interesting!", "Wow!", "I like that!", "Tell me more."
]
foods = ["pizza", "pasta", "salad", "coffee", "tea", "sandwich", "cake"]

lines = []

for i in range(250):  # 250 exchanges will give 500+ lines (You + Learny)
    g = random.choice(greetings)
    q = random.choice(questions)
    act = random.choice(activities)
    rep = random.choice(replies)
    food = random.choice(foods)

    # Conversation 1
    lines.append(f"You: {g}, {q}")
    lines.append(f"Learny: I'm good! How about you?")
    lines.append(f"You: I'm {act}.")
    lines.append(f"Learny: {rep}")

    # Conversation 2 (food)
    lines.append(f"You: I am eating {food}.")
    lines.append(f"Learny: Yum! I love {food} too.")
    
    # Conversation 3 (weather)
    weather = random.choice(["sunny", "rainy", "cloudy", "cold", "hot"])
    lines.append(f"You: The weather is {weather} today.")
    lines.append(f"Learny: Oh, {weather} weather is nice!")

# Save to memory.txt
with open("memory.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("✅ memory.txt generated with", len(lines), "lines.")
