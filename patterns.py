import re

FACT_PATTERNS = [
    # Basic friendly patterns
    (re.compile(r"my name is (.+)", re.IGNORECASE), "name"),
    (re.compile(r"i live in (.+)", re.IGNORECASE), "location"),
    (re.compile(r"i feel (.+)", re.IGNORECASE), "feeling"),
    (re.compile(r"my hobby is (.+)", re.IGNORECASE), "hobby"),
    (re.compile(r"i like to eat (.+)", re.IGNORECASE), "likes_to_eat"),
    (re.compile(r"my favorite food is (.+)", re.IGNORECASE), "favorite_food"),
    (re.compile(r"my favorite color is (.+)", re.IGNORECASE), "favorite_color"),
    (re.compile(r"my favorite song is (.+)", re.IGNORECASE), "favorite_song"),
    (re.compile(r"my favorite movie is (.+)", re.IGNORECASE), "favorite_movie"),

    # Pet-like and activity patterns
    (re.compile(r"my favorite snack is (.+)", re.IGNORECASE), "favorite_snack"),
    (re.compile(r"we should go for a walk in (.+)", re.IGNORECASE), "walk_spot"),
    (re.compile(r"let's play (.+)", re.IGNORECASE), "play_game"),
    (re.compile(r"my favorite animal is (.+)", re.IGNORECASE), "favorite_animal"),
    (re.compile(r"i have a pet named (.+)", re.IGNORECASE), "other_pet"),
    (re.compile(r"(.+) makes me happy", re.IGNORECASE), "makes_happy"),
    (re.compile(r"(.+) makes me sad", re.IGNORECASE), "makes_sad"),

    # General life patterns (less personal)
    (re.compile(r"my birthday is on (.+)", re.IGNORECASE), "birthday"),
    (re.compile(r"i work as (.+)", re.IGNORECASE), "job"),
    (re.compile(r"my dream is to (.+)", re.IGNORECASE), "dream"),
    (re.compile(r"i want to (.+)", re.IGNORECASE), "goal"),
    (re.compile(r"i am from (.+)", re.IGNORECASE), "hometown"),
    (re.compile(r"i usually wake up at (.+)", re.IGNORECASE), "wake_up_time"),
]

def detect_fact(text):
    """Tries to find a fact in the user's input using regex patterns."""
    for pattern, key in FACT_PATTERNS:
        match = pattern.search(text)
        if match:
            # For patterns like "(.+) makes me happy", group(1) is the subject.
            # For others, it's the main object. This logic works for both.
            return key, match.group(1).strip()
    return None