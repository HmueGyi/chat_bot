import re

FACT_PATTERNS = [
    (re.compile(r"my name is (.+)", re.IGNORECASE), "name"),
    (re.compile(r"i am (.+)", re.IGNORECASE), "identity"),
    (re.compile(r"i'm (.+)", re.IGNORECASE), "identity"),
    (re.compile(r"my lover is (.+)", re.IGNORECASE), "lover"),
    (re.compile(r"i live in (.+)", re.IGNORECASE), "location"),
    (re.compile(r"my favorite (?:food|dish) is (.+)", re.IGNORECASE), "favorite_food"),
    (re.compile(r"my favorite (?:color) is (.+)", re.IGNORECASE), "favorite_color"),
]

def detect_fact(text):
    for pattern, key in FACT_PATTERNS:
        match = pattern.search(text)
        if match:
            return key, match.group(1).strip()
    return None
