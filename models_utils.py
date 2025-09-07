# models_utils.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def load_policy_model(model_path="./learny_model"):
    """Load fine-tuned GPT-2 model and tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def generate_candidates(tokenizer, model, prompt, max_new_tokens=50):
    """
    Generate ONE meaningful sentence from GPT-2 / DistilGPT-2.
    Returns a single string.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove prompt from generated text
    text = text[len(prompt):].strip()
    # Take only the first sentence
    if "." in text:
        text = text.split(".")[0].strip() + "."
    # Optional: remove repeated tokens
    text = " ".join(dict.fromkeys(text.split()))
    return text
