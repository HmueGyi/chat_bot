import os
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

def prepare_dataset(memory_file="memory.txt"):
    """Reads conversation history and prepares it for training."""
    if not os.path.exists(memory_file) or os.path.getsize(memory_file) == 0:
        return None
        
    with open(memory_file, "r", encoding="utf-8") as f:
        # Read lines and filter out any empty ones
        lines = [line for line in f.read().splitlines() if line.strip()]

    # Group lines into conversation turns (You: ... Learny: ...)
    samples, buffer = [], []
    for line in lines:
        buffer.append(line)
        if line.startswith("Learny:"):
            samples.append("\n".join(buffer))
            buffer = []
            
    if not samples:
        return None

    print(f"Found {len(samples)} conversation samples in {memory_file}.")
    return Dataset.from_dict({"text": samples})

def fine_tune(memory_file="memory.txt", output_dir="fine-tuned", max_steps=500):
    """Loads a model and fine-tunes it on the conversation history."""
    print("⚡ Fine-tuning process started...")

    dataset = prepare_dataset(memory_file)
    if dataset is None or len(dataset) < 2:
        print("⚠️ Not enough memory available for training. Skipping.")
        return

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        """Tokenize the text data."""
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=128
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Split dataset into training and validation sets
    if len(dataset) < 10: # Avoid splitting if dataset is too small
        train_dataset = dataset
        val_dataset = None
        evaluation_strategy = "no"
    else:
        train_size = int(0.9 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, len(dataset)))
        evaluation_strategy = "steps"

    # Load model, starting from a pre-trained one if it exists
    try:
        model = GPT2LMHeadModel.from_pretrained(output_dir)
        print("✅ Loaded existing fine-tuned model from disk.")
    except (OSError, ValueError):
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        print("✅ No existing model found. Starting fresh fine-tune from 'gpt2-medium'.")

    # Use a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1, # Keep batch size low for background training
        save_steps=100,
        save_total_limit=2,
        logging_steps=20,
        evaluation_strategy=evaluation_strategy,
        eval_steps=100 if val_dataset else None,
        max_steps=max_steps,
        learning_rate=5e-5,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator, # Add the data collator
    )

    print("🚀 Starting model training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Fine-tuning complete. Model saved to '{output_dir}'.")