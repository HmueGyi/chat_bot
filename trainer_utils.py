import os
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

def prepare_dataset(memory_file="memory.txt"):
    if not os.path.exists(memory_file):
        return None
    with open(memory_file, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    samples, buffer = [], []
    for line in lines:
        buffer.append(line)
        if line.startswith("Learny:"):
            samples.append("\n".join(buffer))
            buffer = []
    if not samples:
        return None

    return Dataset.from_dict({"text": samples})

def fine_tune(memory_file="memory.txt", output_dir="fine-tuned", max_steps=500):
    print("⚡ Fine-tuning started...")

    dataset = prepare_dataset(memory_file)
    if dataset is None:
        print("No memory available for training.")
        return

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    if os.path.exists(output_dir):
        try:
            model = GPT2LMHeadModel.from_pretrained(output_dir)
            print("Loaded existing fine-tuned model.")
        except Exception:
            model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            print("Starting fresh fine-tune.")
    else:
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        max_steps=max_steps,
        learning_rate=5e-5,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Fine-tuning complete. Model saved to", output_dir)
