# train_learny.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os

# Disable W&B logging
os.environ["WANDB_DISABLED"] = "true"

# Use smaller model
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# Prepare dataset
def load_dataset(file_path, tokenizer, block_size=64):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,  # smaller block for less memory
        overwrite_cache=True
    )

train_dataset = load_dataset("learny_data.txt", tokenizer)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./learny_model",
    overwrite_output_dir=True,
    num_train_epochs=100,             # reduce epochs
    per_device_train_batch_size=1,  # reduce batch size
    save_steps=200,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save model & tokenizer
trainer.save_model("./learny_model")
tokenizer.save_pretrained("./learny_model")
