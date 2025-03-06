import json
import pandas as pd
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import os

# Set the output directory to D drive
OUTPUT_DIR = "D:/finetuned_llama"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load JSON dataset
with open('processed_dataset_corrected.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Format data for LLaMA training
def format_conversations(example):
    return f"<s>[INST] {example['instruction']} [/INST] {example['response']} </s>"

df['text'] = df.apply(format_conversations, axis=1)

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df[['text']])

# Load smaller LLaMA model (7B parameters)
model_name = "huggyllama/llama-7b"
print(f"Loading model: {model_name}")

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto'
)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=256  # Reduced for memory efficiency
    )

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training setup with laptop-friendly parameters
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy='steps',
    eval_steps=100,
    learning_rate=1e-5,
    per_device_train_batch_size=2,  # Reduced batch size
    gradient_accumulation_steps=4,   # Accumulate gradients
    num_train_epochs=1,             # Reduced epochs
    weight_decay=0.01,
    save_steps=100,
    save_total_limit=2,
    logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
    logging_steps=50,
    fp16=True,                      # Use mixed precision
    push_to_hub=False,
    gradient_checkpointing=True,    # Enable gradient checkpointing
)

print("Setting up trainer...")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting training...")
try:
    trainer.train()
    print("Training completed successfully!")
    
    # Save the model to D drive
    print(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model and tokenizer saved successfully!")
    
except Exception as e:
    print(f"An error occurred during training: {str(e)}")