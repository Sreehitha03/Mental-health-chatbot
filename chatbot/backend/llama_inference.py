import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load fine-tuned model from D drive
MODEL_DIR = "D:/finetuned_llama"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    print(f"Loading model from {MODEL_DIR}")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto'
    )
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")

def generate_response(prompt, max_length=150):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"