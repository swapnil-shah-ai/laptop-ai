from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading distilled adapter...")
model = PeftModel.from_pretrained(base, "./swapnil-distilled-lora")
model = model.merge_and_unload()

prompt = "What was the un-app architecture?"
inputs = tokenizer(prompt, return_tensors="pt")
print("Generating...")
output = model.generate(**inputs, max_new_tokens=200, temperature=0.3, do_sample=True, pad_token_id=tokenizer.pad_token_id)
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n" + "=" * 50)
print("DISTILLED MODEL OUTPUT:")
print("=" * 50)
print(result)
print("=" * 50)
