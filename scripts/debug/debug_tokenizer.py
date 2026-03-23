from transformers import AutoProcessor
import torch

model_id = "models/gemma-3n"
processor = AutoProcessor.from_pretrained(model_id)

text = "<image>\nTest instruction"
inputs = processor(text=text, return_tensors="pt")

print(f"Text: {text}")
print(f"Token IDs: {inputs.input_ids}")
print(f"Tokens: {processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])}")

# Check if special tokens contain <image>
print(f"Special tokens map: {processor.tokenizer.special_tokens_map}")
