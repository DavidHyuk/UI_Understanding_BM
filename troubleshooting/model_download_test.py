from huggingface_hub import HfApi
api = HfApi()
try:
    model_info = api.model_info("google/gemma-3n-E4B-it")
    print(f"Model exists: {model_info.id}")
except Exception as e:
    print(f"Error: {e}")