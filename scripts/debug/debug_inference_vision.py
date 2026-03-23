
import sys
import os
import torch
from PIL import Image

sys.path.append(os.getcwd())
from scripts.src.common.utils import load_model_and_processor, get_dataset, DATASET_CONFIGS

def run_vision_debug():
    print("Loading model...")
        print("Model directory not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model, processor = load_model_and_processor(model_id, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Loading Dataset (SROIE)...")
    try:
        dataset = get_dataset("sroie")
        example = dataset[0]
        image = example["image"]
        dataset_config = DATASET_CONFIGS["sroie"]
        prompt_text = dataset_config["prompt_fn"](example)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    print(f"Prompt: {prompt_text}")
    
    # Prepare Input logic from demo/app.py
    start_ins = "<ctrl99>user\n"
    end_ins = "<ctrl100>\n"
    start_res = "<ctrl99>model\n"
    
    # Process Image
    # processor.image_processor might need image to be PIL
    pixel_values = processor.image_processor(image, return_tensors="pt").pixel_values
    image_tensor = pixel_values[0].permute(1, 2, 0) # [C, H, W] -> [H, W, C]
    # model is wrapper, check underlying model
    if hasattr(model.model, 'dtype') and model.model.dtype == torch.bfloat16:
        image_tensor = image_tensor.to(torch.bfloat16)
        
    prompt_parts = [
        f"{start_ins}{prompt_text}",
        ("image", image_tensor),
        f"{end_ins}{start_res}"
    ]
    
    print("Running Inference...")
    import time
    start_t = time.time()
    try:
        with torch.no_grad():
            # model is now a wrapper, access underlying model
            out = model.model.generate_text_with_vision_input(prompt_parts, num_steps=50)
        end_t = time.time()
        print(f"Output: '{out}'")
        print(f"Inference Time: {end_t - start_t:.2f}s")
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_vision_debug()
