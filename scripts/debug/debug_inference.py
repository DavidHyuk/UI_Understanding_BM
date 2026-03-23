
import sys
import os
import torch

# Add current directory to path so scripts/src/common/utils can be imported
sys.path.append(os.getcwd())

from scripts.src.common.utils import load_model_and_processor

def run_debug():
    print("Loading model...")
        print("Model directory not found.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model, processor = load_model_and_processor(model_id, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        # return # Let it fail or handle

    print("Running inference...")
    # Create simple prompt
    start_ins = "<ctrl99>user\n"
    end_ins = "<ctrl100>\n"
    start_res = "<ctrl99>model\n"
    prompt_text = "Hello"
    
    prompt_parts = [
        f"{start_ins}{prompt_text}{end_ins}{start_res}"
    ]
    
    try:
        with torch.no_grad():
            # model is now a wrapper, access underlying model
            out = model.model.generate_text_with_vision_input(prompt_parts, num_steps=20)
        print(f"Generated Output: '{out}'")
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()
