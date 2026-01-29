import argparse
import os
import sys
import inspect
import platform
from mediapipe.tasks.python.genai import converter

def convert_model(input_dir, output_dir, model_type, backend="cpu"):
    """
    Converts a Hugging Face model to MediaPipe binary format.
    """
    print(f"Starting conversion for model in: {input_dir}")
    print(f"Target Model Type: {model_type}")
    print(f"Output Directory: {output_dir}")

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.bin")

    # Determine checkpoint format (safetensors or pytorch)
    ckpt_format = "safetensors"
    if os.path.exists(os.path.join(input_dir, "pytorch_model.bin")):
        ckpt_format = "pytorch"
    
    print(f"Detected checkpoint format: {ckpt_format}")

    # Smart Configuration
    try:
        sig = inspect.signature(converter.ConversionConfig)
        params = {}
        
        # Standard params
        params['input_ckpt'] = input_dir
        params['ckpt_format'] = ckpt_format
        params['model_type'] = model_type
        params['backend'] = backend
        params['output_dir'] = output_dir

        # Version-specific params detection
        if 'combine_file_type' in sig.parameters:
            params['combine_file_type'] = "one_file"
        elif 'combine_file_only' in sig.parameters:
            params['combine_file_only'] = True
        
        if 'output_tflite_file' in sig.parameters:
            params['output_tflite_file'] = output_path

        print(f"Configuring with detected params: {list(params.keys())}")
        config = converter.ConversionConfig(**params)

        print("Running conversion... This may take a while.")
        converter.convert_checkpoint(config)
        print(f"Conversion complete! Output saved in: {output_dir}")

    except Exception as e:
        print(f"Error during conversion: {e}")
        
        # Specific advice for aarch64 users
        if "GenerateCpuTfLite" in str(e):
            print("\n[CRITICAL ERROR] Platform Incompatibility Detected")
            print(f"Current Platform: {platform.machine()} ({sys.platform})")
            if platform.machine() in ['aarch64', 'arm64']:
                print("The MediaPipe GenAI Converter requires x86_64 architecture (Intel/AMD) to generate the model binary.")
                print("Please run this script on a standard PC or Google Colab, then copy the output '.bin' file to this device.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF Model to MediaPipe Bin")
    parser.add_argument("--input_model_dir", type=str, required=True, help="Path to Hugging Face model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the converted model")
    parser.add_argument("--model_type", type=str, default="GEMMA_2B", 
                        help="Model architecture type")
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "gpu"], help="Target backend optimization")

    args = parser.parse_args()
    
    convert_model(args.input_model_dir, args.output_dir, args.model_type, args.backend)
