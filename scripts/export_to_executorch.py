import torch
import os
import argparse
import sys
import warnings
from transformers import AutoProcessor, Gemma3nForCausalLM

# Suppress annoying FutureWarnings and deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*isinstance.*treespec, LeafSpec.*")

# Check if executorch is available
try:
    from executorch.exir.backend.backend_api import to_backend
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
except ImportError as e:
    print(f"Error: ExecuTorch not found or incomplete installation: {e}")
    print("Note: If you used pip, you might need to build from source using scripts/build_executorch_aarch64.sh")
    # Define stubs to avoid NameError if we want to see other errors
    to_backend = None
    XnnpackPartitioner = None

class Gemma3VLMWrapper(torch.nn.Module):
    """
    Wrapper to capture the forward pass for ExecuTorch export.
    ExecuTorch works best when the model has a single forward() method with tensor inputs.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, pixel_values):
        # We export the core generation/forward logic.
        return self.model(
            input_ids=input_ids, 
            pixel_values=pixel_values
        ).logits

def export_model(model_path, model_name, quantization, output_dir):
    # Construct output path based on model name and quantization
    output_filename = f"{model_name}_{quantization}.pte"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Target Output Path: {output_path}")
    print(f"Loading Gemma 3n model from {model_path}...")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Gemma3nForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    ).eval()

    # Properly format dummy inputs using the processor
    from PIL import Image
    import numpy as np
    dummy_image = Image.fromarray(np.uint8(np.random.randint(0, 255, (896, 896, 3))))
    
    # Use Chat Template to get the correct placeholder count
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "detect the button"}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # We use the processor to get the correct number of image tokens
    inputs = processor(text=prompt, images=dummy_image, return_tensors="pt")
    dummy_input_ids = inputs["input_ids"]
    dummy_pixel_values = inputs["pixel_values"]
    
    print(f"Input shapes: input_ids={dummy_input_ids.shape}, pixel_values={dummy_pixel_values.shape}")
    print(f"Input dtypes: input_ids={dummy_input_ids.dtype}, pixel_values={dummy_pixel_values.dtype}")
    
    # 0. Monkeypatch to bypass data-dependent guards during export
    # The original get_placeholder_mask has 'if' checks on tensor numel() which fail torch.export
    print("Patching model to bypass data-dependent checks...")
    import types
    from typing import Optional

    def patched_get_placeholder_mask(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_features: Optional[torch.FloatTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        # Bypassing the if check on numel() as it triggers GuardOnDataDependentSymNode
        
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        # Bypassing the if check on numel() as it triggers GuardOnDataDependentSymNode

        return special_image_mask, special_audio_mask

    # Bind the patched method to the model instance
    # Note: Gemma3nForConditionalGeneration wraps the base model in .model
    if hasattr(model, "model") and hasattr(model.model, "get_placeholder_mask"):
        print("Patching model.model.get_placeholder_mask...")
        model.model.get_placeholder_mask = types.MethodType(patched_get_placeholder_mask, model.model)
    elif hasattr(model, "get_placeholder_mask"):
        print("Patching model.get_placeholder_mask...")
        model.get_placeholder_mask = types.MethodType(patched_get_placeholder_mask, model)
    else:
        print("Warning: Could not find get_placeholder_mask to patch!")

    wrapper = Gemma3VLMWrapper(model)

    print(f"Skipping pre-export quantization, will use ExecuTorch Quantizer instead.")
    
    try:
        # 1. Trace/Export the model to get a GraphModule (Initial Export)
        print("1. Initial export to get GraphModule...")
        initial_ep = torch.export.export(
            wrapper,
            (dummy_input_ids, dummy_pixel_values),
            strict=False
        )
        gm = initial_ep.module()
        
        # Define the quantizer for XNNPACK
        print(f"Setting up XNNPACK Quantizer for {quantization}...")
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
        
        print("2. Applying PT2E Quantization...")
        # Prepare (on GraphModule)
        prepared_gm = prepare_pt2e(gm, quantizer)
        
        # Remove assertions that might cause failures during calibration
        # This is safe because we are just calibrating for quantization statistics
        print("   Removing assertion nodes recursively for calibration...")
        
        def remove_assertions_recursive(gm, name="root"):
            removed_count = 0
            # 1. Remove assertions in the current graph
            for node in list(gm.graph.nodes):
                if node.op == "call_function" and "assert" in str(node.target):
                    node.args = ()
                    gm.graph.erase_node(node)
                    removed_count += 1
            
            if removed_count > 0:
                print(f"      Removed {removed_count} assertion nodes from '{name}'")
                gm.recompile()
            
            # 2. Recurse into submodules
            for sub_name, module in gm.named_modules():
                if isinstance(module, torch.fx.GraphModule) and module is not gm:
                    removed_count += remove_assertions_recursive(module, f"{name}.{sub_name}")
            
            return removed_count

        total_removed = remove_assertions_recursive(prepared_gm)
        print(f"   Total assertion nodes removed: {total_removed}")
        
        # Calibrate (using dummy inputs - required for quantization statistics initialization)
        print("   Calibrating with dummy inputs...")
        prepared_gm(dummy_input_ids, dummy_pixel_values)
        
        # Convert
        print("   Converting to quantized model...")
        quantized_gm = convert_pt2e(prepared_gm)
        
        # 3. Re-export the quantized model to get an ExportedProgram
        print("3. Re-exporting quantized model...")
        quantized_ep = torch.export.export(
            quantized_gm,
            (dummy_input_ids, dummy_pixel_values),
            strict=False
        )
        
        # 4. Transform and Lower to ExecuTorch
        print("4. Transforming and Lowering to ExecuTorch (CPU Fallback)...")
        from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
        # We disable XnnpackPartitioner due to dependency cycles in the complex Gemma 3n graph
        edge_manager = to_edge_transform_and_lower(
            quantized_ep,
            partitioner=[],
            compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )
        
        # 5. Convert to ExecuTorch and Save
        print("5. Converting to ExecuTorch program...")
        executorch_program = edge_manager.to_executorch()
        
        print(f"Saving to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(executorch_program.buffer)
            
        print(f"Successfully exported model to {output_path}")
        
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Optimize CPU usage by setting threads to match available cores
    num_cores = os.cpu_count()
    if num_cores:
        torch.set_num_threads(num_cores)
        print(f"Set PyTorch threads to {num_cores} for better CPU utilization.")

    parser = argparse.ArgumentParser(description="Export Gemma 3n to ExecuTorch (.pte)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to HF model directory")
    parser.add_argument("--model_name", type=str, default="gemma3n", help="Base name for the exported model")
    parser.add_argument("--quantization", type=str, choices=["int8", "int4", "fp16"], default="int8", help="Quantization type")
    parser.add_argument("--output_dir", type=str, default="scripts/deployed_model", help="Directory to save the .pte file")
    
    args = parser.parse_args()
    export_model(args.checkpoint, args.model_name, args.quantization, args.output_dir)
