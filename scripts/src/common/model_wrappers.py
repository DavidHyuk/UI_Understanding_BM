import os
import sys
import torch
from abc import ABC, abstractmethod
import transformers
from transformers import AutoProcessor, AutoModelForMultimodalLM

# Use relative path from this file to ensure robustness regardless of CWD
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_src_dir = os.path.abspath(os.path.join(current_dir, '..'))
if scripts_src_dir not in sys.path:
    sys.path.append(scripts_src_dir)

class BaseModelWrapper(ABC):
    
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.load()

    @abstractmethod
    def load(self):
        """Loads the model and processor."""
        pass

    @abstractmethod
    def generate_content(self, prompt_text, image, **kwargs):
        """
        Generates content based on text prompt and image.
        
        Args:
            prompt_text (str): The text prompt.
            image (PIL.Image): The input image.
            **kwargs: Additional generation parameters (e.g., max_new_tokens).
            
        Returns:
            str: The generated text.
        """
        pass
    
    def get_processor(self):
        return self.processor

class HuggingFaceWrapper(BaseModelWrapper):
    """Wrapper for standard HuggingFace models (e.g., Gemma 3n)."""

    def load(self):
        is_mlx = getattr(self, "use_mlx", False)
        
        dtype = torch.bfloat16
        if str(self.device) == "mps":
            dtype = torch.float16
            
        print(f"Loading HuggingFace model: {self.model_path} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True
        )

    def generate_content(self, prompt_text, image, **kwargs):
        # Default parameters
        max_new_tokens = kwargs.get('max_new_tokens', 256) # standard default
        do_sample = kwargs.get('do_sample', False)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        # Apply chat template
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        
        target_dtype = torch.float16 if str(self.device) == "mps" else torch.bfloat16
        inputs = {k: v.to(target_dtype) if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=kwargs.get('top_p'),
                top_k=kwargs.get('top_k'),
                repetition_penalty=kwargs.get('repetition_penalty', 1.0)
            )
            
        input_len = inputs["input_ids"].shape[1]
        
        # SROIE hack: check if we should skip special tokens (can be passed via kwargs or we make it generic)
        # For now, let's stick to default behavior, maybe allow skip_special_tokens in kwargs
        skip_special_tokens = kwargs.get('skip_special_tokens', True)
        
        generated_text = self.processor.decode(generated_ids[0][input_len:], skip_special_tokens=skip_special_tokens)
        generated_text = generated_text.replace("<end_of_turn>", "").replace("<eos>", "").strip()
        return generated_text

class MLXModelWrapper(BaseModelWrapper):
    """Wrapper for Apple MLX framework models (e.g., mlx-vlm)."""
    
    def load(self):
        try:
            import mlx.core as mx
            from mlx_vlm import load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import generate
        except ImportError:
            raise ImportError("MLX backend requested, but mlx or mlx_vlm is not installed. Run `pip install mlx mlx-vlm`.")

        print(f"Loading MLX model: {self.model_path}...")
        self.model, self.processor = load(self.model_path)
        
        # Save a reference to the needed functions
        self.apply_chat_template = apply_chat_template
        self.mlx_generate = generate

    def generate_content(self, prompt_text, image, **kwargs):
        max_new_tokens = kwargs.get('max_new_tokens', 256)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        # Apply MLX chat template
        prompt = self.apply_chat_template(self.processor, self.model.config, messages)
        
        # MLX generate implementation
        from mlx_vlm.utils import generate
        # MLX takes image paths or URLs natively, but since we receive a PIL image we might need to save it or handle it.
        # mlx_vlm often can handle PIL images directly or lists of PIL images depending on the version.
        try:
            generated_text = generate(
                self.model,
                self.processor,
                prompt,
                [image],
                max_tokens=max_new_tokens,
                verbose=False
            )
        except Exception as e:
            # Fallback for older mlx-vlm if they need paths
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png") as temp_img:
                image.save(temp_img.name)
                generated_text = generate(
                    self.model,
                    self.processor,
                    prompt,
                    [temp_img.name],
                    max_tokens=max_new_tokens,
                    verbose=False
                )
        
        generated_text = generated_text.replace("<end_of_turn>", "").replace("<eos>", "").strip()
        return generated_text

class ModelFactory:
    """Factory to create the appropriate model wrapper."""
    
    @staticmethod
    def get_model(model_id, device):
        # Intelligent device selection logic
        # Skip if running in DDP mode (process group initialized) to avoid overriding rank-based device assignment
        is_ddp = False
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_ddp = True

        target_device = device
        
        # Only apply heuristic if NOT in DDP mode
        if not is_ddp and str(device).startswith("cuda"):
            if torch.cuda.device_count() > 1:
                if "gemma-3n" in model_id:
                    target_device = "cuda:1"
                    print(f"Assigning Gemma 3n to {target_device}")

        return HuggingFaceWrapper(model_id, target_device)

