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
        print(f"Loading HuggingFace model: {self.model_path} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
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
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
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
