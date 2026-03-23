import os
from safetensors import safe_open
import json

folder_path = "./models/gemma-3n"
safetensors_files = [f for f in os.listdir(folder_path) if f.endswith('.safetensors')]
safetensors_files.sort()

print(f"Checking safetensors in {folder_path}...\n")

for file_name in safetensors_files:
    file_path = os.path.join(folder_path, file_name)
    print(f"=== {file_name} ===")
    
    dtypes = set()
    
    try:
        # Read just the header metadata (very fast, doesn't load tensors)
        with open(file_path, "rb") as f:
            header_size_bytes = f.read(8)
            header_size = int.from_bytes(header_size_bytes, "little")
            header_json = f.read(header_size).decode("utf-8")
            header = json.loads(header_json)
            
            for key, val in header.items():
                if key != "__metadata__":
                    dtypes.add(val.get("dtype", "unknown"))
                    
        print(f"Data types present: {', '.join(dtypes)}")
        if all(dtype == "F16" for dtype in dtypes):
            print("Status: All tensors are FP16 (F16).")
        else:
            print("Status: Contains mixed or non-FP16 types.")
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        
    print()
