import os
import folder_paths
from pathlib import Path

class GGUFLoader:
    @classmethod
    def INPUT_TYPES(s):
        # Find all .gguf files in models directory recursively
        models_dir = folder_paths.models_dir
        gguf_files = []
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith('.gguf'):
                    # Get relative path from models_dir
                    rel_path = os.path.relpath(os.path.join(root, file), models_dir)
                    gguf_files.append(rel_path)
        
        return {
            "required": {
                "gguf_name": (gguf_files, {"default": gguf_files[0] if gguf_files else ""}),
            },
            "optional": {
                "gguf_path_override": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_gguf"
    CATEGORY = "🦙 ComfyUI-LLama"

    def load_gguf(self, gguf_name, gguf_path_override=""):
        # Use override path if provided, otherwise use selected file
        if gguf_path_override and os.path.exists(gguf_path_override):
            gguf_path = gguf_path_override
        else:
            gguf_path = os.path.join(folder_paths.models_dir, gguf_name)
        
        if not gguf_path or not os.path.exists(gguf_path):
            raise ValueError(f"GGUF file not found: {gguf_path}")
        
        # Return a dict with path for compatibility
        return ({"path": gguf_path},)